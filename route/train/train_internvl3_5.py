import os
import inspect
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from typing import List
import logging
from collections import defaultdict
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import wandb

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from route.prompt import ROUTER_PROMPT
from route.train.router_common import (
    LABELS,
    LABEL_TO_ID,
    make_label_key,
    normalize_label_parts as normalize_router_label_parts,
    save_router_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _supports_inputs_embeds(model) -> bool:
    try:
        return "inputs_embeds" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _get_internvl_model(model):
    base_model = getattr(model, "base_model", None)
    wrapped = getattr(base_model, "model", None) if base_model is not None else None
    return wrapped if wrapped is not None else model


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


class MultiLabelRouteDataset(Dataset):
    """Dataset for multi-label routing classification."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        num_image_token: int,
        max_length: int = 512,
        input_size: int = 448,
        max_num: int = 12,
    ):
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token
        self.max_length = max_length
        self.input_size = input_size
        self.max_num = max_num

        logger.info(f"Loading data from {data_path}")
        with open(data_path, "r") as f:
            raw_data = json.load(f)

        logger.info(f"Loaded {len(raw_data)} samples")
        raw_data = [d for d in raw_data if "gt_retrieval" in d and "question" in d]
        logger.info(f"Filtered to {len(raw_data)} samples with gt_retrieval and question")

        if len(raw_data) == 0:
            raise ValueError("No valid samples found in training data!")

        self.data = []
        for item in raw_data:
            labels = self.normalize_label_parts(item["gt_retrieval"])
            self.data.append(
                {
                    "question": item["question"],
                    "query_image": item.get("query_image"),
                    "labels": labels,
                }
            )

        logger.info(f"Loaded {len(self.data)} multi-label samples")

        label_counts = defaultdict(int)
        for item in self.data:
            for label in item["labels"]:
                label_counts[label] += 1
        total_labels = sum(label_counts.values())

        logger.info("Label distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / total_labels if total_labels else 0.0
            logger.info(f"  {label}: {count} ({pct:.1f}%)")

    def __len__(self):
        return len(self.data)

    def normalize_label_parts(self, label: str) -> List[str]:
        return normalize_router_label_parts(label)

    def __getitem__(self, idx):
        item = self.data[idx]

        if "question" not in item or "labels" not in item:
            raise ValueError(f"Sample at index {idx} missing required fields")

        question = item["question"].strip()
        if not question:
            raise ValueError(f"Empty question at index {idx}")

        labels = item["labels"]
        label_vec = [0.0] * len(LABELS)
        for label in labels:
            label_vec[LABEL_TO_ID[label]] = 1.0

        prompt = ROUTER_PROMPT.format(query=question)

        message = ""
        pixel_values = None

        if item.get("query_image"):
            query_image_path = item["query_image"]
            try:
                pixel_values = load_image(query_image_path, input_size=self.input_size, max_num=self.max_num)
                message += "<image>\n"
            except Exception as e:
                logger.warning(f"Failed to load image {query_image_path}: {e}")

        message += prompt

        if pixel_values is not None:
            num_patches = pixel_values.size(0)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches) + IMG_END_TOKEN
            message = message.replace("<image>", image_tokens, 1)
            image_flags = torch.ones((num_patches, 1), dtype=torch.long)
        else:
            pixel_values = torch.zeros((1, 3, self.input_size, self.input_size), dtype=torch.float32)
            image_flags = torch.zeros((1, 1), dtype=torch.long)

        full_text = f"User: {message}\nAssistant:"

        tokens = self.tokenizer(
            full_text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )

        result = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": label_vec,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }
        return result


class InternVLMultiLabelClassifier(nn.Module):
    """Classifier head on top of InternVL3.5."""

    def __init__(self, backbone, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            text_cfg = getattr(backbone.config, "text_config", None)
            hidden_size = getattr(text_cfg, "hidden_size", None)
        if hidden_size is None:
            llm_cfg = getattr(backbone.config, "llm_config", None)
            hidden_size = getattr(llm_cfg, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not determine hidden size for InternVL classifier.")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.config = backbone.config
        self.accepts_loss_kwargs = False

    def _get_base_model(self):
        return _get_internvl_model(self.backbone)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        base_model = self._get_base_model()
        try:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        except TypeError:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs,
            )

        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None and getattr(outputs, "hidden_states", None) is not None:
            hidden_states = outputs.hidden_states[-1]
        if hidden_states is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden_states = outputs[0]
        if hidden_states is None:
            raise ValueError("Could not extract hidden states from InternVL outputs.")

        if attention_mask is not None and attention_mask.ndim == 2:
            seq_lengths = attention_mask.sum(dim=1) - 1
        else:
            pad_token_id = getattr(self.config, "pad_token_id", None)
            if pad_token_id is not None and input_ids is not None:
                seq_lengths = (input_ids != pad_token_id).sum(dim=1) - 1
            else:
                seq_lengths = torch.full(
                    (hidden_states.size(0),),
                    hidden_states.size(1) - 1,
                    device=hidden_states.device,
                    dtype=torch.long,
                )

        seq_lengths = seq_lengths.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, seq_lengths]
        pooled = self.dropout(pooled)

        if self.classifier.weight.device != pooled.device:
            self.classifier = self.classifier.to(pooled.device)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device).float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return {"loss": loss, "logits": logits}


class InternVLDataCollator:
    """Pads text and concatenates image patches for InternVL."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
            )

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        pixel_values_list = []
        image_flags_list = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            label = feature["labels"]

            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()

            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(label)

            pixel_values_list.append(feature["pixel_values"])
            image_flags_list.append(feature["image_flags"])

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.float32)

        batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        batch["image_flags"] = torch.cat(image_flags_list, dim=0)

        return batch


def compute_metrics_multi_label(eval_pred, label_threshold: float = 0.8):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    probs = 1 / (1 + np.exp(-logits))
    labels = labels.astype(int)
    preds = (probs >= label_threshold).astype(int)

    subset_accuracy = (preds == labels).all(axis=1).mean()

    tp = (preds & labels).sum(axis=0)
    fp = (preds & (1 - labels)).sum(axis=0)
    fn = ((1 - preds) & labels).sum(axis=0)

    precision_per_label = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall_per_label = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1_per_label = np.divide(
        2 * precision_per_label * recall_per_label,
        precision_per_label + recall_per_label,
        out=np.zeros_like(precision_per_label, dtype=float),
        where=(precision_per_label + recall_per_label) > 0,
    )

    macro_precision = float(precision_per_label.mean()) if precision_per_label.size else 0.0
    macro_recall = float(recall_per_label.mean()) if recall_per_label.size else 0.0
    macro_f1 = float(f1_per_label.mean()) if f1_per_label.size else 0.0

    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "subset_accuracy": float(subset_accuracy),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Train InternVL3.5 multi-label router (multi-hot)")
    parser.add_argument("--model-name", type=str, default="OpenGVLab/InternVL3_5-1B", help="Model name or path")
    parser.add_argument("--train-data", type=str, default="route/train/data/train_data.json", help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="route/train/checkpoints/internvl3_5_1b", help="Output directory for model checkpoints")
    parser.add_argument("--num-train-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--input-size", type=int, default=448, help="Image input size")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max-num", type=int, default=12, help="Maximum number of image tiles")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--classifier-dropout", type=float, default=0.1, help="Dropout for classifier head")
    parser.add_argument("--label-threshold", type=float, default=0.8, help="Sigmoid probability threshold for metrics")
    parser.add_argument("--use-lora", action="store_true", default=True, help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA r parameter")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--wandb-project", type=str, default="internvl3_5-router", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Use Weights & Biases for logging")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        logger.info(f"Initialized wandb project: {args.wandb_project}")

    logger.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    backbone = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    if args.use_lora:
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        backbone = get_peft_model(backbone, lora_config)
        internvl_model = _get_internvl_model(backbone)
        if not _supports_inputs_embeds(internvl_model):
            logger.info("Patching PEFT forward to drop unsupported inputs_embeds")
            original_forward = backbone.base_model.forward
            try:
                vision_device = next(internvl_model.vision_model.parameters()).device
            except StopIteration:
                vision_device = None

            def patched_forward(*args, **kwargs):
                kwargs.pop("inputs_embeds", None)
                pixel_values = kwargs.get("pixel_values")
                image_flags = kwargs.get("image_flags")
                if pixel_values is not None and vision_device is not None and pixel_values.device != vision_device:
                    kwargs["pixel_values"] = pixel_values.to(vision_device)
                if image_flags is not None:
                    kwargs["image_flags"] = image_flags.cpu()
                return original_forward(*args, **kwargs)

            backbone.base_model.forward = patched_forward
        backbone.print_trainable_parameters()

    internvl_model = _get_internvl_model(backbone)
    internvl_model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    num_image_token = internvl_model.num_image_token

    model = InternVLMultiLabelClassifier(
        backbone=backbone,
        num_labels=len(LABELS),
        dropout=args.classifier_dropout,
    )
    model.train()

    logger.info("Creating dataset")
    full_dataset = MultiLabelRouteDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        num_image_token=num_image_token,
        max_length=args.max_length,
        input_size=args.input_size,
        max_num=args.max_num,
    )

    logger.info("Creating stratified train/eval split (90% train, 10% eval)")
    label_to_indices = defaultdict(list)
    for idx in range(len(full_dataset)):
        item = full_dataset.data[idx]
        label_key = make_label_key(item["labels"])
        label_to_indices[label_key].append(idx)

    train_indices = []
    eval_indices = []
    rng = np.random.default_rng(42)
    for label, indices in label_to_indices.items():
        shuffled_indices = rng.permutation(indices)
        split_point = int(0.9 * len(shuffled_indices))
        train_indices.extend(shuffled_indices[:split_point].tolist())
        eval_indices.extend(shuffled_indices[split_point:].tolist())
        logger.info(f"  {label}: {split_point} train, {len(shuffled_indices) - split_point} eval")

    train_dataset = Subset(full_dataset, train_indices)
    eval_dataset = Subset(full_dataset, eval_indices)
    logger.info(f"Total - Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    data_collator = InternVLDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_safetensors=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["wandb"] if args.use_wandb else ["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics_multi_label(p, args.label_threshold),
        processing_class=tokenizer,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if hasattr(model.backbone, "config"):
        model.backbone.config.save_pretrained(args.output_dir)
    if args.use_lora and hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(args.output_dir)

    save_router_config(
        args.output_dir,
        model_family="internvl3_5",
        classifier_dropout=args.classifier_dropout,
        label_threshold=args.label_threshold,
        use_lora=args.use_lora,
        input_size=args.input_size,
        max_num=args.max_num,
    )

    if args.use_wandb:
        wandb.finish()

    logger.info("Training completed!")


if __name__ == "__main__":
    import argparse

    main()
