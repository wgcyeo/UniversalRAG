import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from typing import List
import logging
from collections import defaultdict
import numpy as np
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


class MultiLabelRouteDataset(Dataset):
    """Dataset for multi-label routing classification."""

    def __init__(self, data_path: str):
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
            self.data.append({
                "question": item["question"],
                "query_image": item.get("query_image"),
                "labels": labels,
            })

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

        return {
            "question": question,
            "query_image": item.get("query_image"),
            "labels": item["labels"],
        }


class Qwen3VLMultiLabelClassifier(nn.Module):
    """Multi-label classifier head on top of Qwen3-VL."""

    def __init__(self, backbone, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = backbone.config.text_config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.config = backbone.config
        self.accepts_loss_kwargs = False

    def _get_base_model(self):
        if hasattr(self.backbone, "get_base_model"):
            base_model = self.backbone.get_base_model()
        else:
            base_model = self.backbone
        return base_model.model if hasattr(base_model, "model") else base_model

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        base_model = self._get_base_model()
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
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
        if self.classifier.weight.dtype != pooled.dtype:
            pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device).float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return {"loss": loss, "logits": logits}


class Qwen3VLMultiLabelCollator:
    """Batch-processes text/images for Qwen3-VL and builds multi-label targets."""

    def __init__(self, processor, image_size: int = 224, pad_to_multiple_of: int = 8):
        self.processor = processor
        self.image_size = image_size
        self.pad_to_multiple_of = pad_to_multiple_of
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    def make_image_content(self, image_path: str) -> dict:
        image_content = {"type": "image", "image": image_path}
        if self.image_size > 0:
            image_content.update(
                {
                    "resized_height": self.image_size,
                    "resized_width": self.image_size,
                }
            )
        return image_content

    def make_label_vector(self, labels: List[str]) -> List[float]:
        label_vec = [0.0] * len(LABELS)
        for label in labels:
            label_vec[LABEL_TO_ID[label]] = 1.0
        return label_vec

    def __call__(self, features):
        texts = []
        labels = []
        image_inputs = []

        for feature in features:
            prompt = ROUTER_PROMPT.format(query=feature["question"])
            user_content = []
            if feature.get("query_image"):
                user_content.append(self.make_image_content(feature["query_image"]))
            user_content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": user_content}]

            if feature.get("query_image"):
                try:
                    sample_images = process_vision_info(messages)[0]
                    if sample_images is not None:
                        image_inputs.extend(sample_images)
                except Exception as e:
                    logger.warning(f"Failed to load image {feature['query_image']}: {e}")
                    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            labels.append(self.make_label_vector(feature["labels"]))

        processor_kwargs = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.pad_to_multiple_of:
            processor_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
        if image_inputs:
            processor_kwargs["images"] = image_inputs

        batch = self.processor(**processor_kwargs)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
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
    parser = argparse.ArgumentParser(description="Train Qwen3-VL multi-label router (multi-hot)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Model name or path")
    parser.add_argument("--train-data", type=str, default="route/train/data/train_data.json", help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="route/train/checkpoints/qwen3_vl_2b", help="Output directory for model checkpoints")
    parser.add_argument("--num-train-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Square resize for query images; set 0 to use processor default")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--classifier-dropout", type=float, default=0.1, help="Dropout for classifier head")
    parser.add_argument("--label-threshold", type=float, default=0.8, help="Sigmoid probability threshold for metrics")
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True, help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA r parameter")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--wandb-project", type=str, default="qwen3-vl-router", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=True, help="Use Weights & Biases for logging")

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

    logger.info(f"Loading model and processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2",
    )

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
        backbone.print_trainable_parameters()

    model = Qwen3VLMultiLabelClassifier(
        backbone=backbone,
        num_labels=len(LABELS),
        dropout=args.classifier_dropout,
    )

    logger.info("Creating dataset")
    full_dataset = MultiLabelRouteDataset(
        data_path=args.train_data,
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

    data_collator = Qwen3VLMultiLabelCollator(
        processor=processor,
        image_size=args.image_size,
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
        processing_class=processor,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    base_model = model.backbone.get_base_model() if hasattr(model.backbone, "get_base_model") else model.backbone
    if hasattr(base_model, "config"):
        base_model.config.save_pretrained(args.output_dir)

    if args.use_lora and hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(args.output_dir)

    save_router_config(
        args.output_dir,
        model_family="qwen3_vl",
        classifier_dropout=args.classifier_dropout,
        label_threshold=args.label_threshold,
        use_lora=args.use_lora,
        image_size=args.image_size,
    )

    if args.use_wandb:
        wandb.finish()

    logger.info("Training completed!")


if __name__ == "__main__":
    import argparse

    main()
