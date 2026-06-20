import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from transformers.utils import WEIGHTS_NAME
from peft import LoraConfig, get_peft_model
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

    def __init__(self, data_path: str, tokenizer):
        self.tokenizer = tokenizer

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

    def compute_max_length(self):
        """Compute the maximum sequence length in the dataset."""
        logger.info("Computing maximum sequence length from dataset...")
        max_len = 0
        for idx in range(len(self.data)):
            item = self.data[idx]
            prompt = ROUTER_PROMPT.format(query=item["question"].strip())
            inputs = self.tokenizer(prompt, padding=False, truncation=False, return_tensors="pt")
            seq_len = inputs["input_ids"].size(1)
            max_len = max(max_len, seq_len)

        logger.info(f"Maximum sequence length in dataset: {max_len}")
        return max_len

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

        inputs = self.tokenizer(
            prompt,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        features = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": label_vec,
        }

        return features


class T5GemmaMultiLabelClassifier(nn.Module):
    """Multi-label classifier head on top of T5-Gemma."""

    def __init__(self, backbone, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            encoder_cfg = getattr(backbone.config, "encoder", None)
            if isinstance(encoder_cfg, dict):
                hidden_size = encoder_cfg.get("hidden_size")
            else:
                hidden_size = getattr(encoder_cfg, "hidden_size", None)
        if hidden_size is None:
            decoder_cfg = getattr(backbone.config, "decoder", None)
            if isinstance(decoder_cfg, dict):
                hidden_size = decoder_cfg.get("hidden_size")
            else:
                hidden_size = getattr(decoder_cfg, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not determine hidden size for T5-Gemma classifier.")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.config = backbone.config

    def _get_encoder(self):
        if hasattr(self.backbone, "get_encoder"):
            return self.backbone.get_encoder()
        if hasattr(self.backbone, "encoder"):
            return self.backbone.encoder
        return self.backbone

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        encoder = self._get_encoder()
        try:
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs,
            )
        except TypeError:
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden_states = outputs[0]
        if hidden_states is None:
            raise ValueError("Could not extract hidden states from T5-Gemma outputs.")

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


class T5GemmaMultiLabelCollator:
    """Pads text batches for T5-Gemma."""

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

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()

            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.float32)

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


class T5GemmaTrainer(Trainer):
    def _save(self, output_dir: str | None = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()
        cpu_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
        torch.save(cpu_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    parser = argparse.ArgumentParser(description="Train T5-Gemma multi-label router (multi-hot)")
    parser.add_argument("--model-name", type=str, default="google/t5gemma-2-270m-270m", help="Model name or path")
    parser.add_argument("--train-data", type=str, default="route/train/data/train_data.json", help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="route/train/checkpoints/t5gemma_270m", help="Output directory for model checkpoints")
    parser.add_argument("--num-train-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--classifier-dropout", type=float, default=0.1, help="Dropout for classifier head")
    parser.add_argument("--label-threshold", type=float, default=0.8, help="Sigmoid probability threshold for metrics")
    parser.add_argument("--use-lora", action="store_true", default=True, help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA r parameter")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--wandb-project", type=str, default="t5gemma-router", help="Wandb project name")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    backbone = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if args.use_lora:
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()

    model = T5GemmaMultiLabelClassifier(
        backbone=backbone,
        num_labels=len(LABELS),
        dropout=args.classifier_dropout,
    )
    model.train()

    logger.info("Creating dataset")
    full_dataset = MultiLabelRouteDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
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

    data_collator = T5GemmaMultiLabelCollator(
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

    trainer = T5GemmaTrainer(
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
        model_family="t5gemma",
        classifier_dropout=args.classifier_dropout,
        label_threshold=args.label_threshold,
        use_lora=args.use_lora,
    )

    if args.use_wandb:
        wandb.finish()

    logger.info("Training completed!")


if __name__ == "__main__":
    import argparse

    main()
