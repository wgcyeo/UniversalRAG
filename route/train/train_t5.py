from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, set_seed
from datasets import Dataset, ClassLabel
import argparse
import logging
import os
import wandb
import shutil

os.environ["WANDB_MODE"] = "online"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="Model name")
    parser.add_argument("--input_dir", type=str, default="route/train/data/train_data_t5.json", help="Directory containing training data")
    parser.add_argument("--train_size", type=float, default=0.9, help="Proportion of dataset to use for training")
    parser.add_argument("--max_input_length", type=int, default=512, help="Max input length for tokenization")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for training")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for training")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total number of checkpoints saved")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every n steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every n steps")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluate every n steps")
    parser.add_argument("--output_dir", type=str, default="route/train/temp", help="Directory to save temporary files")
    parser.add_argument("--checkpoint_dir", type=str, default="route/train/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    args.output_dir = args.output_dir + f"/{args.model_name.split('/')[-1].replace('flan-', '')}"
    args.checkpoint_dir = args.checkpoint_dir + f"/{args.model_name.split('/')[-1].replace('flan-', '')}"

    set_seed(args.seed)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    logger.info(f"Loaded pretrained model: {args.model_name}")

    dataset_path = args.input_dir
    dataset = Dataset.from_json(dataset_path)

    sources = list(set(dataset["source"]))
    class_label = ClassLabel(names=sources)

    dataset = dataset.map(
        lambda x: {"source_label": class_label.str2int(x["source"])},
        remove_columns=["source"]
    )

    dataset = dataset.cast_column("source_label", class_label)
    split_dataset = dataset.train_test_split(train_size=args.train_size, stratify_by_column="source_label", seed=42)

    train_dataset, val_dataset = split_dataset["train"], split_dataset["test"]

    def _preprocess_data(examples):
        inputs = examples["question"]
        model_inputs = tokenizer(inputs, max_length=args.max_input_length, padding=True, truncation=True)
        model_inputs["labels"] = tokenizer(examples["gt_retrieval"], max_length=args.max_input_length, padding=False, truncation=True)["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(_preprocess_data, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(_preprocess_data, batched=True, remove_columns=val_dataset.column_names)

    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = predictions.argmax(axis=-1) if predictions.ndim > 2 else predictions

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        matches = [pred == label for pred, label in zip(decoded_preds, decoded_labels)]
        accuracy = sum(matches) / len(matches)

        return {"accuracy": accuracy}

    wandb.init(project="UniversalRAG", name=args.model_name, config=vars(args))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        fp16=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.checkpoint_dir)
    logger.info(f"Model saved to: {args.checkpoint_dir}")

    wandb.finish()
    
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

if __name__ == "__main__":
    main()
