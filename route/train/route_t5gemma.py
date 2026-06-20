import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file as safetensors_load_file
import logging
from tqdm import tqdm
from tabulate import tabulate

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from route.prompt import ROUTER_PROMPT
from route.train.router_common import (
    LABELS,
    normalize_retrieval,
    select_labels_from_scores,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ALLOWED_CATEGORIES = [label.lower() for label in LABELS]
RETRIEVAL_METHODS = ALLOWED_CATEGORIES + ["error"]


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
        encoder = self._get_encoder()
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
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


class T5GemmaMultiLabelRouter:
    """Router using fine-tuned T5-Gemma multi-label classifier."""

    def __init__(
        self,
        model_path: str,
        device: str = None,
        classifier_dropout: float = 0.1,
        use_lora: bool = True,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading model from {model_path} on {self.device}")
        model_name = PeftConfig.from_pretrained(model_path).base_model_name_or_path if use_lora else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if use_lora:
            backbone = PeftModel.from_pretrained(backbone, model_path)

        self.model = T5GemmaMultiLabelClassifier(
            backbone=backbone,
            num_labels=len(LABELS),
            dropout=classifier_dropout,
        )
        self._load_classifier_weights(model_path)
        self.model.eval()

    def _load_classifier_weights(self, model_path: str):
        safetensors_path = os.path.join(model_path, "model.safetensors")
        bin_path = os.path.join(model_path, "pytorch_model.bin")

        state_dict = None
        if os.path.exists(safetensors_path):
            state_dict = safetensors_load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")

        if state_dict is None:
            logger.warning("No saved state dict found for classifier head; using backbone weights only.")
            return

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading classifier weights: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading classifier weights: {unexpected}")

    def _build_inputs(self, query: str):
        return self._build_batch_inputs([query])

    def _build_batch_inputs(self, queries: list[str]):
        prompts = [ROUTER_PROMPT.format(query=query.strip()) for query in queries]
        return self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    def _empty_prediction(self, return_scores: bool = False):
        if return_scores:
            return ["no"], [1.0]
        return ["no"]

    def _prediction_from_probs(
        self,
        probs: list[float],
        threshold: float = 0.8,
        return_scores: bool = False,
    ):
        label_scores = list(zip(LABELS, probs))
        selected, selected_scores = select_labels_from_scores(label_scores, threshold=threshold)
        if return_scores:
            return selected, selected_scores
        return selected

    def route(
        self,
        query: str,
        threshold: float = 0.8,
        return_scores: bool = False,
    ) -> list:
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return self._empty_prediction(return_scores=return_scores)

        inputs = self._build_inputs(query)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs["logits"]
        probs = torch.sigmoid(logits).cpu().tolist()[0]
        return self._prediction_from_probs(
            probs,
            threshold=threshold,
            return_scores=return_scores,
        )

    def route_batch(
        self,
        queries: list,
        batch_size: int = 8,
        threshold: float = 0.8,
        return_scores: bool = False,
    ) -> list:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        predictions = [None] * len(queries)
        indexed_queries = []
        empty_count = 0
        for idx, query in enumerate(queries):
            if query and query.strip():
                indexed_queries.append((idx, query))
            else:
                predictions[idx] = self._empty_prediction(return_scores=return_scores)
                empty_count += 1

        if empty_count:
            logger.warning(f"Found {empty_count} empty queries; assigning 'no'.")

        for i in tqdm(range(0, len(indexed_queries), batch_size), desc="Routing queries"):
            batch_items = indexed_queries[i:i + batch_size]
            batch_indices = [idx for idx, _ in batch_items]
            batch_queries = [query for _, query in batch_items]

            inputs = self._build_batch_inputs(batch_queries)
            inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model(**inputs)

            probs_batch = torch.sigmoid(outputs["logits"]).cpu().tolist()
            for idx, probs in zip(batch_indices, probs_batch):
                predictions[idx] = self._prediction_from_probs(
                    probs,
                    threshold=threshold,
                    return_scores=return_scores,
                )

        return predictions


def main():
    parser = argparse.ArgumentParser(description="Route queries using trained T5-Gemma multi-label model")
    parser.add_argument("--input-dir", type=str, default="dataset/query", help="Directory containing input query files (JSON)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--input", type=str, help="Input query file (JSON) or single query string")
    parser.add_argument("--output", type=str, help="Output file for predictions (JSON)")
    parser.add_argument("--output-dir", type=str, default="route/results/t5gemma_270m", help="Directory to save predictions when processing a directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for processing")
    parser.add_argument("--threshold", type=float, default=0.8, help="Sigmoid probability threshold for selecting labels")
    parser.add_argument("--classifier-dropout", type=float, default=0.1, help="Dropout for classifier head")
    parser.add_argument("--use-lora", action="store_true", default=True, help="Load LoRA adapters from the model path")
    parser.add_argument("--return-scores", action="store_true", default=True, help="Include per-label probabilities for selected retrieval classes")

    args = parser.parse_args()

    router = T5GemmaMultiLabelRouter(
        args.model_path,
        classifier_dropout=args.classifier_dropout,
        use_lora=args.use_lora,
    )

    def prediction_to_string(prediction) -> str:
        if isinstance(prediction, list):
            return normalize_retrieval("+".join(prediction))
        if isinstance(prediction, str):
            return normalize_retrieval(prediction)
        return "error"

    def run_on_file(input_path: str, output_path: str = None, return_summary: bool = False):
        logger.info(f"Loading queries from {input_path}")
        with open(input_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            queries = [item.get("question", item.get("query", "")) for item in data]
        else:
            queries = [data.get("question", data.get("query", ""))]

        predictions = router.route_batch(
            queries,
            batch_size=args.batch_size,
            threshold=args.threshold,
            return_scores=args.return_scores,
        )

        if isinstance(data, list):
            for item, pred in zip(data, predictions):
                if args.return_scores:
                    labels, scores = pred
                    item["retrieval"] = labels
                    item["retrieval_scores"] = scores
                else:
                    item["retrieval"] = pred
        else:
            if args.return_scores:
                labels, scores = predictions[0]
                data["retrieval"] = labels
                data["retrieval_scores"] = scores
            else:
                data["retrieval"] = predictions[0]

        result_row = None
        if return_summary:
            count = {rm: 0 for rm in RETRIEVAL_METHODS}
            correct = 0
            gt_total = 0

            rows = data if isinstance(data, list) else [data]
            for row in rows:
                pred_norm = prediction_to_string(row.get("retrieval"))
                if pred_norm not in count:
                    count[pred_norm] = 0
                count[pred_norm] += 1

                gt = row.get("gt_retrieval")
                if isinstance(gt, str):
                    gt_norm = normalize_retrieval(gt)
                    gt_total += 1
                    if pred_norm == gt_norm:
                        correct += 1

            count["accuracy"] = round(correct / gt_total, 4) if gt_total else 0.0
            result_row = {"Path": os.path.basename(input_path)}
            result_row.update(count)

        if output_path:
            logger.info(f"Saving results to {output_path}")
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            print(json.dumps(data, indent=2))

        return result_row

    input_dir = args.input_dir
    if input_dir is None and args.input and os.path.isdir(args.input):
        input_dir = args.input

    if input_dir:
        targets = [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".json")
        ]
        if not targets:
            raise ValueError("No JSON files found in the input directory.")
        os.makedirs(args.output_dir, exist_ok=True)
        overall_results = []
        for target in targets:
            output_path = os.path.join(args.output_dir, os.path.basename(target))
            result_row = run_on_file(target, output_path=output_path, return_summary=True)
            if result_row is not None:
                overall_results.append(result_row)
        if overall_results:
            print(tabulate(overall_results, headers="keys", tablefmt="fancy_grid"))
    elif args.input and os.path.isfile(args.input):
        result_row = run_on_file(args.input, output_path=args.output, return_summary=bool(args.output))
        if result_row is not None:
            print(tabulate([result_row], headers="keys", tablefmt="fancy_grid"))
    elif args.input:
        prediction = router.route(
            args.input,
            threshold=args.threshold,
            return_scores=args.return_scores,
        )
        if args.return_scores:
            labels, scores = prediction
            result = {
                "query": args.input,
                "retrieval": labels,
                "retrieval_scores": scores,
            }
        else:
            result = {"query": args.input, "retrieval": prediction}
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))

    else:
        logger.info("Entering interactive mode. Type 'exit' to quit.")
        while True:
            query = input("\nEnter query: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            if query:
                prediction = router.route(
                    query,
                    threshold=args.threshold,
                    return_scores=args.return_scores,
                )
                if args.return_scores:
                    labels, scores = prediction
                    print(f"Prediction: {labels} (scores: {scores})")
                else:
                    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    import argparse

    main()
