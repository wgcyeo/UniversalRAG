import os
import json
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file as safetensors_load_file
import logging
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
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

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _supports_inputs_embeds(model) -> bool:
    try:
        return "inputs_embeds" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _get_internvl_model(model):
    base_model = getattr(model, "base_model", None)
    wrapped = getattr(base_model, "model", None) if base_model is not None else None
    return wrapped if wrapped is not None else model


def _load_internvl_backbone(model_name: str, dtype, device: str):
    load_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    device_map = "auto" if device == "cuda" else None
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    try:
        model = AutoModel.from_pretrained(model_name, **load_kwargs)
        used_device_map = device_map is not None
    except RuntimeError as exc:
        if "Tensor.item() cannot be called on meta tensors" not in str(exc):
            raise
        logger.warning(
            "InternVL init does not support meta tensors; retrying without device_map/low_cpu_mem_usage."
        )
        load_kwargs.pop("device_map", None)
        load_kwargs["low_cpu_mem_usage"] = False
        model = AutoModel.from_pretrained(model_name, **load_kwargs)
        used_device_map = False
    if device == "cuda" and not used_device_map:
        model = model.to(device)
    return model


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

    def _get_base_model(self):
        return _get_internvl_model(self.backbone)

    def _encode_hidden_states(self, base_model, input_ids, attention_mask, kwargs):
        """Run only the LLM decoder stack (no vocab LM head) to get hidden states.

        The InternVL chat forward always projects hidden states through the
        language model's vocab head, which allocates (batch, seq, vocab) logits
        and OOMs at large batch sizes. The classifier only needs hidden states,
        so we merge image embeddings ourselves and call ``language_model.model``
        (the decoder stack) directly. Falls back to the full forward if the
        expected attributes are missing.
        """
        language_model = getattr(base_model, "language_model", None)
        decoder = getattr(language_model, "model", None)
        get_input_embeddings = getattr(base_model, "get_input_embeddings", None)
        img_context_token_id = getattr(base_model, "img_context_token_id", None)
        if (
            language_model is None
            or decoder is None
            or get_input_embeddings is None
            or img_context_token_id is None
            or not hasattr(base_model, "extract_feature")
        ):
            return None

        pixel_values = kwargs.get("pixel_values")
        image_flags = kwargs.get("image_flags")
        has_image_tokens = bool((input_ids == img_context_token_id).any())
        if pixel_values is None or not has_image_tokens:
            # Text-only batch: skip the vision tower entirely.
            inputs_embeds = get_input_embeddings()(input_ids)
        else:
            input_embeds = get_input_embeddings()(input_ids).clone()
            vit_embeds = base_model.extract_feature(pixel_values)
            if image_flags is not None:
                flags = image_flags.squeeze(-1).to(vit_embeds.device)
                vit_embeds = vit_embeds[flags == 1]

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            selected = (input_ids.reshape(B * N) == img_context_token_id)
            n_selected = int(selected.sum())
            if n_selected > 0:
                vit_flat = vit_embeds.reshape(-1, C).to(input_embeds.dtype)
                n_token = min(n_selected, vit_flat.size(0))
                input_embeds[selected.nonzero(as_tuple=True)[0][:n_token]] = vit_flat[:n_token]
            inputs_embeds = input_embeds.reshape(B, N, C)

        outputs = decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden_states = outputs[0]
        return hidden_states

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        base_model = self._get_base_model()
        hidden_states = self._encode_hidden_states(base_model, input_ids, attention_mask, kwargs)

        if hidden_states is None:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
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
        if self.classifier.weight.dtype != pooled.dtype:
            pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device).float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return {"loss": loss, "logits": logits}


class InternVLMultiLabelRouter:
    """Router using fine-tuned InternVL3.5 multi-label classifier."""

    def __init__(
        self,
        model_path: str,
        device: str = None,
        classifier_dropout: float = 0.1,
        use_lora: bool = False,
        input_size: int = 448,
        max_num: int = 12,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.input_size = input_size
        self.max_num = max_num

        logger.info(f"Loading model from {model_path} on {self.device}")
        model_name = PeftConfig.from_pretrained(model_path).base_model_name_or_path if use_lora else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        state_dict = self._load_state_dict(model_path)
        backbone_state, has_backbone_prefix = self._extract_backbone_state(state_dict)

        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        if model_name == model_path and not os.path.exists(os.path.join(model_path, "modeling_internvl_chat.py")):
            raise ValueError(
                "Base model code is missing in the checkpoint. "
                "Use a full checkpoint or a LoRA adapter with adapter_config.json."
            )
        backbone = _load_internvl_backbone(model_name, torch_dtype, self.device)
        if state_dict is not None:
            load_state = backbone_state if has_backbone_prefix else state_dict
            missing, unexpected = backbone.load_state_dict(load_state, strict=False)
            if missing:
                logger.warning(f"Missing keys when loading backbone weights: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading backbone weights: {unexpected}")

        if use_lora:
            backbone = PeftModel.from_pretrained(backbone, model_path)
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

        internvl_model = _get_internvl_model(backbone)
        internvl_model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.num_image_token = internvl_model.num_image_token
        self.vision_device, self.vision_dtype = self._resolve_vision_context(internvl_model)

        self.model = InternVLMultiLabelClassifier(
            backbone=backbone,
            num_labels=len(LABELS),
            dropout=classifier_dropout,
        )
        self._load_classifier_weights(state_dict)
        self.model.eval()

    def _load_state_dict(self, model_path: str):
        safetensors_path = os.path.join(model_path, "model.safetensors")
        bin_path = os.path.join(model_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            return safetensors_load_file(safetensors_path)
        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu")
        logger.warning("No saved state dict found; using backbone weights only.")
        return None

    def _extract_backbone_state(self, state_dict):
        if not state_dict:
            return {}, False
        backbone_state = {}
        has_prefix = False
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                backbone_state[key[len("backbone."):]] = value
                has_prefix = True
        return backbone_state, has_prefix

    def _load_classifier_weights(self, state_dict):
        if not state_dict:
            return
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading classifier weights: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading classifier weights: {unexpected}")

    def _resolve_vision_context(self, internvl_model):
        vision_model = getattr(internvl_model, "vision_model", None)
        param = None
        if vision_model is not None:
            try:
                param = next(vision_model.parameters())
            except StopIteration:
                param = None
        if param is None:
            try:
                param = next(internvl_model.parameters())
            except StopIteration:
                return self.device, torch.float32
        return param.device, param.dtype

    def _build_features(self, query: str, query_image: str = None):
        """Build per-sample features (un-batched, no tensor stacking)."""
        prompt = ROUTER_PROMPT.format(query=query.strip())

        message = ""
        pixel_values = None
        if query_image:
            try:
                pixel_values = load_image(query_image, input_size=self.input_size, max_num=self.max_num)
                message += "<image>\n"
            except Exception as e:
                logger.warning(f"Failed to load image {query_image}: {e}")

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
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }

    def _collate_features(self, features: list, pad_to_multiple_of: int = 8):
        """Pad text and concatenate image patches across a batch of features."""
        max_length = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of:
            max_length = (
                (max_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
            )

        pad_token_id = self.tokenizer.pad_token_id or 0

        input_ids_batch = []
        attention_mask_batch = []
        pixel_values_list = []
        image_flags_list = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]

            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            pixel_values_list.append(feature["pixel_values"])
            image_flags_list.append(feature["image_flags"])

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "pixel_values": torch.cat(pixel_values_list, dim=0),
            "image_flags": torch.cat(image_flags_list, dim=0),
        }

    def _to_model_inputs(self, batch: dict):
        model_inputs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if key == "image_flags":
                    model_inputs[key] = value.cpu()
                elif key == "pixel_values":
                    target_device = self.vision_device or self.device
                    target_dtype = self.vision_dtype if value.is_floating_point() else None
                    model_inputs[key] = value.to(device=target_device, dtype=target_dtype)
                else:
                    model_inputs[key] = value.to(self.device)
            else:
                model_inputs[key] = value
        return model_inputs

    def _predict_batch(self, features: list, threshold: float, return_scores: bool):
        batch = self._collate_features(features)
        model_inputs = self._to_model_inputs(batch)

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        probs_batch = torch.sigmoid(outputs["logits"]).cpu().tolist()

        predictions = []
        for probs in probs_batch:
            label_scores = list(zip(LABELS, probs))
            selected, selected_scores = select_labels_from_scores(label_scores, threshold=threshold)
            predictions.append((selected, selected_scores) if return_scores else selected)
        return predictions

    def route(
        self,
        query: str,
        query_image: str = None,
        threshold: float = 0.8,
        return_scores: bool = False,
    ) -> list:
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return (["no"], [1.0]) if return_scores else ["no"]

        features = self._build_features(query, query_image=query_image)
        return self._predict_batch([features], threshold=threshold, return_scores=return_scores)[0]

    def route_batch(
        self,
        queries: list,
        query_images: list = None,
        batch_size: int = 8,
        threshold: float = 0.8,
        return_scores: bool = False,
    ) -> list:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if query_images is None:
            query_images = [None] * len(queries)

        if len(queries) != len(query_images):
            raise ValueError("queries and query_images must have the same length")

        predictions = [None] * len(queries)
        indexed_items = []
        empty_count = 0
        for idx, (query, image) in enumerate(zip(queries, query_images)):
            if query and query.strip():
                indexed_items.append((idx, query, image))
            else:
                predictions[idx] = (["no"], [1.0]) if return_scores else ["no"]
                empty_count += 1

        if empty_count:
            logger.warning(f"Found {empty_count} empty queries; assigning 'no'.")

        for i in tqdm(range(0, len(indexed_items), batch_size), desc="Routing queries"):
            batch_items = indexed_items[i:i + batch_size]
            features = [
                self._build_features(query, query_image=image)
                for _, query, image in batch_items
            ]
            batch_preds = self._predict_batch(features, threshold=threshold, return_scores=return_scores)
            for (idx, _, _), pred in zip(batch_items, batch_preds):
                predictions[idx] = pred

        return predictions


def prediction_to_string(prediction) -> str:
    if isinstance(prediction, list):
        return normalize_retrieval("+".join(prediction))
    if isinstance(prediction, str):
        return normalize_retrieval(prediction)
    return "error"


def main():
    parser = argparse.ArgumentParser(description="Route queries using trained InternVL3.5 multi-label model")
    parser.add_argument("--input-dir", type=str, default="dataset/query", help="Directory containing input query files (JSON)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--input", type=str, help="Input query file (JSON) or single query string")
    parser.add_argument("--output", type=str, help="Output file for predictions (JSON)")
    parser.add_argument("--output-dir", type=str, default="route/results/internvl3_5_1b", help="Directory to save predictions when processing a directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for processing")
    parser.add_argument("--threshold", type=float, default=0.8, help="Sigmoid probability threshold for selecting labels")
    parser.add_argument("--input-size", type=int, default=448, help="Image input size")
    parser.add_argument("--max-num", type=int, default=12, help="Maximum number of image tiles")
    parser.add_argument("--classifier-dropout", type=float, default=0.1, help="Dropout for classifier head")
    parser.add_argument("--use-lora", action="store_true", default=True, help="Load LoRA adapters from the model path")
    parser.add_argument("--return-scores", action="store_true", default=True, help="Include per-label probabilities for selected retrieval classes")

    args = parser.parse_args()

    router = InternVLMultiLabelRouter(
        args.model_path,
        classifier_dropout=args.classifier_dropout,
        use_lora=args.use_lora,
        input_size=args.input_size,
        max_num=args.max_num,
    )

    def run_on_file(input_path: str, output_path: str = None, return_summary: bool = False):
        logger.info(f"Loading queries from {input_path}")
        with open(input_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            queries = [item.get("question", item.get("query", "")) for item in data]
            query_images = [item.get("query_image", None) for item in data]
        else:
            queries = [data.get("question", data.get("query", ""))]
            query_images = [data.get("query_image", None)]

        predictions = router.route_batch(
            queries,
            query_images=query_images,
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
