import json
import os
import re
from typing import Iterable


LABELS = ["no", "paragraph", "document", "table", "image", "clip", "video"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ROUTER_CONFIG_NAME = "router_config.json"


def normalize_label_parts(label) -> list[str]:
    if isinstance(label, str):
        raw_parts = label.split("+")
    elif isinstance(label, Iterable):
        raw_parts = []
        for item in label:
            raw_parts.extend(str(item).split("+"))
    else:
        raw_parts = str(label).split("+")

    normalized = []
    for part in raw_parts:
        cleaned = re.sub(r"^[^a-z]+|[^a-z]+$", "", str(part).strip().lower())
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        aliases = {
            "none": "no",
            "no retrieval": "no",
            "noretrieval": "no",
        }
        mapped = aliases.get(cleaned, cleaned.replace(" ", ""))
        if mapped not in LABEL_TO_ID:
            raise ValueError(f"Unknown router label: {part}")
        if mapped not in normalized:
            normalized.append(mapped)

    if not normalized:
        raise ValueError(f"Could not normalize label: {label}")

    if "no" in normalized and len(normalized) > 1:
        normalized = [part for part in normalized if part != "no"]

    return normalized


def labels_to_multihot(labels: list[str]) -> list[float]:
    vector = [0.0] * len(LABELS)
    for label in labels:
        vector[LABEL_TO_ID[label]] = 1.0
    return vector


def make_label_key(labels: list[str]) -> str:
    return "+".join([label for label in LABELS if label in labels])


def normalize_retrieval(value) -> str:
    try:
        labels = normalize_label_parts(value)
    except ValueError:
        return "error"
    return make_label_key(labels)


def select_labels_from_scores(
    label_scores: list[tuple[str, float]],
    threshold: float = 0.8,
) -> tuple[list[str], list[float]]:
    ordered_scores = sorted(label_scores, key=lambda item: item[1], reverse=True)
    score_by_label = {label: score for label, score in ordered_scores}

    selected_pairs = [(label, score) for label, score in ordered_scores if score >= threshold]
    if any(label != "no" for label, _ in selected_pairs):
        selected_pairs = [pair for pair in selected_pairs if pair[0] != "no"]

    if len(selected_pairs) > 1:
        selected_labels = {label for label, _ in selected_pairs}
        if "paragraph" in selected_labels and "document" in selected_labels:
            drop_label = "paragraph" if score_by_label["paragraph"] < score_by_label["document"] else "document"
            selected_pairs = [pair for pair in selected_pairs if pair[0] != drop_label]
        selected_labels = {label for label, _ in selected_pairs}
        if "clip" in selected_labels and "video" in selected_labels:
            drop_label = "clip" if score_by_label["clip"] < score_by_label["video"] else "video"
            selected_pairs = [pair for pair in selected_pairs if pair[0] != drop_label]

    if not selected_pairs:
        selected_pairs = [ordered_scores[0]]

    selected_label_set = {label for label, _ in selected_pairs}
    selected_labels = [label for label in LABELS if label in selected_label_set]
    selected_scores = [score_by_label[label] for label in selected_labels]
    return selected_labels, selected_scores


def save_router_config(output_dir: str, **kwargs) -> None:
    os.makedirs(output_dir, exist_ok=True)
    config = {
        "labels": LABELS,
        "problem_type": "multi_label_classification",
        "loss": "binary_cross_entropy_with_logits",
        "selection": "sigmoid_threshold",
    }
    config.update(kwargs)
    with open(os.path.join(output_dir, ROUTER_CONFIG_NAME), "w") as f:
        json.dump(config, f, indent=2)
