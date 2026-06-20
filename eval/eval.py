import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

from universalrag.retriever import TextRetriever, TableRetriever, ImgRetriever, ClipRetriever, VidRetriever

import importlib


ALLOWED_MODALITIES = {"no", "paragraph", "document", "table", "image", "clip", "video"}
ROUTER_MODEL_ALIASES = {
    "gpt-5": "gpt-5",
    "qwen3-vl-2b": "qwen3_vl_2b",
    "qwen3_vl_2b": "qwen3_vl_2b",
    "internvl3_5-1b": "internvl3_5_1b",
    "internvl3_5_1b": "internvl3_5_1b",
    "t5gemma-270m": "t5gemma_270m",
    "t5gemma_270m": "t5gemma_270m",
}


def normalize_router_model_name(router_model):
    return ROUTER_MODEL_ALIASES.get(router_model, router_model)


class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_module = self._load_model_module()
        self.model, self.processor, self.tokenizer = self.model_module.load_model(model_path)

    def _load_model_module(self):
        if "Qwen3-VL" in self.model_path:
            module_name = "qwen3_vl"
        elif "InternVL3_5" in self.model_path:
            module_name = "internvl3_5"
        elif "Molmo2" in self.model_path:
            module_name = "molmo2"
        else:
            raise ValueError(f"Unsupported model type: {self.model_path}")
        return importlib.import_module(f"models.{module_name}")

    def inference(self, query, **kwargs):
        return self.model_module.inference(self.model, self.processor, self.tokenizer, query, **kwargs)


def reformat(row):
    query, data_type = row["question"], row["source"]
    if data_type in ["mmlu", "MRAG-Bench", "lvbench"]:
        return f"{query} Please respond with only a single letter (A, B, C, or D)."
    elif data_type in ["natural_questions", "hotpotqa", "HybridQA", "infoseek"]:
        return f"{query} Please respond with only the exact answer."
    elif data_type in ["webqa", "videorag_wikihow", "videorag_synth"]:
        return f"{query} Please respond in a complete sentence."
    else:
        raise ValueError(f"Invalid data type: {data_type}")


def load_parquet(path):
    return pd.read_parquet(path, engine="fastparquet")


def load_clip_data_from_parquet(parquet_paths):
    frame_num_data = {}
    frame_time_data = {}
    for path in parquet_paths:
        df = load_parquet(path)
        frame_num_data.update(df["frame_num"].to_dict())
        frame_time_data.update(df["frame_time"].to_dict())
    return frame_num_data, frame_time_data


def parse_nframes(nframes_str):
    if not nframes_str:
        return {}
    nframes_dict = {}
    for pair in nframes_str.split(","):
        if not pair:
            continue
        parts = pair.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid nframes entry: {pair}")
        modality, value = parts
        nframes_dict[modality.strip()] = int(value)
    return nframes_dict


def normalize_modalities(raw_retrieval):
    if raw_retrieval is None:
        return ["no"]
    if isinstance(raw_retrieval, list):
        modalities = raw_retrieval
    elif isinstance(raw_retrieval, str):
        if "+" in raw_retrieval:
            modalities = [part.strip() for part in raw_retrieval.split("+") if part.strip()]
        else:
            modalities = [raw_retrieval.strip()]
    else:
        modalities = ["no"]

    normalized = []
    for modality in modalities:
        if not modality:
            continue
        label = modality.lower()
        if label == "error":
            continue
        if label in ALLOWED_MODALITIES:
            normalized.append(label)

    if not normalized:
        return ["no"]
    if len(normalized) > 1 and "no" in normalized:
        normalized = [m for m in normalized if m != "no"]
    return normalized


def build_existing_cache(model_dir, target, top_k):
    return {
        "model_dir": model_dir,
        "target": target,
        "top_k": top_k,
        "data": {},
    }


def load_existing_results(cache, modality):
    if modality in cache["data"]:
        return cache["data"][modality]

    file_path = os.path.join(
        "eval",
        "results",
        cache["model_dir"],
        modality,
        f"{cache['target']}_top{cache['top_k']}.json",
    )
    if not os.path.exists(file_path):
        cache["data"][modality] = None
        return None

    with open(file_path, "r") as f:
        rows = json.load(f)
    cache["data"][modality] = {row["index"]: row for row in rows}
    return cache["data"][modality]


def get_existing_row(cache, modality, index):
    results = load_existing_results(cache, modality)
    if results is None:
        return None
    return results.get(index)


def init_inference_resources(model_path, target, alpha):
    model = ModelLoader(model_path)

    retrievers = {
        "paragraph": TextRetriever(
            queryfeats_path=f"eval/features/query/qwen3/{target}.pkl",
            textfeats_path=["eval/features/paragraph.pkl"],
        ),
        "document": TextRetriever(
            queryfeats_path=f"eval/features/query/qwen3/{target}.pkl",
            textfeats_path=["eval/features/document.pkl"],
        ),
        "table": TableRetriever(
            queryfeats_path=f"eval/features/query/qwen3/{target}.pkl",
            tablefeats_path=["eval/features/table.pkl"],
        ),
        "image": ImgRetriever(
            queryfeats_path=f"eval/features/query/vlm2vec/{target}.pkl",
            imgfeats_path=["eval/features/image.pkl"],
        ),
        "clip": ClipRetriever(
            queryfeats_path=f"eval/features/query/vlm2vec/{target}.pkl",
            clipfeats_path=["eval/features/clip.pkl"],
            clipscriptfeats_path=["eval/features/clip_script.pkl"],
            alpha=alpha,
        ),
        "video": VidRetriever(
            queryfeats_path=f"eval/features/query/vlm2vec/{target}.pkl",
            vidfeats_path=["eval/features/video.pkl"],
            vidscriptfeats_path=["eval/features/video_script.pkl"],
            alpha=alpha,
        ),
    }

    data = {
        "paragraph": load_parquet("dataset/paragraph.parquet"),
        "document": load_parquet("dataset/document.parquet"),
        "table": load_parquet("dataset/table.parquet"),
        "clip_frames": load_clip_data_from_parquet(["dataset/clip.parquet"]),
    }

    return model, retrievers, data


def retrieve_for_modality(modality, row, retrievers, data, top_k, nframes_dict):
    if modality == "no":
        return [], {}

    if modality == "paragraph":
        retrieved, _ = retrievers["paragraph"].retrieve(row["index"], top_k=top_k)
        retrieved_texts = [data["paragraph"].loc[doc]["text"] for doc in retrieved]
        return retrieved, {"retrieved_texts": retrieved_texts}
    if modality == "document":
        retrieved, _ = retrievers["document"].retrieve(row["index"], top_k=top_k, long_text=True)
        retrieved_texts = [data["document"].loc[doc]["text"] for doc in retrieved]
        return retrieved, {"retrieved_texts": retrieved_texts}
    if modality == "table":
        retrieved, _ = retrievers["table"].retrieve(row["index"], top_k=top_k)
        retrieved_texts = [data["table"].loc[doc]["text"] for doc in retrieved]
        return retrieved, {"retrieved_texts": retrieved_texts}
    if modality == "image":
        retrieved, _ = retrievers["image"].retrieve(row["index"], top_k=top_k)
        return retrieved, {"retrieved_images": retrieved}
    if modality == "clip":
        retrieved, _ = retrievers["clip"].retrieve(row["index"], top_k=top_k)
        clipframenum, clipframetime = data["clip_frames"]
        retrieved_videos = [clip.rsplit("_", 1)[0] for clip in retrieved]
        startend_frames = [clipframenum[video_id] for video_id in retrieved]
        startend_times = [clipframetime[video_id] for video_id in retrieved]
        return retrieved, {
            "retrieved_videos": retrieved_videos,
            "startend_frames": startend_frames,
            "startend_times": startend_times,
            "nframes": nframes_dict.get("clip", 32),
        }
    if modality == "video":
        retrieved, _ = retrievers["video"].retrieve(row["index"], top_k=top_k)
        return retrieved, {
            "retrieved_videos": retrieved,
            "nframes": nframes_dict.get("video", 32),
        }
    raise ValueError(f"Invalid modality: {modality}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        choices=[
            "Qwen/Qwen3-VL-8B-Instruct",
            "OpenGVLab/InternVL3_5-8B",
            "allenai/Molmo2-4B",
            "microsoft/Phi-4-multimodal-instruct",
            "google/gemma-3-4b-it",
        ],
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--router-model",
        type=str,
        required=True,
        help="Router model name used in script/3_route.sh; result-folder aliases are also accepted",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=[
            "mmlu",
            "natural_questions",
            "hotpotqa",
            "hybridqa",
            "webqa",
            "mrag_bench",
            "infoseek",
            "lvbench",
            "videorag_wikihow",
            "videorag_synth",
        ],
        help="Target dataset for evaluation",
    )
    parser.add_argument("--top-k", type=int, default=1, help="Number of top retrieval to use")
    parser.add_argument("--alpha", type=float, default=0.2, help="Weight for video script features (0 to 1)")
    parser.add_argument(
        "--nframes",
        type=str,
        default="clip:32,video:32",
        help="Number of frames to process for each modality, e.g. 'clip:8,video:32'",
    )
    args = parser.parse_args()

    model_path = args.model_path
    router_model_input = args.router_model
    router_model = normalize_router_model_name(router_model_input)
    router_model_display = (
        router_model
        if router_model_input == router_model
        else f"{router_model_input} (results folder: {router_model})"
    )
    target = args.target
    top_k = args.top_k
    alpha = args.alpha
    nframes_dict = parse_nframes(args.nframes)

    print(
        "LVLM Model: {}, Router Model: {}, Target: {}, Top-k: {}, Alpha: {}, NFrames: {}".format(
            model_path, router_model_display, target, top_k, alpha, args.nframes
        )
    )

    target_file = os.path.join("route", "results", router_model, f"{target}.json")
    with open(target_file, "r") as f:
        data = json.load(f)

    model_dir = model_path.split("/")[-1].replace(".", "_")
    existing_cache = build_existing_cache(model_dir, target, top_k)

    model = None
    retrievers = None
    data_sources = None

    for row in tqdm(data):
        modalities = normalize_modalities(row.get("retrieval"))
        if len(modalities) == 1:
            modality = modalities[0]
            existing_row = get_existing_row(existing_cache, modality, row.get("index"))
            if existing_row is not None:
                row["retrieved"] = existing_row.get("retrieved", [])
                row["response"] = existing_row.get("response", "")
                row["retrieval_used"] = modality
                continue

        if model is None:
            model, retrievers, data_sources = init_inference_resources(model_path, target, alpha)
        query = reformat(row)
        query_image = row.get("query_image", None)

        if len(modalities) == 1:
            modality = modalities[0]
            retrieved, inference_kwargs = retrieve_for_modality(
                modality, row, retrievers, data_sources, top_k, nframes_dict
            )
            response = model.inference(query, query_image=query_image, **inference_kwargs)
            row["retrieved"] = retrieved
            row["response"] = response
            row["retrieval_used"] = modality
            continue

        retrieved_all = {}
        retrieved_flat = []
        retrieved_texts = []
        retrieved_images = []
        retrieved_videos = []
        startend_frames = []
        startend_times = []
        has_video = False

        for modality in modalities:
            retrieved, inference_kwargs = retrieve_for_modality(
                modality, row, retrievers, data_sources, top_k, nframes_dict
            )
            retrieved_all[modality] = retrieved
            retrieved_flat.extend(retrieved)

            if "retrieved_texts" in inference_kwargs:
                retrieved_texts.extend(inference_kwargs["retrieved_texts"])
            if "retrieved_images" in inference_kwargs:
                retrieved_images.extend(inference_kwargs["retrieved_images"])
            if "retrieved_videos" in inference_kwargs:
                retrieved_videos.extend(inference_kwargs["retrieved_videos"])
                if modality == "clip":
                    startend_frames.extend(inference_kwargs.get("startend_frames", []))
                    startend_times.extend(inference_kwargs.get("startend_times", []))
                else:
                    count = len(inference_kwargs["retrieved_videos"])
                    startend_frames.extend([[None, None]] * count)
                    startend_times.extend([None] * count)
                    has_video = True

        combined_kwargs = {}
        if retrieved_texts:
            combined_kwargs["retrieved_texts"] = retrieved_texts
        if retrieved_images:
            combined_kwargs["retrieved_images"] = retrieved_images
        if retrieved_videos:
            combined_kwargs["retrieved_videos"] = retrieved_videos
            combined_kwargs["startend_frames"] = startend_frames
            combined_kwargs["startend_times"] = startend_times
            combined_kwargs["nframes"] = nframes_dict.get("video", 32) if has_video else nframes_dict.get("clip", 32)

        response = model.inference(query, query_image=query_image, **combined_kwargs)
        row["retrieved"] = retrieved_flat
        row["response"] = response
        row["retrieval_used"] = "+".join(modalities)
        row["retrieved_all"] = retrieved_all

    output_dir = os.path.join("eval", "results", model_dir, router_model)
    os.makedirs(output_dir, exist_ok=True)
    nframes_tag = args.nframes.replace(",", "_").replace(":", "")
    output_file = os.path.join(
        output_dir,
        f"{target}_top{top_k}_{alpha}_{nframes_tag}.json",
    )

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
