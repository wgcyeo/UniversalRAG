import os
import random
import json
import pickle
import argparse
from tqdm import tqdm

random.seed(42)

from retrieve.retrieve_text import BGETextRetriever
from retrieve.retrieve_image import InternImgRetriever
from retrieve.retrieve_clip import InternClipRetriever
from retrieve.retrieve_video import InternVidRetriever

import importlib

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_module = self._load_model_module()
        self.model, self.processor, self.tokenizer = self.model_module.load_model(model_path)

    def _load_model_module(self):
        if "InternVL2_5" in self.model_path:
            module_name = "internvl2_5"
        elif "Qwen2.5-VL" in self.model_path:
            module_name = "qwen2_5_vl"
        elif "Phi-3.5-vision" in self.model_path:
            module_name = "phi_3_5_vision"
        else:
            raise ValueError(f"Unsupported model type: {self.model_path}")
        return importlib.import_module(f"utils.models.{module_name}")

    def inference(self, query, **kwargs):
        return self.model_module.inference(self.model, self.processor, self.tokenizer, query, **kwargs)

def reformat(row):
    query, data_type = row["question"], row["source"]
    if data_type in ["mmlu", "lvbench"]:
        return f"{query} Please respond with only a single letter (A, B, C, or D)."
    elif data_type in ["natural_questions", "hotpotqa", "squad"]:
        return f"{query} Please respond with only the exact answer."
    elif data_type in ["webqa", "videorag_wikihow", "videorag_synth"]:
        return f"{query} Please respond in a complete sentence."
    else:
        raise ValueError(f"Invalid data type: {data_type}")

def load_pickles(paths):
    data = {}
    for path in paths:
        with open(path, 'rb') as f:
            data.update(pickle.load(f))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL2_5-8B",
                        choices=["OpenGVLab/InternVL2_5-8B", "Qwen/Qwen2.5-VL-7B-Instruct","microsoft/Phi-3.5-vision-instruct"], help="Path to the model checkpoint")
    parser.add_argument("--router_model", type=str, default="distilbert", choices=["gpt", "t5-large", "distilbert"], help="Router model to use")
    parser.add_argument("--target", type=str, required=True, choices=["mmlu", "squad", "natural_questions", "hotpotqa", "webqa", "lvbench", "videorag_wikihow", "videorag_synth"], help="Target dataset for evaluation")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top retrieval to use")
    parser.add_argument("--alpha", type=float, default=0.2, help="Weight for image caption or clip/video script features (0 to 1)")
    parser.add_argument("--nframes", type=str, default="clip:32,video:32", help="Number of frames to process for each modality, e.g. 'clip:8,video:32'")
    args = parser.parse_args()

    model_path = args.model_path
    router_model = args.router_model
    target = args.target
    top_k = args.top_k
    alpha = args.alpha
    nframes_dict = {modality: int(k) for modality, k in (pair.split(":") for pair in args.nframes.split(","))} if args.nframes else {}

    print(f"LVLM Model: {model_path}, Router Model: {router_model}, Target: {target}, Top-k: {top_k}, Alpha: {alpha}, NFrames: {args.nframes}")

    model = ModelLoader(model_path)

    retriever_paragraph = BGETextRetriever(
        queryfeats_path=f"eval/features/query/bge-large/{target}.pkl",
        textfeats_path=[
            "eval/features/text/squad.pkl",
            "eval/features/text/natural_questions.pkl",
        ],
    )
    retriever_document = BGETextRetriever(
        queryfeats_path=f"eval/features/query/bge-large/{target}.pkl",
        textfeats_path=[
            "eval/features/text/hotpotqa.pkl",
        ],
    )
    retriever_image = InternImgRetriever(
        queryfeats_path=f"eval/features/query/internvideo/{target}.pkl",
        imgfeats_path=[
            "eval/features/image/webqa.pkl",
        ],
        imgcapfeats_path=[
            "eval/features/image/webqa_imgcap.pkl",
        ],
        alpha=alpha,
    )
    clipframenum = load_pickles(["eval/features/clip/howto100m_clipframenum.pkl", "eval/features/clip/lvbench_clipframenum.pkl"])
    clipframetime = load_pickles(["eval/features/clip/lvbench_clipframetime.pkl"])
    retriever_clip = InternClipRetriever(
        queryfeats_path=f"eval/features/query/internvideo/{target}.pkl",
        clipfeats_path=[
            "eval/features/clip/howto100m.pkl",
            "eval/features/clip/lvbench.pkl",
        ],
        clipscriptfeats_path=[
            "eval/features/video/howto100m_vidscript.pkl",
            "eval/features/clip/lvbench_clipscript.pkl",
        ],
        alpha=alpha,
    )
    retriever_video = InternVidRetriever(
        queryfeats_path=f"eval/features/query/internvideo/{target}.pkl",
        vidfeats_path=[
            'eval/features/video/howto100m.pkl',
            'eval/features/video/lvbench.pkl',
        ],
        vidscriptfeats_path=[
            'eval/features/video/howto100m_vidscript.pkl',
            'eval/features/video/lvbench_vidscript.pkl',
        ],
        alpha=alpha,
    )

    target_file = f"route/results/{router_model}/{target}.json"
    with open(target_file, 'r') as f:
        data = json.load(f)

    for row in tqdm(data):
        query = reformat(row)
        modality = row["retrieval"]

        if modality == "error":
            modality = random.choice(["no", "paragraph", "document", "image", "clip", "video"])

        if modality in "no":
            response = model.inference(query)
            retrieved = []
        elif modality in ["paragraph", "document"]:
            if modality == "paragraph":
                retrieved, _ = retriever_paragraph.retrieve(row['index'], top_k=top_k)
            elif modality == "document":
                retrieved, _ = retriever_document.retrieve(row['index'], top_k=top_k)
            retrieved_texts = [open(doc, 'r').read() for doc in retrieved]
            response = model.inference(query, retrieved_texts=retrieved_texts)
        elif modality == "image":
            retrieved, _ = retriever_image.retrieve(row['index'], top_k=top_k)
            response = model.inference(query, retrieved_images=retrieved)
        elif modality == "clip":
            retrieved, _ = retriever_clip.retrieve(row['index'], top_k=top_k)
            retrieved_videos = [clip.rsplit('_', 1)[0] for clip in retrieved]
            startend_frames = [clipframenum[video_id] for video_id in retrieved]
            startend_times = [clipframetime[video_id] if video_id in clipframetime else None for video_id in retrieved]
            response = model.inference(query, retrieved_videos=retrieved_videos, startend_frames=startend_frames, startend_times=startend_times, nframes=nframes_dict.get("clip", 32))
        elif modality == "video":
            retrieved, _ = retriever_video.retrieve(row['index'], top_k=top_k)
            response = model.inference(query, retrieved_videos=retrieved, nframes=nframes_dict.get("video", 32))
        else:
            raise ValueError(f"Invalid modality: {modality}")

        row['retrieved'] = retrieved
        row['response'] = response

    os.makedirs(f"eval/results/{model_path.split("/")[-1]}/{router_model}", exist_ok=True)
    output_file = f"eval/results/{model_path.split('/')[-1]}/{router_model}/{target}_top{top_k}_{alpha}_{args.nframes.replace(",", "_").replace(":", "")}.json"

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
