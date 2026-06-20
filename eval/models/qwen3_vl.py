from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

from utils.custom_qwen_vl_utils import process_vision_info
from utils.utils import get_scripts_for_videos

def load_model(model_path):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor, None

def inference(model, processor, tokenizer, query, **kwargs):
    
    messages = [
        {
            "role": "user",
            "content": []
        },
    ]

    prefixes = []

    if kwargs.get("retrieved_texts"):
        messages[0]["content"].extend([
            {"type": "text", "text": f"Relevant document {index+1}:\n{text}"} for index, text in enumerate(kwargs["retrieved_texts"])
        ])
        prefixes.append("documents")

    if kwargs.get("retrieved_images"):
        if kwargs.get("use_caption", False):
            messages[0]["content"].extend([
                entry for index, image_path in enumerate(kwargs["retrieved_images"])
                for entry in [
                    {"type": "text", "text": f"Relevant image {index+1}:\n{kwargs['img_metadata'][image_path]['caption']}"},
                    {"type": "image", "image": image_path}
                ]
            ])
        else:
            messages[0]["content"].extend([
                {"type": "image", "image": image_path} for image_path in kwargs["retrieved_images"]
            ])
        prefixes.append("images")

    if kwargs.get("retrieved_videos"):
        if kwargs.get("use_scripts", False):
            scripts = get_scripts_for_videos(kwargs["retrieved_videos"], kwargs.get("startend_times"))
            messages[0]["content"].extend([
                entry for index, (video_path, startend_frame) in enumerate(zip(kwargs["retrieved_videos"], kwargs.get("startend_frames", [[None, None]] * len(kwargs["retrieved_videos"]))))
                for entry in [
                    {"type": "text", "text": f"Relevant video {index+1}:\n{scripts[index]}"},
                    {"type": "video", "video": video_path, "nframes": kwargs.get("nframes", 32), "video_start_fr": startend_frame[0], "video_end_fr": startend_frame[1]}
                ]
            ])
        else:
            messages[0]["content"].extend([
                {"type": "video", "video": video_path, "nframes": kwargs.get("nframes", 32), "video_start_fr": startend_frame[0], "video_end_fr": startend_frame[1]}
                for video_path, startend_frame in zip(kwargs["retrieved_videos"], kwargs.get("startend_frames", [[None, None]] * len(kwargs["retrieved_videos"])))
            ])
        prefixes.append("videos")

    if prefixes:
        if len(prefixes) == 1:
            prefix_text = prefixes[0]
        elif len(prefixes) == 2:
            prefix_text = " and ".join(prefixes)
        else:
            prefix_text = ", ".join(prefixes[:-1]) + ", and " + prefixes[-1]
        query = f"Considering the given {prefix_text},\n" + query

    if kwargs.get("query_image") is not None:
        messages[0]["content"].append({"type": "image", "image": kwargs["query_image"]})

    messages[0]["content"].extend([{"type": "text", "text": query}])

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", **video_kwargs).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=kwargs.get("max_new_tokens", 1024))
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0].strip()
