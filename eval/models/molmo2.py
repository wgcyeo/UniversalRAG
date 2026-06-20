from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

from utils.custom_molmo_utils import process_vision_info
from utils.utils import get_scripts_for_videos

def load_model(model_path):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True,
        dtype="auto",
        device_map="auto"
    )
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
                entry for index, (video_path, startend_time) in enumerate(zip(kwargs["retrieved_videos"], kwargs.get("startend_times", [[None, None]] * len(kwargs["retrieved_videos"]))))
                for entry in [
                    {"type": "text", "text": f"Relevant video {index+1}:\n{scripts[index]}"},
                    {"type": "video", "video": video_path, "num_frames": kwargs.get("nframes", 32), "clip": startend_time if startend_time and startend_time[0] is not None else None}
                ]
            ])
        else:
            messages[0]["content"].extend([
                {"type": "video", "video": video_path, "num_frames": kwargs.get("nframes", 32), "clip": startend_time if startend_time and startend_time[0] is not None else None}
                for video_path, startend_time in zip(kwargs["retrieved_videos"], kwargs.get("startend_times", [[None, None]] * len(kwargs["retrieved_videos"])))
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

    image_inputs, video_inputs, video_kwargs = process_vision_info([messages])
    
    if video_inputs is None:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
    else:
        videos, video_metadatas = zip(*video_inputs)
        videos, video_metadatas = list(videos), list(video_metadatas)
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            videos=videos,
            video_metadata=video_metadatas,
            text=text,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=kwargs.get("max_new_tokens", 1024))
    
    generated_tokens = output_ids[0, inputs['input_ids'].size(1):]
    output_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text
