from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

from utils.custom_qwen_vl_utils import process_vision_info
from utils.utils import get_scripts_for_videos

def load_model(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
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

    if "retrieved_texts" in kwargs:
        messages[0]["content"].extend([
            {"type": "text", "text": f"Relevant document {index+1}:\n{text}"} for index, text in enumerate(kwargs["retrieved_texts"])
        ])
        query = "Considering the given documents,\n" + query

    elif "retrieved_images" in kwargs:
        if kwargs.get("use_caption", False):
            messages[0]["content"].extend([
                entry for index, image_path in enumerate(kwargs["retrieved_images"])
                for entry in [
                    {"type": "text", "text": f"Relevant image {index+1}:\n{kwargs['img_metadata'][image_path]['caption']}"},
                    {"image": image_path, "max_pixels": 224 * 224}
                ]
            ])
        else:
            messages[0]["content"].extend([
                {"image": image_path, "max_pixels": 224 * 224} for image_path in kwargs["retrieved_images"]
            ])
        query = "Considering the given images,\n" + query

    elif "retrieved_videos" in kwargs:
        if kwargs.get("use_scripts", False):
            scripts = get_scripts_for_videos(kwargs["retrieved_videos"], kwargs.get("startend_times"))
            messages[0]["content"].extend([
                entry for index, (video_path, startend_frame) in enumerate(zip(kwargs["retrieved_videos"], kwargs.get("startend_frames", [[None, None]] * len(kwargs["retrieved_videos"]))))
                for entry in [
                    {"type": "text", "text": f"Relevant video {index+1}:\n{scripts[index]}"},
                    {"video": video_path, "max_pixels": 224 * 224, "nframes": kwargs.get("nframes", 32), "video_start_fr": startend_frame[0], "video_end_fr": startend_frame[1]}
                ]
            ])
        else:
            messages[0]["content"].extend([
                {"video": video_path, "max_pixels": 224 * 224, "nframes": kwargs.get("nframes", 32), "video_start_fr": startend_frame[0], "video_end_fr": startend_frame[1]}
                for video_path, startend_frame in zip(kwargs["retrieved_videos"], kwargs.get("startend_frames", [[None, None]] * len(kwargs["retrieved_videos"])))
            ])
        query = "Considering the given videos,\n" + query

    messages[0]["content"].extend([{"type": "text", "text": query}])

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=video_kwargs.get("fps"), padding=True, return_tensors="pt").to('cuda')

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=kwargs.get("max_new_tokens", 1024))
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]
