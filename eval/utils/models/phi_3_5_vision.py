import numpy as np
import torch
import cv2
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoProcessor

import warnings

warnings.filterwarnings("ignore")

from utils.utils import get_scripts_for_videos

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        _attn_implementation='flash_attention_2',
    ).cuda()
    image_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, num_crops=16)
    video_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, num_crops=4)
    return model, image_processor, video_processor

def load_video(video_path, max_frames_num, video_start_fr=None, video_end_fr=None):
    if max_frames_num == 0:
        return Image.new("RGB", (224, 224), (0, 0, 0))
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    start_frame = int(max(0, video_start_fr) if video_start_fr is not None else 0)
    end_frame = int(min(total_frame_num - 1, video_end_fr) if video_end_fr is not None else total_frame_num - 1)
    uniform_sampled_frames = np.linspace(start_frame, end_frame, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    spare_frames = [Image.fromarray(cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)) for frame in spare_frames]
    return spare_frames

def inference(model, image_processor, video_processor, query, **kwargs):

    message = ""
    images = None

    if "retrieved_texts" in kwargs:
        for index, text in enumerate(kwargs["retrieved_texts"]):
            message += f"Relevant document {index+1}:\n{text}\n"
        query = "Considering the given documents,\n" + query

    elif "retrieved_images" in kwargs:
        images = []
        for index, image_path in enumerate(kwargs["retrieved_images"]):
            if kwargs.get("use_caption", False):
                message += f"Relevant image {index+1}:\n<|image_{index + 1}|>\n{kwargs['img_metadata'][image_path]['caption']}\n"
            else:
                message += f"<|image_{index + 1}|>\n"
            images.append(Image.open(image_path))
        query = "Considering the given images,\n" + query

    elif "retrieved_videos" in kwargs:
        images = []
        nframes = kwargs.get("nframes", 32)
        for index, video_path in enumerate(kwargs["retrieved_videos"]):
            if kwargs.get("use_scripts", False):
                script = get_scripts_for_videos([video_path], [kwargs.get("startend_times")[index]])[0]
                message += f"Relevant video {index+1}:\n{script}\n"
            message += ''.join(f"<|image_{index * nframes + fid + 1}|>\n" for fid in range(nframes))
            frames = load_video(video_path, nframes, video_start_fr=kwargs.get("startend_frames", [[None, None]])[index][0], video_end_fr=kwargs.get("startend_frames", [[None, None]])[index][1])
            images.extend(frames)
        query = "Considering the given videos,\n" + query

    messages = [
        {
            "role": "user",
            "content": message + query
        }
    ]

    processor = video_processor if kwargs.get("retrieved_videos") else image_processor
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, images=images, return_tensors='pt').to('cuda')

    generation_config = {"max_new_tokens": 1024, "temperature": 0.0, "do_sample": False}

    output_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_config)
    output_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    outputs = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    del inputs, output_ids
    torch.cuda.empty_cache()

    return outputs.strip()
