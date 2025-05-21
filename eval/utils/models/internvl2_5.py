from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

import warnings

warnings.filterwarnings("ignore")

from utils.utils import get_scripts_for_videos

def load_model(model_path):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, None, tokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound[0] and bound[1]:
        start_idx = max(first_idx, bound[0])
        end_idx = min(bound[1], max_frame)
    else:
        start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def inference(model, processor, tokenizer, query, **kwargs):

    message = ""
    pixel_values = None
    generation_kwargs = {"history": None, "return_history": False}

    if "retrieved_texts" in kwargs:
        for index, text in enumerate(kwargs["retrieved_texts"]):
            message += f"Relevant document {index+1}:\n{text}\n"
        query = "Considering the given documents,\n" + query

    elif "retrieved_images" in kwargs:
        pixel_values = []
        for index, image_path in enumerate(kwargs["retrieved_images"]):
            pixel_values.append(load_image(image_path).to(torch.bfloat16).cuda())
            if kwargs.get("use_caption", False):
                message += f"Relevant image {index+1}:\n<image>\n{kwargs['img_metadata'][image_path]['caption']}\n"
            else:
                message += "<image>\n"
        generation_kwargs["num_patches_list"] = [pixel_value.size(0) for pixel_value in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)
        query = "Considering the given images,\n" + query

    elif "retrieved_videos" in kwargs:
        pixel_values = []
        num_patches_list = []
        for index, video_path in enumerate(kwargs["retrieved_videos"]):
            pixel_value, num_patches = load_video(video_path, bound=kwargs.get("startend_frames", [[None, None]])[index], num_segments=kwargs.get("nframes", 32))
            pixel_values.append(pixel_value)
            num_patches_list.append(num_patches)
            message += f"Relevant video {index+1}:\n" + ''.join([f'Frame-{i+1}: <image>\n' for i in range(len(num_patches))])
            if kwargs.get("use_scripts", False):
                script = get_scripts_for_videos([video_path], [kwargs.get("startend_times")[index]])[0]
                message += script + "\n"
        pixel_values = torch.cat(pixel_values).to(torch.bfloat16).cuda()
        num_patches_list = [item for sublist in num_patches_list for item in sublist]
        generation_kwargs["num_patches_list"] = num_patches_list
        query = "Considering the given videos,\n" + query

    message += query
    
    generation_config = dict(max_new_tokens=kwargs.get("max_new_tokens", 1024), do_sample=True)
    outputs = model.chat(tokenizer, pixel_values, message, generation_config, **generation_kwargs)
    return outputs.strip()
