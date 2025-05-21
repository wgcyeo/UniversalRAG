import os
import sys
import cv2
import numpy as np
import torch
import pickle
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

if not os.getenv('INTERNVIDEO_PATH'):
    raise EnvironmentError("Environment variable `INTERNVIDEO_PATH` is not set.")
internvideo2_path = os.path.join(os.getenv('INTERNVIDEO_PATH'), 'InternVideo2/multi_modality')
sys.path.append(internvideo2_path)

from utils.config import Config, eval_dict_leaf
from demo.utils import setup_internvideo2

device = 'cuda'

config = Config.from_file(os.path.join(internvideo2_path, 'demo/internvideo2_stage2_config.py'))
config = eval_dict_leaf(config)
config.model.vision_encoder.pretrained = os.path.join(internvideo2_path, config.model.vision_encoder.pretrained)
config.model.text_encoder.config = os.path.join(internvideo2_path, config.model.text_encoder.config)
config.pretrained_path = os.path.join(internvideo2_path, config.pretrained_path)
intern_model, _ = setup_internvideo2(config)
intern_model.to(device)

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

def normalize(data):
    return (data / 255.0 - v_mean) / v_std

def image2tensor(image, target_size=(224, 224), device=torch.device('cuda')):
    image = cv2.resize(image[:, :, ::-1], target_size)
    image_tensor = np.expand_dims(normalize(image), axis=(0, 1))
    image_tensor = np.transpose(image_tensor, (0, 1, 4, 2, 3))
    image_tensor = torch.from_numpy(image_tensor).to(device, non_blocking=True).float()
    return image_tensor

def extract_image_feats(input_path, output_path, num_splits=4, split_index=None, disable_prog=False):
    """
    Extract image features and save them as a pickle file.

    Args:
        input_path (str): Path to the directory containing image metadata.
        output_path (str): Path to save the pickle file.
        num_splits (int): Number of splits to divide the total files into.
        split_index (int, optional): Index of the split to process (0-based).
        disable_prog (bool): Whether to disable tqdm progress bars.
    """
    with open(input_path, 'r') as f:
        image_metadata = json.load(f)
    all_images = list(image_metadata.keys())
    total_images = len(all_images)

    if num_splits <= 0:
        raise ValueError("num_splits must be a positive integer.")
    if split_index is not None and (split_index < 0 or split_index >= num_splits):
        raise ValueError("split_index must be between 0 and num_splits - 1.")

    split_size = (total_images + num_splits - 1) // num_splits

    if split_index is None:
        split_images = all_images
    else:
        split_start = split_index * split_size
        split_end = min(split_start + split_size, total_images)
        split_images = all_images[split_start:split_end]

    if split_index == 0 or split_index == None:
        image_caps_feats = {}
        for image_path, image_meta in image_metadata.items():
            image_cap_feat = intern_model.get_txt_feat(image_meta["caption"]).cpu().numpy().squeeze(0)
            image_caps_feats[image_path] = image_cap_feat
        base, ext = os.path.splitext(output_path)
        cap_output_path = f"{base}_imgcap{ext}"
        os.makedirs(os.path.dirname(cap_output_path), exist_ok=True)
        with open(cap_output_path, 'wb') as f:
            pickle.dump(image_caps_feats, f)

    image_feats = {}
    for image_path in tqdm(split_images, desc=f"Processing {'all images' if split_index is None else f'split {split_index + 1}/{num_splits}'}", disable=disable_prog):
        image = cv2.imread(image_path)
        if image is None:
            print(f'[ERROR] Unable to read image: {image_path}')
            continue

        size_t = config.get('size_t', 224)
        image_tensor = image2tensor(image, target_size=(size_t, size_t), device=device)
        image_feature = intern_model.get_vid_feat(image_tensor).cpu().numpy().squeeze(0)
        image_feats[image_path] = image_feature

    if split_index is not None:
        base, ext = os.path.splitext(output_path)
        split_output_path = f"{base}_split{split_index + 1}{ext}"
    else:
        split_output_path = output_path

    os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
    with open(split_output_path, 'wb') as f:
        pickle.dump(image_feats, f)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Extract image features and save them as a pickle file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the directory containing image metadata.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the pickle file.")
    parser.add_argument("--num_splits", type=int, default=1, help="Number of splits to divide the total files into.")
    parser.add_argument("--split_index", type=int, default=None, help="Index of the split to process (0-based).")
    parser.add_argument("--disable_prog", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    extract_image_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        num_splits=args.num_splits,
        split_index=args.split_index,
        disable_prog=args.disable_prog
    )
