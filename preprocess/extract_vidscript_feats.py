import os
import sys
import re
import pickle
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

def extract_text_from_file(filepath):
    if filepath.endswith('.en.vtt'):
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        text_lines = []
        for line in lines:
            if re.match(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', line):
                continue
            if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
                continue
            if line.strip() == '':
                continue
            clean_line = re.sub(r'<[^>]+>', '', line).strip()
            if clean_line:
                text_lines.append(clean_line)

        return ' '.join(text_lines)
    elif filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read().strip()
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

def extract_vidscript_feats(input_path, output_path, disable_prog=False):
    """
    Extract video script features and save them as a pickle file.

    Args:
        input_path (str): Path to the directory containing 'videos' and 'scripts' subfolders.
        output_path (str): Path to save the pickle file.
        disable_prog (bool): Whether to disable tqdm progress bars.
    """
    videos_dir = os.path.join(input_path, "videos")
    scripts_dir = os.path.join(input_path, "scripts")
    all_videos = sorted([f for f in os.listdir(videos_dir) if f.endswith('.mp4')])

    features = {}
    for video in tqdm(all_videos, desc="Processing video scripts", disable=disable_prog):
        video_id = os.path.splitext(video)[0]
        video_path = os.path.join(videos_dir, video)
        text_file_vtt = os.path.join(scripts_dir, video_id + '.en.vtt')
        text_file_txt = os.path.join(scripts_dir, video_id + '.txt')

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        if not os.path.exists(text_file_vtt) and not os.path.exists(text_file_txt):
            print(f"Text not found: {text_file_vtt} or {text_file_txt}")
            continue

        text_file = text_file_vtt if os.path.exists(text_file_vtt) else text_file_txt
        text_data = extract_text_from_file(text_file)
        features[video_path] = intern_model.get_txt_feat(text_data).cpu().numpy().squeeze(0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Extract video script features and save them as a pickle file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the directory containing 'videos' and 'scripts' subfolders.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the pickle file.")
    parser.add_argument("--disable_prog", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    extract_vidscript_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        disable_prog=args.disable_prog
    )
