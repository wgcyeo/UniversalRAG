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

def extract_text_by_time_range(filepath, start_seconds, end_seconds):
    def time_to_seconds(time_str):
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    text_lines = []
    within_range = False
    for line in lines:
        time_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
        if time_match:
            start, end = time_match.groups()
            start_sec = time_to_seconds(start)
            end_sec = time_to_seconds(end)
            within_range = start_seconds <= end_sec and end_seconds >= start_sec
            continue
        if within_range:
            if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
                continue
            if line.strip() == '':
                continue
            clean_line = re.sub(r'<[^>]+>', '', line).strip()
            if clean_line:
                text_lines.append(clean_line)

    return ' '.join(text_lines)

def extract_clipscript_feats(input_path, output_path, disable_prog=False):
    """
    Extract clip script features and save them as a pickle file.

    Args:
        input_path (str): Path to the pickle file with frame clipping time information.
        output_path (str): Path to save the pickle file.
        disable_prog (bool): Whether to disable tqdm progress bars.
    """

    with open(input_path, 'rb') as f:
        frame_data = pickle.load(f)
    all_clips = list(frame_data.keys())

    features = {}
    for clip in tqdm(all_clips, desc="Processing clip scripts", disable=disable_prog):
        video_path = clip.rsplit('_', 1)[0]
        script_path = video_path.replace('videos', 'scripts').replace('.mp4', '.en.vtt')
        start_time, end_time = map(int, frame_data[clip])
        text_data = extract_text_by_time_range(script_path, start_time, end_time)
        features[clip] = intern_model.get_txt_feat(text_data).cpu().numpy().squeeze(0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser(description="Extract clip script features and save them as a pickle file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the pickle file with frame clipping time information.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the pickle file.")
    parser.add_argument("--disable_prog", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    extract_clipscript_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        disable_prog=args.disable_prog
    )
