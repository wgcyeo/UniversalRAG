import os
import sys
import cv2
import pickle
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

if not os.getenv('INTERNVIDEO_PATH'):
    raise EnvironmentError("Environment variable `INTERNVIDEO_PATH` is not set.")
internvideo2_path = os.path.join(os.getenv('INTERNVIDEO_PATH'), 'InternVideo2/multi_modality')
sys.path.append(internvideo2_path)

from utils.config import Config, eval_dict_leaf
from demo.utils import _frame_from_video, setup_internvideo2, frames2tensor

device = 'cuda'

config = Config.from_file(os.path.join(internvideo2_path, 'demo/internvideo2_stage2_config.py'))
config = eval_dict_leaf(config)
config.model.vision_encoder.pretrained = os.path.join(internvideo2_path, config.model.vision_encoder.pretrained)
config.model.text_encoder.config = os.path.join(internvideo2_path, config.model.text_encoder.config)
config.pretrained_path = os.path.join(internvideo2_path, config.pretrained_path)
intern_model, _ = setup_internvideo2(config)
intern_model.to(device)

def ensure_length_16(frames):
    multiplier = 16 // len(frames)
    remainder = 16 % len(frames)
    extended_frames = [frame for frame in frames for _ in range(multiplier)]
    extended_frames.extend(frames[:remainder])
    return extended_frames

def extract_video_feats(input_path, output_path, num_splits=1, split_index=None, disable_prog=False):
    """
    Extract video features for all .mp4 files in a directory and save them as a pickle file.

    Args:
        input_path (str): Path to the directory containing video files.
        output_path (str): Path to save the pickle file.
        num_splits (int): Number of splits to divide the total files into.
        split_index (int, optional): Index of the split to process (0-based).
        disable_prog (bool): Whether to disable tqdm progress bars.
    """
    all_videos = sorted([f for f in os.listdir(input_path) if f.endswith('.mp4')])
    total_videos = len(all_videos)

    if num_splits <= 0:
        raise ValueError("num_splits must be a positive integer.")
    if split_index is not None and (split_index < 0 or split_index >= num_splits):
        raise ValueError("split_index must be between 0 and num_splits - 1.")

    split_size = (total_videos + num_splits - 1) // num_splits

    if split_index is None:
        split_videos = all_videos
    else:
        split_start = split_index * split_size
        split_end = min(split_start + split_size, total_videos)
        split_videos = all_videos[split_start:split_end]

    features = {}
    for filename in tqdm(split_videos, desc=f"Processing {'all videos' if split_index is None else f'split {split_index + 1}/{num_splits}'}", disable=disable_prog):
        filepath = os.path.join(input_path, filename)
        video = cv2.VideoCapture(filepath)
        frames = [x for x in _frame_from_video(video)]
        video.release()

        if not frames:
            print(f'[ERROR] Video length is zero: {filepath}')
            continue

        if len(frames) < 16:
            frames = ensure_length_16(frames)

        fn = config.get('num_frames', 16)
        size_t = config.get('size_t', 224)
        frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)
        del frames

        feature = intern_model.get_vid_feat(frames_tensor).cpu().numpy().squeeze(0)
        del frames_tensor
        features[filepath] = feature

    if split_index is not None:
        base, ext = os.path.splitext(output_path)
        split_output_path = f"{base}_split{split_index + 1}{ext}"
    else:
        split_output_path = output_path

    os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
    with open(split_output_path, 'wb') as f:
        pickle.dump(features, f)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Extract video features and save them as a pickle file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the directory containing .mp4 files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the pickle file.")
    parser.add_argument("--num_splits", type=int, default=1, help="Number of splits to divide the total files into.")
    parser.add_argument("--split_index", type=int, default=None, help="Index of the split to process (0-based).")
    parser.add_argument("--disable_prog", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    extract_video_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        num_splits=args.num_splits,
        split_index=args.split_index,
        disable_prog=args.disable_prog
    )
