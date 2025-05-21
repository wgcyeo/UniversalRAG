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

def extract_clip_feats(input_path, output_path, num_splits=4, split_index=None, disable_prog=False):
    """
    Extract clip features and save them as a pickle file.

    Args:
        input_path (str): Path to the pickle file with frame clipping information.
        output_path (str): Path to save the pickle file.
        num_splits (int): Number of splits to divide the total files into.
        split_index (int, optional): Index of the split to process (0-based).
        disable_prog (bool): Whether to disable tqdm progress bars.
    """

    with open(input_path, 'rb') as f:
        frame_data = pickle.load(f)
    all_clips = list(frame_data.keys())
    total_clips = len(all_clips)

    if num_splits <= 0:
        raise ValueError("num_splits must be a positive integer.")
    if split_index is not None and (split_index < 0 or split_index >= num_splits):
        raise ValueError("split_index must be between 0 and num_splits - 1.")

    split_size = (total_clips + num_splits - 1) // num_splits

    if split_index is None:
        split_clips = all_clips
    else:
        split_start = split_index * split_size
        split_end = min(split_start + split_size, total_clips)
        split_clips = all_clips[split_start:split_end]

    features = {}
    for clip_path in tqdm(split_clips, desc=f"Processing {'all clips' if split_index is None else f'split {split_index + 1}/{num_splits}'}", disable=disable_prog):
        video_path = clip_path.rsplit('_', 1)[0]
        start_frame, end_frame = map(int, frame_data[clip_path])
        video = cv2.VideoCapture(video_path)
        frames = []
        frame_number = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if start_frame <= frame_number <= end_frame:
                frames.append(frame)
            elif frame_number > end_frame:
                break
            frame_number += 1
        video.release()

        if not frames:
            print(f'[ERROR] Empty frames for clip: {clip_path}')
            continue

        if len(frames) < 16:
            frames = ensure_length_16(frames)

        fn = config.get('num_frames', 16)
        size_t = config.get('size_t', 224)
        frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)
        del frames

        feature = intern_model.get_vid_feat(frames_tensor).cpu().numpy().squeeze(0)
        del frames_tensor
        features[clip_path] = feature

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

    parser = argparse.ArgumentParser(description="Extract clip features and save them as a pickle file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the pickle file with frame clipping information.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the pickle file.")
    parser.add_argument("--num_splits", type=int, default=4, help="Number of splits to divide the total files into.")
    parser.add_argument("--split_index", type=int, default=None, help="Index of the split to process (0-based).")
    parser.add_argument("--disable_prog", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    extract_clip_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        num_splits=args.num_splits,
        split_index=args.split_index,
        disable_prog=args.disable_prog,
    )
