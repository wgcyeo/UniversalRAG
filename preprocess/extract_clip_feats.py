import os
import re
import pickle
import warnings
import tempfile
import cv2
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from universalrag.embedding import EmbeddingModel

embedding_model = EmbeddingModel()

def extract_video_clip(video_path, start_frame, end_frame, output_path):
    """Extract a clip from a video and save it to a temporary file."""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_number = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if start_frame <= frame_number <= end_frame:
            out.write(frame)
        elif frame_number > end_frame:
            break
        frame_number += 1
    
    video.release()
    out.release()
    return frame_number > start_frame  # Return True if any frames were written

def extract_text_by_time_range(filepath, start_seconds, end_seconds):
    """Extract text from VTT/TXT file within a time range."""
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

def extract_clip_feats(input_path, output_path, num_splits=4, split_index=None):
    """
    Extract clip features and save them as a pickle file.
    Script features are extracted only on the first split (split_index == 0 or None).

    Args:
        input_path (str): Path to the parquet file with clip_id, frame_num, and frame_time columns.
        output_path (str): Path to save the pickle file.
        num_splits (int): Number of splits to divide the total files into.
        split_index (int, optional): Index of the split to process (0-based).
    """

    # Load clip data from parquet file
    clip_df = pd.read_parquet(input_path)
    frame_data = clip_df['frame_num'].to_dict()
    time_data = clip_df['frame_time'].to_dict()
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

    # Only encode scripts on first iteration (split_index == 0 or None)
    if split_index == 0 or split_index is None:
        script_features = {}
        for clip in tqdm(all_clips, desc="Processing clip scripts"):
            video_path = clip.rsplit('_', 1)[0]
            vtt_path = video_path.replace('videos', 'scripts').replace('.mp4', '.en.vtt')
            txt_path = video_path.replace('videos', 'scripts').replace('.mp4', '.txt')
            start_time, end_time = map(int, time_data[clip])

            text_data = ''
            if os.path.exists(vtt_path):
                text_data = extract_text_by_time_range(vtt_path, start_time, end_time)
            elif os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as sf:
                    text_data = sf.read()
            else:
                print(f"[WARN] No script found for {video_path}: looked for {vtt_path} and {txt_path}")

            if text_data is None or text_data.strip() == '':
                print(f"[WARN] No text extracted for {clip}: time range {start_time}-{end_time}")
                text_data = 'video script not found'

            script_features[clip] = embedding_model.encode_vis_query(text_data).squeeze(0)
        
        # Save script features with suffix
        base, ext = os.path.splitext(output_path)
        script_output_path = f"{base}_clipscript{ext}"
        os.makedirs(os.path.dirname(script_output_path), exist_ok=True)
        with open(script_output_path, 'wb') as f:
            pickle.dump(script_features, f)
    
    # Process video clip features
    features = {}
    for clip_path in tqdm(split_clips, desc=f"Processing {'all clips' if split_index is None else f'split {split_index + 1}/{num_splits}'}"):
        video_path = clip_path.rsplit('_', 1)[0]
        start_frame, end_frame = map(int, frame_data[clip_path])
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = extract_video_clip(video_path, start_frame, end_frame, tmp_path)
            if not success:
                print(f'[ERROR] Empty frames for clip: {clip_path}')
                continue
            
            feature = embedding_model.encode_video(tmp_path).squeeze(0)
            features[clip_path] = feature
        except Exception as e:
            print(f"\033[91mError processing {tmp_path}/{clip_path}: {e}\033[0m")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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
    parser.add_argument("--input-path", type=str, required=True, help="Path to the parquet file with clip_id, frame_num, and frame_time columns.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the pickle file.")
    parser.add_argument("--num-splits", type=int, default=4, help="Number of splits to divide the total files into.")
    parser.add_argument("--split-index", type=int, default=None, help="Index of the split to process (0-based).")
    args = parser.parse_args()

    extract_clip_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        num_splits=args.num_splits,
        split_index=args.split_index,
    )
