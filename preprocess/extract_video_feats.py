import os
import re
import pickle
import pandas as pd
from tqdm import tqdm

from universalrag.embedding import EmbeddingModel

embedding_model = EmbeddingModel()

def extract_text_from_file(filepath):
    """Extract text from VTT or TXT file."""
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

def extract_video_feats(input_path, output_path, num_splits=1, split_index=None):
    """
    Extract video features from a parquet file containing video paths and save them as a pickle file.
    Script features are extracted only on the first split (split_index == 0 or None).

    Args:
        input_path (str): Path to the parquet file containing video paths in index.
        output_path (str): Path to save the pickle file.
        num_splits (int): Number of splits to divide the total files into.
        split_index (int, optional): Index of the split to process (0-based).
    """
    df = pd.read_parquet(input_path)
    all_videos = df.index.tolist()
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

    # Only encode scripts on first iteration (split_index == 0 or None)
    if split_index == 0 or split_index is None:
        script_features = {}
        for video_path in tqdm(all_videos, desc="Processing video scripts"):
            script_path = os.path.dirname(os.path.dirname(video_path))
            scripts_dir = os.path.join(script_path, 'scripts')
            video_id = os.path.splitext(os.path.basename(video_path))[0]
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
            script_features[video_path] = embedding_model.encode_vis_query(text_data).squeeze(0)
        
        # Save script features with suffix
        base, ext = os.path.splitext(output_path)
        script_output_path = f"{base}_vidscript{ext}"
        os.makedirs(os.path.dirname(script_output_path), exist_ok=True)
        with open(script_output_path, 'wb') as f:
            pickle.dump(script_features, f)
    
    # Process video features
    features = {}
    for filename in tqdm(split_videos, desc=f"Processing {'all videos' if split_index is None else f'split {split_index + 1}/{num_splits}'}"):
        try:
            feature = embedding_model.encode_video(os.path.abspath(filename)).squeeze(0)
            features[filename] = feature
        except Exception as e:
            print(f"\033[91mError processing {filename}: {e}\033[0m")

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
    parser.add_argument("--input-path", type=str, required=True, help="Path to the directory containing 'videos' and optionally 'scripts' subfolders.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the pickle file.")
    parser.add_argument("--num-splits", type=int, default=1, help="Number of splits to divide the total files into.")
    parser.add_argument("--split-index", type=int, default=None, help="Index of the split to process (0-based).")
    args = parser.parse_args()

    extract_video_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        num_splits=args.num_splits,
        split_index=args.split_index
    )
