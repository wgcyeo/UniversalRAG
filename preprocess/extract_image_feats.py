import os
import pickle
import pandas as pd

from universalrag.embedding import EmbeddingModel

embedding_model = EmbeddingModel()

def extract_image_feats(input_path, output_path, num_splits=4, split_index=None):
    """
    Extract image features and save them as a pickle file.

    Args:
        input_path (str): Path to the parquet file containing image metadata.
        output_path (str): Path to save the pickle file.
        num_splits (int): Number of splits to divide the total files into.
        split_index (int, optional): Index of the split to process (0-based).
    """
    df = pd.read_parquet(input_path, engine='fastparquet')
    all_images = df.index.tolist()
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

    encoded_features = embedding_model.encode_image(split_images)

    features = {}
    for image_path, feature in zip(split_images, encoded_features):
        features[image_path] = feature

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

    parser = argparse.ArgumentParser(description="Extract image features and save them as a pickle file.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the parquet file containing image metadata.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the pickle file.")
    parser.add_argument("--num-splits", type=int, default=1, help="Number of splits to divide the total files into.")
    parser.add_argument("--split-index", type=int, default=None, help="Index of the split to process (0-based).")
    args = parser.parse_args()

    extract_image_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        num_splits=args.num_splits,
        split_index=args.split_index
    )
