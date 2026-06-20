import os
import pickle
import pandas as pd
from tqdm import tqdm

from universalrag.embedding import EmbeddingModel

model = EmbeddingModel()
tokenizer = model.text_model.model.tokenizer

def split_text_into_chunks(text, max_tokens):
    """
    Split a text into chunks of up to max_tokens using the tokenizer.

    Args:
        text (str): The input text to split.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    token_ids = tokenizer.encode(text, truncation=False)
    tokenized_chunks = [token_ids[i:i + max_tokens] for i in range(0, len(token_ids), max_tokens)]
    chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in tokenized_chunks]
    return chunks

def extract_text_feats(input_path, output_path, num_splits=1, split_index=None, long_text=False, max_tokens=512, batch_size=32):
    """
    Extract features from a parquet file and save them as a pickle file.

    Args:
        input_path (str): Path to the parquet file with 'psg_id' and 'text' columns.
        output_path (str): Path to save the pickle file.
        num_splits (int): Number of splits to divide the total rows into.
        split_index (int, optional): Index of the split to process (0-based).
        long_text (bool): Whether to split text into chunks.
        max_tokens (int): Maximum number of tokens per text chunk if long_text is `True`.
        batch_size (int): Batch size for encoding.
    """
    df = pd.read_parquet(input_path, engine='fastparquet')
    df = df.sort_index()
    total_rows = len(df)

    if num_splits <= 0:
        raise ValueError("num_splits must be a positive integer.")
    if split_index is not None and (split_index < 0 or split_index >= num_splits):
        raise ValueError("split_index must be between 0 and num_splits - 1.")

    split_size = (total_rows + num_splits - 1) // num_splits

    if split_index is None:
        split_df = df
    else:
        split_start = split_index * split_size
        split_end = min(split_start + split_size, total_rows)
        split_df = df.iloc[split_start:split_end]

    features = {}
    texts = []
    psg_ids = []
    for psg_id, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f'Processing {"all text" if split_index is None else f"split {split_index + 1}/{num_splits}"}'):
        text = row['text']
        if long_text:
            chunks = split_text_into_chunks(text, max_tokens)
            for i, chunk in enumerate(chunks):
                chunk_psg_id = f"{psg_id}_part{i + 1}"
                if chunk.strip() == "":
                    continue
                texts.append(chunk)
                psg_ids.append(chunk_psg_id)
        else:
            texts.append(text)
            psg_ids.append(psg_id)

    encoded_features = model.encode(texts, batch_size=batch_size)
    for psg_id, feature in zip(psg_ids, encoded_features):
        features[psg_id] = feature

    if split_index is not None:
        base, ext = os.path.splitext(output_path)
        split_output_path = f"{base}_split{split_index + 1}{ext}"
    else:
        split_output_path = output_path

    os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
    with open(split_output_path, 'wb') as f:
        pickle.dump(features, f)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract text features from a parquet file and save them as a pickle file.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the parquet file with 'psg_id' and 'text' columns.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the pickle file.")
    parser.add_argument("--num-splits", type=int, default=1, help="Number of splits to divide the total rows into.")
    parser.add_argument("--split-index", type=int, default=None, help="Index of the split to process (0-based).")
    parser.add_argument("--long-text", action="store_true", help="Split text into chunks.")
    args = parser.parse_args()

    extract_text_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        num_splits=args.num_splits,
        split_index=args.split_index,
        long_text=args.long_text
    )
