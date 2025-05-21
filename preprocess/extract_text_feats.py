import os
import pickle
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('BAAI/bge-large-en-v1.5')
tokenizer = model.tokenizer

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

def extract_text_feats(input_path, output_path, num_splits=1, split_index=None, disable_prog=False, long_text=False, max_tokens=512, batch_size=256):
    """
    Extract features for all .txt files in a directory and save them as a pickle file.

    Args:
        input_path (str): Path to the directory containing .txt files.
        output_path (str): Path to save the pickle file.
        num_splits (int): Number of splits to divide the total files into.
        split_index (int, optional): Index of the split to process (0-based).
        disable_prog (bool): Whether to disable tqdm progress bars.
        long_text (bool): Whether to split text into chunks.
        max_tokens (int): Maximum number of tokens per text chunk if long_text is `True`.
        batch_size (int): Batch size for encoding.
    """
    all_texts = sorted([f for f in os.listdir(input_path) if f.endswith('.txt')])
    total_texts = len(all_texts)

    if num_splits <= 0:
        raise ValueError("num_splits must be a positive integer.")
    if split_index is not None and (split_index < 0 or split_index >= num_splits):
        raise ValueError("split_index must be between 0 and num_splits - 1.")

    split_size = (total_texts + num_splits - 1) // num_splits

    if split_index is None:
        split_texts = all_texts
    else:
        split_start = split_index * split_size
        split_end = min(split_start + split_size, total_texts)
        split_texts = all_texts[split_start:split_end]

    features = {}
    texts = []
    filepaths = []
    for filename in tqdm(split_texts, desc=f'Processing {"all text" if split_index is None else f"split {split_index + 1}/{num_splits}"}', disable=disable_prog):
        filepath = os.path.join(input_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        if long_text:
            chunks = split_text_into_chunks(text, max_tokens)
            for i, chunk in enumerate(chunks):
                chunk_filepath = f"{filepath}_part{i + 1}"
                texts.append(chunk)
                filepaths.append(chunk_filepath)
        else:
            texts.append(text)
            filepaths.append(filepath)

    with torch.no_grad():
        encoded_features = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=not disable_prog)

    for filepath, feature in zip(filepaths, encoded_features):
        features[filepath] = feature

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

    parser = argparse.ArgumentParser(description="Extract text features and save them as a pickle file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the directory containing .txt files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the pickle file.")
    parser.add_argument("--num_splits", type=int, default=1, help="Number of splits to divide the total files into.")
    parser.add_argument("--split_index", type=int, default=None, help="Index of the split to process (0-based).")
    parser.add_argument("--disable_prog", action="store_true", help="Disable progress bars.")
    parser.add_argument("--long_text", action="store_true", help="Split text into chunks.")
    args = parser.parse_args()

    extract_text_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        num_splits=args.num_splits,
        split_index=args.split_index,
        disable_prog=args.disable_prog,
        long_text=args.long_text
    )
