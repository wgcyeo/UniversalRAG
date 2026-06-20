import os
import pickle
import pandas as pd
from tqdm import tqdm
import re
from typing import List, Tuple

from universalrag.embedding import EmbeddingModel

model = EmbeddingModel()

def parse_table_file(filepath: str) -> Tuple[str, List[str], List[List[str]]]:
    """
    Parse a table file and extract table name, headers, and rows.
    
    Args:
        filepath (str): Path to the table file.
        
    Returns:
        Tuple[str, List[str], List[List[str]]]: Table name, headers, and rows.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    lines = content.split('\n')
    
    # Extract table name (first line after "Table:")
    table_name = ""
    for line in lines:
        if line.startswith("Table:"):
            table_name = line.replace("Table:", "").strip()
            break
    
    # Find the header line and data lines
    headers = []
    rows = []
    
    found_separator = False
    for i, line in enumerate(lines):
        # Skip empty lines and the table name line
        if not line.strip() or line.startswith("Table:"):
            continue
            
        # Look for separator line (dashes)
        if re.match(r'^-+$', line.strip()):
            found_separator = True
            continue
        
        # Split by | character and clean up
        if '|' in line:
            parts = [part.strip() for part in line.split('|')]
            # Remove empty parts from beginning/end
            parts = [part for part in parts if part]
            
            if not found_separator and not headers:
                # This is the header line
                headers = parts
            elif found_separator:
                # This is a data row
                if len(parts) == len(headers):  # Only include complete rows
                    rows.append(parts)
    
    return table_name, headers, rows

def create_dense_row_text(headers: List[str], row: List[str], table_name: str = "") -> str:
    """
    Create a textual representation of a table row for embedding.
    Uses the format: "[column name] is [cell value], [column name] is [cell value]"
    
    Args:
        headers (List[str]): Column headers.
        row (List[str]): Row values.
        table_name (str): Name of the table.
        
    Returns:
        str: Textual representation of the row.
    """
    # Create header-value pairs in the format "[column name] is [cell value]"
    pairs = []
    for header, value in zip(headers, row):
        if value and value.strip():  # Only include non-empty values
            pairs.append(f"{header} is {value}")
    
    # Combine into a single text representation with commas
    row_text = ", ".join(pairs)
    
    # Optionally prepend table name for context
    if table_name:
        row_text = f"{table_name}: {row_text}"
    
    return row_text

def extract_table_feats(input_path, output_path, batch_size=16, num_splits=4, split_index=None, include_table_name=True):
    """
    Extract dense row embeddings for all table from a parquet file and save them as a pickle file.
    Uses row serialization format: "[column name] is [cell value], [column name] is [cell value]"

    Args:
        input_path (str): Path to the .parquet file.
        output_path (str): Path to save the pickle file.
        batch_size (int): Batch size for encoding.
        num_splits (int): Number of splits to divide the total files into.
        split_index (int, optional): Index of the split to process (0-based).
        include_table_name (bool): Whether to include table name in row text.
    """
    df = pd.read_parquet(input_path)
    
    if 'text' in df.columns:
        # Parse tables and create row embeddings
        all_texts = []
        all_identifiers = []
        
        for idx, row in df.iterrows():
            table_text = row['text']
            
            # Parse the table text to extract rows
            try:
                lines = table_text.strip().split('\n')
                table_name = ""
                headers = []
                rows = []
                
                found_separator = False
                for line in lines:
                    if line.startswith("Table:"):
                        table_name = line.replace("Table:", "").strip()
                        continue
                    
                    if not line.strip():
                        continue
                    
                    if re.match(r'^-+$', line.strip()):
                        found_separator = True
                        continue
                    
                    if '|' in line:
                        parts = [part.strip() for part in line.split('|')]
                        parts = [part for part in parts if part]
                        
                        if not found_separator and not headers:
                            headers = parts
                        elif found_separator and len(parts) == len(headers):
                            rows.append(parts)
                
                # Create dense row embeddings for each row
                for row_idx, data_row in enumerate(rows):
                    row_text = create_dense_row_text(
                        headers, data_row,
                        table_name if include_table_name else ""
                    )
                    all_texts.append(row_text)
                    all_identifiers.append(f"{idx}_row{row_idx}")
                    
            except Exception as e:
                print(f"Warning: Failed to process table {idx}: {str(e)}")
                continue
    else:
        raise ValueError(f"Parquet file must contain 'text' column for table data. Available columns: {list(df.columns)}")
    
    total_items = len(all_texts)

    if num_splits <= 0:
        raise ValueError("num_splits must be a positive integer.")
    if split_index is not None and (split_index < 0 or split_index >= num_splits):
        raise ValueError("split_index must be between 0 and num_splits - 1.")

    split_size = (total_items + num_splits - 1) // num_splits

    if split_index is None:
        split_texts = all_texts
        split_identifiers = all_identifiers
    else:
        split_start = split_index * split_size
        split_end = min(split_start + split_size, total_items)
        split_texts = all_texts[split_start:split_end]
        split_identifiers = all_identifiers[split_start:split_end]

    features = {}
    texts = []
    filepaths = []

    for idx, (identifier, text) in enumerate(tqdm(zip(split_identifiers, split_texts), desc=f'Processing {"all items" if split_index is None else f"split {split_index + 1}/{num_splits}"}', total=len(split_identifiers))):
        texts.append(text)
        filepaths.append(identifier)

    encoded_features = model.encode_text(texts, batch_size=batch_size)

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

    parser = argparse.ArgumentParser(description="Extract table features using dense row embeddings and save them as a pickle file.")
    parser.add_argument("--input-path", type=str, help="Path to the .parquet file.")
    parser.add_argument("--output-path", type=str, help="Path to save the pickle file.")
    parser.add_argument("--num-splits", type=int, default=4, help="Number of splits to divide the total files into.")
    parser.add_argument("--split-index", type=int, help="Index of the split to process (0-based).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding.")
    parser.add_argument("--include-table-name", action="store_true", default=True, help="Include table name in row text representation.")
    args = parser.parse_args()

    extract_table_feats(
        args.input_path,
        args.output_path,
        num_splits=args.num_splits,
        split_index=args.split_index,
        batch_size=args.batch_size,
        include_table_name=args.include_table_name
    )