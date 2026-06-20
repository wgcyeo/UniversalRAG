import os
import pandas as pd
import argparse
from tqdm import tqdm

def merge_datasets(input_files, output_file):
    """
    Merge multiple parquet dataset files into a single parquet file.
    
    Args:
        input_files: List of paths to input parquet files
        output_file: Path to output merged parquet file
    """
    dataframes = []
    
    for file_path in tqdm(input_files, desc="Loading parquet files"):
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path, engine='fastparquet')
                # Reset index to get psg_id as a column if it's currently the index
                if df.index.name == 'psg_id':
                    df = df.reset_index()
                dataframes.append(df)
                print(f"Loaded {len(df)} entries from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not dataframes:
        raise ValueError("No valid parquet files were loaded.")
    
    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Deduplicate by psg_id if it exists
    if 'psg_id' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['psg_id'])
        merged_df.set_index('psg_id', inplace=True)
    
    # Save merged dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_parquet(output_file)
    print(f"Successfully saved {len(merged_df)} entries to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple parquet dataset files into one.")
    parser.add_argument('--input-files', type=str, nargs='+', required=True,
                        help='List of input parquet files to merge')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output parquet file path')
    args = parser.parse_args()
    
    merge_datasets(args.input_files, args.output_file)