import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def write_text_file(args):
    row, output_dir = args
    file_path = os.path.join(output_dir, f"{row['corpus_id']}.txt")
    with open(file_path, 'w') as file:
        file.write(row['text'])

def extract_text(input_dir, output_dir):
    dataframes = []
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    for file_name in tqdm(parquet_files, desc="Collecting contexts"):
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_parquet(file_path)
        dataframes.append(df)
    if not dataframes:
        raise ValueError("No parquet files found in the folder.")

    combined_df = pd.concat(dataframes, ignore_index=True)

    args_iter = ((row, output_dir) for _, row in combined_df.iterrows())
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(write_text_file, args_iter), total=len(combined_df), desc="Extracting texts"))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract texts from HotpotQA parquet files.")
    parser.add_argument('--input', type=str, default='hotpot_qa_corpus', help='Input folder containing parquet files')
    parser.add_argument('--output', type=str, default='text', help='Output directory for extracted texts')
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    extract_text(input_dir, output_dir)
