import os
import pandas as pd
from tqdm import tqdm

def extract_text(input_dir, output_file):
    dataframes = []
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    for file_name in tqdm(parquet_files, desc="Collecting contexts"):
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_parquet(file_path)
        dataframes.append(df)
    if not dataframes:
        raise ValueError("No parquet files found in the folder.")

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Keep only corpus_id and text columns, rename corpus_id to psg_id for consistency
    output_df = combined_df[['corpus_id', 'text']].copy()
    output_df['corpus_id'] = 'hotpotqa-' + output_df['corpus_id'].astype(str)
    output_df.columns = ['psg_id', 'text']
    
    # Deduplicate by psg_id
    output_df = output_df.drop_duplicates(subset=['psg_id'])
    
    output_df.set_index('psg_id', inplace=True)
    output_df.to_parquet(output_file)
    print(f"Saved {len(output_df)} passages to {output_file}")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract texts from HotpotQA parquet files.")
    parser.add_argument('--input', type=str, default='hotpot_qa_corpus', help='Input folder containing parquet files')
    parser.add_argument('--output', type=str, default='hotpotqa.parquet', help='Output parquet file for extracted texts')
    args = parser.parse_args()

    input_dir = args.input
    output_file = args.output

    extract_text(input_dir, output_file)
