import pandas as pd
import pyarrow.json as paj
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def chunk_text(text, chunk_size=600):
    """Split text into chunks of 600 chars, with last chunk allowed 600-1200 chars"""
    chunks = []
    idx = 0
    while idx < len(text):
        chunk_end = idx + chunk_size
        remaining = len(text) - chunk_end
        
        if remaining <= 1200:
            chunks.append(text[idx:])
            break
        else:
            chunks.append(text[idx:chunk_end])
            idx = chunk_end
    return chunks

def process_row(args):
    """Process a single row and return list of (psg_id, text) tuples"""
    wikidata_id, text = args
    chunks = chunk_text(text)
    return [(f'infoseek-{wikidata_id}_{chunk_idx}', chunk) for chunk_idx, chunk in enumerate(chunks)]

def extract_text(input_file, output_file, sample_size=200_000, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()
    
    table = paj.read_json(
        input_file,
        read_options=paj.ReadOptions(block_size=1 << 20)
    )
    
    df = table.to_pandas()
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    data_pairs = list(zip(df['wikidata_id'], df['wikipedia_content']))
    
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_row, data_pairs), total=len(data_pairs), desc="Processing rows"))
    
    output_data = []
    for row_result in results:
        output_data.extend(row_result)
    
    output_df = pd.DataFrame(output_data, columns=['psg_id', 'text'])
    output_df.set_index('psg_id', inplace=True)
    output_df.to_parquet(output_file)
    print(f"Saved {len(output_df)} passages to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract texts from Infoseek JSON file.")
    parser.add_argument('--input', type=str, default='Wiki6M_ver_1_0.jsonl.gz', help='Input JSON file')
    parser.add_argument('--output', type=str, default='infoseek_text.parquet', help='Output parquet file for extracted texts')
    parser.add_argument('--sample-size', type=int, default=200_000, help='Number of samples to extract')
    args = parser.parse_args()
    
    extract_text(args.input, args.output, args.sample_size)