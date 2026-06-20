import os
import json
import pandas as pd
from tqdm import tqdm

def extract_text(query_file, request_tok_path, output_file):
    with open(query_file, 'r') as f:
        hybridqa_data = json.load(f)

    print(f"Processing {len(hybridqa_data)} entries from {query_file}...")

    rows = []
    seen_ids = set()
    
    for entry in tqdm(hybridqa_data, desc="Extracting text"):
        gt_table = entry['gt_tables']
        
        request_file_path = os.path.join(request_tok_path, f"{gt_table}.json")
        
        if os.path.exists(request_file_path):
            try:
                with open(request_file_path, 'r') as f:
                    passages_data = json.load(f)
                
                for idx, (wiki_path, passage_text) in enumerate(passages_data.items()):
                    psg_id = f"hybridqa-{gt_table}_{idx}"
                    if psg_id not in seen_ids:
                        seen_ids.add(psg_id)
                        rows.append({
                            'psg_id': psg_id,
                            'text': passage_text
                        })
            
            except Exception as e:
                print(f"Error processing {request_file_path}: {e}")
        else:
            print(f"Warning: File not found: {request_file_path}")
        
    df = pd.DataFrame(rows)
    df.set_index('psg_id', inplace=True)
    df.to_parquet(output_file)
    print(f"Saved {len(df)} passages to {output_file}")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract texts from HybridQA request_tok files.")
    parser.add_argument('--query', type=str, default='../query/hybridqa.json', help='Query JSON file')
    parser.add_argument('--request-tok', type=str, default='WikiTables-WithLinks/request_tok', help='Input directory for request_tok files')
    parser.add_argument('--output', type=str, default='hybridqa_text.parquet', help='Output parquet file for extracted texts')
    args = parser.parse_args()

    query_file = args.query
    request_tok_path = args.request_tok
    output_file = args.output

    extract_text(query_file, request_tok_path, output_file)