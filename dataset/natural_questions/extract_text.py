import json
import pandas as pd
from tqdm import tqdm

def extract_text(input_file, output_file):
    with open(input_file) as json_file:
        original_data = json.load(json_file)

    all_contexts = []
    for data in tqdm(original_data, desc="Collecting contexts"):
        all_contexts.extend(data['positive_ctxs'] + data['negative_ctxs'] + data['hard_negative_ctxs'])

    # Deduplicate by psg_id and create dataframe
    seen_ids = set()
    rows = []
    for ctx in tqdm(all_contexts, desc="Processing contexts"):
        psg_id = ctx['passage_id']
        if psg_id not in seen_ids:
            seen_ids.add(psg_id)
            rows.append({
                'psg_id': 'nq-' + str(psg_id),
                'text': ctx['text']
            })

    df = pd.DataFrame(rows)
    df.set_index('psg_id', inplace=True)
    df.to_parquet(output_file)
    print(f"Saved {len(df)} passages to {output_file}")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract texts from biencoder NQ JSON file.")
    parser.add_argument('--input', type=str, default='biencoder-nq-dev.json', help='Input JSON file')
    parser.add_argument('--output', type=str, default='nq.parquet', help='Output parquet file for extracted texts')
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    extract_text(input_file, output_file)
