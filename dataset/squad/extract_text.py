import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def write_text_file(args):
    ctx, output_dir = args
    psg_id = ctx['psg_id']
    text = ctx['text']
    out_path = os.path.join(output_dir, f'{psg_id}.txt')
    with open(out_path, 'w') as text_file:
        text_file.write(text)

def extract_text(input_file, output_dir):
    with open(input_file) as json_file:
        original_data = json.load(json_file)

    all_contexts = []
    for data in tqdm(original_data, desc="Collecting contexts"):
        all_contexts.extend(data['positive_ctxs'] + data['negative_ctxs'] + data['hard_negative_ctxs'])

    args_iter = ((ctx, output_dir) for ctx in all_contexts)
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(write_text_file, args_iter), total=len(all_contexts), desc="Extracting texts"))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract texts from biencoder SQuAD JSON file.")
    parser.add_argument('--input', type=str, default='biencoder-squad1-dev.json', help='Input JSON file')
    parser.add_argument('--output', type=str, default='text', help='Output directory for extracted texts')
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    extract_text(input_file, output_dir)
