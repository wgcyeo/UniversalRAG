import os
import pickle
import json
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('BAAI/bge-large-en-v1.5')
instruction = "Represent this sentence for searching relevant passages: "

def extract_query_feats_bge(input, output_path, batch_size=512):
    """
    Extract query features from the input JSON file and save them as a pickle file.
    Args:
        input (str): Path to the input JSON file.
        output_path (str): Path to save the pickle file.
    """
    with open(input, 'r') as f:
        data = json.load(f)

    id2feat = {}
    query_ids = []
    texts = []
    for row in data:
        query_ids.append(row['index'])
        texts.append(instruction + row['question'])

    encoded_features = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    for query_id, feature in zip(query_ids, encoded_features):
        id2feat[query_id] = torch.tensor(feature).cuda()

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, os.path.splitext(os.path.basename(input))[0] + '.pkl'), 'wb') as f:
        pickle.dump(id2feat, f)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Extract query features using BGE and save them as a pickle file.")
    parser.add_argument("--input_path", type=str, default="dataset/query", help="Path to the input directory containing JSON files.")
    parser.add_argument("--output_path", type=str, default="eval/features/query/bge-large", help="Path to save the output pickle files.")
    args = parser.parse_args()

    inputs = [os.path.join(args.input_path, input) for input in os.listdir(args.input_path)]

    for input in tqdm(inputs):
        extract_query_feats_bge(input, args.output_path)
