import os
import pickle
import json
from tqdm import tqdm

from universalrag.embedding import EmbeddingModel

embedding_model = EmbeddingModel()

def extract_query_feats_vlm2vec(input, output_path, batch_size=64, alpha=0.2):
    """
    Extract query features from the input JSON file and save them as a pickle file.
    If query_image exists, encode it and ensemble with text features.
    Args:
        input (str): Path to the input JSON file.
        output_path (str): Path to save the pickle file.
        batch_size (int): Batch size for encoding. (default: 64)
        alpha (float): Weight for image features in ensemble. (default: 0.2)
    """
    with open(input, 'r') as f:
        data = json.load(f)
    
    query_ids = []
    text_data_list = []
    image_data_list = []
    has_images = []
    
    for row in data:
        query_ids.append(row['index'])
        text_data_list.append(row['question'])
        
        if 'query_image' in row and row['query_image']:
            has_images.append(True)
            image_data_list.append(row['query_image'])
        else:
            has_images.append(False)
            image_data_list.append(None)
    
    # Encode text queries
    text_features = embedding_model.encode_vis_query(text_data_list, batch_size=batch_size)
    
    # Encode images if they exist
    if any(has_images):
        images_to_encode = []
        image_indices = []
        
        for idx, (has_image, image_path) in enumerate(zip(has_images, image_data_list)):
            if has_image:
                images_to_encode.append(image_path)
                image_indices.append(idx)
        
        if images_to_encode:
            image_features = embedding_model.encode_image(images_to_encode, batch_size=32)
            
            # Ensemble text and image features
            for feat_idx, data_idx in enumerate(image_indices):
                text_features[data_idx] = (1 - alpha) * text_features[data_idx] + alpha * image_features[feat_idx]
    
    id2feat = {}
    for query_id, feat in zip(query_ids, text_features):
        id2feat[query_id] = feat

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, os.path.splitext(os.path.basename(input))[0] + '.pkl'), 'wb') as f:
        pickle.dump(id2feat, f)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Extract query features from VLM2Vec and save them as a pickle file.")
    parser.add_argument("--input-path", type=str, default="dataset/query", help="Path to the input directory containing JSON files.")
    parser.add_argument("--output-path", type=str, default="eval/features/query/vlm2vec", help="Path to save the output pickle files.")
    args = parser.parse_args()

    inputs = [os.path.join(args.input_path, input) for input in os.listdir(args.input_path)]

    for input in tqdm(inputs):
        extract_query_feats_vlm2vec(input, args.output_path)
