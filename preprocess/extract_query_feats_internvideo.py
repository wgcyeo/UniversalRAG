import os
import sys
import pickle
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

if not os.getenv('INTERNVIDEO_PATH'):
    raise EnvironmentError("Environment variable `INTERNVIDEO_PATH` is not set.")
internvideo2_path = os.path.join(os.getenv('INTERNVIDEO_PATH'), 'InternVideo2/multi_modality')
sys.path.append(internvideo2_path)

from utils.config import Config, eval_dict_leaf
from demo.utils import setup_internvideo2

device='cuda'

config = Config.from_file(os.path.join(internvideo2_path, 'demo/internvideo2_stage2_config.py'))
config = eval_dict_leaf(config)
config.model.vision_encoder.pretrained = os.path.join(internvideo2_path, config.model.vision_encoder.pretrained)
config.model.text_encoder.config = os.path.join(internvideo2_path, config.model.text_encoder.config)
config.pretrained_path = os.path.join(internvideo2_path, config.pretrained_path)
intern_model, _ = setup_internvideo2(config)
intern_model.to(device)

def extract_query_feats_internvideo(input, output_path):
    """
    Extract query features from the input JSON file and save them as a pickle file.
    Args:
        input (str): Path to the input JSON file.
        output_path (str): Path to save the pickle file.
    """
    with open(input, 'r') as f:
        data = json.load(f)

    id2feat = {}
    for row in data:
        query_id = row['index']
        text_data = row['question']
        id2feat[query_id] = intern_model.get_txt_feat(text_data).squeeze(0)

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, os.path.splitext(os.path.basename(input))[0] + '.pkl'), 'wb') as f:
        pickle.dump(id2feat, f)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Extract query features from InternVideo and save them as a pickle file.")
    parser.add_argument("--input_path", type=str, default="dataset/query", help="Path to the input directory containing JSON files.")
    parser.add_argument("--output_path", type=str, default="eval/features/query/internvideo", help="Path to save the output pickle files.")
    args = parser.parse_args()

    inputs = [os.path.join(args.input_path, input) for input in os.listdir(args.input_path)]

    for input in tqdm(inputs):
        extract_query_feats_internvideo(input, args.output_path)
