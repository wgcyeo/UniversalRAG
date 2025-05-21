import os
import logging
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

output_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extract_image.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(output_log_file, mode='a'),
        # logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_row(args):
    index, base64_image, output_dir = args
    try:
        image = Image.open(BytesIO(base64.b64decode(base64_image))).convert('RGB')
        out_path = os.path.join(output_dir, f'{index}.jpg')
        image.save(out_path)
        logger.info(f"Extracted image: {out_path}")
    except Exception as e:
        logger.error(f"Error processing row {index}: {e}")

def extract_image(input_file, output_dir):
    df = pd.read_csv(input_file, sep='\t', header=None)
    args_iter = ((row[0], row[1], output_dir) for _, row in df.iterrows())
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_row, args_iter), total=len(df), desc="Extracting images"))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract images from a TSV file containing base64-encoded images.")
    parser.add_argument('--input', type=str, default='imgs.tsv', help='Input TSV file')
    parser.add_argument('--output', type=str, default='images', help='Output directory for extracted images')
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    extract_image(input_file, output_dir)
