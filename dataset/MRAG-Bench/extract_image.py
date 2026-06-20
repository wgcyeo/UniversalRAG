import pandas as pd
import os
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

output_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extract_image.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(output_log_file, mode='a'),
    ]
)

logger = logging.getLogger(__name__)

def process_row(args):
    """Process a single row to extract image data"""
    row_id, image_data, image_dir, image_type, image_idx = args
    try:
        if image_data is not None:
            image_bytes = image_data['bytes']
            if image_idx is not None:
                output_path = os.path.join(image_dir, f'{row_id}_{image_idx}.jpg')
            else:
                output_path = os.path.join(image_dir, f'{row_id}.jpg')
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            logger.info(f"Extracted {image_type} image: {output_path}")
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Error processing {image_type} for row {row_id}: {e}")

def extract_images(input_path, output_path):
    """Extract all image types from parquet files"""
    parquet_files = list(Path(input_path).glob("*.parquet"))

    all_args = []
    
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        
        for idx, row in df.iterrows():
            row_id = row.get('id', idx)
            
            if 'image' in df.columns:
                all_args.append((row_id, row.get('image'), os.path.join(output_path, "query"), 'query_image', None))
            
            if 'gt_images' in df.columns and row['gt_images'] is not None:
                for img_idx, gt_image_data in enumerate(row['gt_images']):
                    all_args.append((row_id, gt_image_data, output_path, 'gt_image', img_idx))
    
    if all_args:
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(process_row, all_args), total=len(all_args), desc="Extracting images"))
    
    logger.info(f"Image extraction completed. Processed {len(all_args)} images.")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract images from parquet files.")
    parser.add_argument('--input', type=str, default='data', help='Directory containing parquet files')
    parser.add_argument('--output', type=str, default='images', help='Base output directory for extracted images')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "query"), exist_ok=True)

    extract_images(args.input, args.output)
