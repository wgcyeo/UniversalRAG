import os
import pickle
import argparse
from tqdm import tqdm

def merge_features(input_files, output_file):
    """
    Merge multiple pickle feature files into a single pickle file.
    
    Args:
        input_files: List of paths to input pickle files
        output_file: Path to output merged pickle file
    """
    merged_features = {}
    
    for file_path in tqdm(input_files, desc="Loading pickle files"):
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    features = pickle.load(f)
                
                # Handle different pickle file structures
                if isinstance(features, dict):
                    # If it's a dictionary, merge keys
                    print(f"Loaded {len(features)} entries from {file_path}")
                    for key, value in features.items():
                        if key in merged_features:
                            print(f"Warning: Duplicate key '{key}' found in {file_path}, skipping...")
                        else:
                            merged_features[key] = value
                elif isinstance(features, list):
                    # If it's a list, convert to dict with indices as keys
                    print(f"Loaded {len(features)} entries (list format) from {file_path}")
                    base_name = os.path.basename(file_path).replace('.pkl', '')
                    for idx, value in enumerate(features):
                        key = f"{base_name}_{idx}"
                        merged_features[key] = value
                else:
                    print(f"Warning: Unsupported format in {file_path}, expected dict or list")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not merged_features:
        raise ValueError("No valid pickle files were loaded or all files were empty.")
    
    # Save merged features
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(merged_features, f)
    
    print(f"Successfully saved {len(merged_features)} entries to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple pickle feature files into one.")
    parser.add_argument('--input-files', type=str, nargs='+', required=True,
                        help='List of input pickle files to merge')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output pickle file path')
    args = parser.parse_args()
    
    merge_features(args.input_files, args.output_file)
