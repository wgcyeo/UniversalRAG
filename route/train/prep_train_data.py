import json
import random
from pathlib import Path

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get the base directory (should work when executed from base folder)
    base_dir = Path.cwd()
    query_dir = base_dir / "dataset" / "query"
    
    # Define the query files to sample from
    query_files = [
        "mmlu.json",
        "natural_questions.json",
        "hotpotqa.json",
        "hybridqa.json",
        "webqa.json",
        "mrag_bench.json",
        "infoseek.json",
        "lvbench.json",
        "videorag_synth.json",
        "videorag_wikihow.json"
    ]
    
    # Sample rate
    sample_rate = 0.3
    
    # Store all sampled data
    all_sampled = []
    
    # Process each file
    for filename in query_files:
        filepath = query_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        print(f"Processing {filename}...")
        
        # Load the data
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Sample 30% of the data
        sample_size = int(len(data) * sample_rate)
        sampled = random.sample(data, sample_size)
        
        print(f"  Original size: {len(data)}, Sampled: {sample_size}")
        
        # Add to combined list
        all_sampled.extend(sampled)
    
    # Shuffle the combined data
    random.shuffle(all_sampled)
    
    # Save to output file in route/train/data directory
    output_dir = base_dir / "route" / "train" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train_data.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_sampled, f, indent=4)
    
    print(f"\nTotal sampled queries: {len(all_sampled)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
