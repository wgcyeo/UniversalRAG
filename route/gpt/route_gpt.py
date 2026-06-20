import os
import sys
import json
import re
from tqdm import tqdm
from tabulate import tabulate

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from universalrag.llm import LLMModel
from route.prompt import ROUTER_PROMPT

ALLOWED_CATEGORIES = ["no", "paragraph", "document", "table", "image", "clip", "video"]
retrieval_methods = ALLOWED_CATEGORIES + ["error"]

def _clean_category_token(token: str) -> str:
    token = token.strip().lower()
    return re.sub(r"^[^a-z]+|[^a-z]+$", "", token)

def normalize_retrieval(response: str) -> str:
    if not response:
        return "error"
    text = response.strip().lower()
    if not text:
        return "error"

    parts = text.split("+")
    tokens = []
    for part in parts:
        cleaned = _clean_category_token(part)
        if not cleaned or cleaned not in ALLOWED_CATEGORIES:
            return "error"
        tokens.append(cleaned)

    if len(set(tokens)) != len(tokens):
        return "error"

    ordered = [cat for cat in ALLOWED_CATEGORIES if cat in tokens]
    if len(ordered) != len(tokens):
        return "error"
    return "+".join(ordered)

def route_with_gpt(target, output_path, model_name: str):
    with open(target, 'r') as f:
        dataset = json.load(f)

    model = LLMModel(model_name=model_name)

    results = []
    count = {rm: 0 for rm in retrieval_methods}
    correct = 0

    # Prepare batch prompts
    batch_prompts = []
    for item in dataset:
        prompt_text = ROUTER_PROMPT.format(query=item["question"])
        batch_prompts.append(prompt_text)

    # Process all items in batch using generate_batch
    responses = model.generate_batch(batch_prompts)

    # Process results
    for item, response in tqdm(zip(dataset, responses), total=len(dataset), desc=f"Routing {os.path.basename(target)} with {model.model_name}"):
        retrieval = normalize_retrieval(response)
        if retrieval not in count:
            count[retrieval] = 0
        count[retrieval] += 1

        if retrieval == item["gt_retrieval"].lower():
            correct += 1

        item["retrieval"] = retrieval
        results.append(item)

    count["accuracy"] = round(correct / len(dataset), 4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    result_row = {"Path": os.path.basename(target)}
    result_row.update(count)
    return result_row

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="dataset/query", help="Directory containing the input data")
    parser.add_argument("--output-dir", type=str, default="route/results/gpt-5", help="Directory to save results")
    parser.add_argument("--model-name", type=str, default="gpt-5", help="OpenAI model name")
    args = parser.parse_args()

    if os.path.isdir(args.input_dir):
        targets = [os.path.join(args.input_dir, fname) for fname in os.listdir(args.input_dir) if fname.endswith('.json')]
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        overall_results = []
        for target in targets:
            print(f"Processing {target} with model {args.model_name}...")
            output_path = os.path.join(output_dir, os.path.basename(target))
            result_row = route_with_gpt(target, output_path, args.model_name)
            overall_results.append(result_row)
        print(tabulate(overall_results, headers="keys", tablefmt="fancy_grid"))
    elif os.path.isfile(args.input_dir) and args.input_dir.endswith('.json'):
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(args.input_dir))
        result_row = route_with_gpt(args.input_dir, output_path, args.model_name)
        print(tabulate([result_row], headers="keys", tablefmt="fancy_grid"))
    else:
        raise ValueError("Invalid target. Please provide a valid JSON file or directory containing JSON files.")
