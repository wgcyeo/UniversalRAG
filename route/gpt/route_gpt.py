import os
import json
from tqdm import tqdm
from tabulate import tabulate
from openai import OpenAI

from prompt import ROUTER_PROMPT

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

retrieval_methods = ["no", "paragraph", "document", "image", "clip", "video", "error"]

def route_with_gpt(target, output_path):
    with open(target, 'r') as f:
        dataset = json.load(f)

    results = []
    count = {rm: 0 for rm in retrieval_methods}
    correct = 0

    for item in tqdm(dataset, desc=f"Routing {os.path.basename(target)} with GPT-4o"):
        prompt_text = ROUTER_PROMPT.format(query=item["question"])
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
                    max_tokens=1
                )
                retrieval = response.choices[0].message.content.strip().lower()
                break
            except Exception as e:
                retrieval = "error"

        retrieval = retrieval if retrieval in retrieval_methods else "error"
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
    parser.add_argument("--input_dir", type=str, default="dataset/query", help="Directory containing the input data")
    parser.add_argument("--output_dir", type=str, default="route/results/gpt", help="Directory to save results")
    args = parser.parse_args()

    if os.path.isdir(args.input_dir):
        targets = [os.path.join(args.input_dir, fname) for fname in os.listdir(args.input_dir) if fname.endswith('.json')]
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        overall_results = []
        for target in targets:
            output_path = os.path.join(output_dir, os.path.basename(target))
            result_row = route_with_gpt(target, output_path)
            overall_results.append(result_row)
        print(tabulate(overall_results, headers="keys", tablefmt="fancy_grid"))
    elif os.path.isfile(args.input_dir) and args.input_dir.endswith('.json'):
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(args.input_dir))
        result_row = route_with_gpt(args.input_dir, output_path)
        print(tabulate([result_row], headers="keys", tablefmt="fancy_grid"))
    else:
        raise ValueError("Invalid target. Please provide a valid JSON file or directory containing JSON files.")
