from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

import os
import json
from tqdm import tqdm
from tabulate import tabulate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

retrieval_methods = ["no", "paragraph", "document", "image", "clip", "video"]

def route(questions, model, tokenizer, max_input_length=512):
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    indices = torch.argmax(probabilities, dim=-1).tolist()
    confidences = probabilities.max(dim=-1).values.tolist()
    translated_outputs = [retrieval_methods[index] for index in indices]
    return translated_outputs, confidences

def main(data_paths, output_path, batch_size=256, max_input_length=512):
    overall_results = []

    for path in data_paths:
        with open(path, 'r') as file:
            data = json.load(file)

        count = count = {rm: 0 for rm in retrieval_methods}
        correct = 0
        retrieval_confs = 0
        
        questions = [item["question"].rsplit("\n")[0] for item in data]
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Routing {os.path.basename(path)} with {model.config._name_or_path}"):
            batch_questions = questions[i:i+batch_size]
            batch_outputs, batch_confidences = route(batch_questions, model, tokenizer, max_input_length)
            for j, (output, confidence) in enumerate(zip(batch_outputs, batch_confidences)):
                data[i + j]["retrieval"] = output
                data[i + j]["retrieval_conf"] = confidence
                if output == data[i + j]["gt_retrieval"].lower(): correct += 1
                count[output] += 1
            retrieval_confs += sum(batch_confidences)

        count["accuracy"] = round(correct / len(data), 4)
        count["avg_conf"] = round(retrieval_confs / len(data), 4)
        
        result_row = {"Path": os.path.basename(path)}
        result_row.update(count)
        overall_results.append(result_row)

        with open(os.path.join(output_path, os.path.basename(path)), 'w') as outfile:
            json.dump(data, outfile, indent=4)

    print(tabulate(overall_results, headers="keys", tablefmt="fancy_grid"))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="router/train/checkpoints/distilbert", help="Directory to load checkpoints")
    parser.add_argument("--input_dir", type=str, default="eval/data", help="Directory containing the input data")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, default="router/results", help="Directory to save results")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(args.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)

    input_path = [os.path.join(args.input_dir, fname) for fname in os.listdir(args.input_dir) if fname.endswith('.json')]
    model_size = os.path.basename(checkpoint_dir)
    output_path = os.path.join(args.output_dir, model_size)
    os.makedirs(output_path, exist_ok=True)

    main(input_path, output_path, args.batch_size)
