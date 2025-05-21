from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn.functional as F

import os
import sys
import json
from tqdm import tqdm
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from route.gpt.prompt import ROUTER_PROMPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

retrieval_methods = ["no", "paragraph", "document", "image", "clip", "video", "error"]

def route(questions, max_new_tokens=1):
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True)
    logits = outputs.scores[0]
    probabilities = F.softmax(logits, dim=-1)
    answers = torch.argmax(probabilities, dim=-1).tolist()
    decoded_answers = tokenizer.batch_decode(answers, skip_special_tokens=True)
    
    retrievals = []
    probabilities_list = []
    for idx, answer in enumerate(decoded_answers):
        answer = answer.lower()
        if answer not in retrieval_methods:
            retrieval = "error"
            probability = 0.0
        else:
            retrieval = answer
            probability = probabilities[idx][answers[idx]].item()
        retrievals.append(retrieval)
        probabilities_list.append(probability)
    
    return retrievals, probabilities_list

def main(input_path, output_path, batch_size=128):
    overall_results = []

    for path in input_path:
        with open(path, 'r') as file:
            data = json.load(file)

        count = count = {rm: 0 for rm in retrieval_methods}
        correct = 0
        retrieval_confs = 0
        
        questions = [item["question"].rsplit("\n")[0] for item in data]
        questions = [ROUTER_PROMPT.format(query=question) for question in questions]
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Routing {os.path.basename(path)} with {model.config._name_or_path}"):
            batch_questions = questions[i:i+batch_size]
            batch_retrievals, batch_probabilities = route(batch_questions)
            for j, (retrieval, probability) in enumerate(zip(batch_retrievals, batch_probabilities)):
                data[i + j]["retrieval"] = retrieval
                data[i + j]["retrieval_conf"] = probability
                if retrieval == data[i + j]["gt_retrieval"].lower(): correct += 1
                count[retrieval] += 1
            retrieval_confs += sum(batch_probabilities)

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
    parser.add_argument("--checkpoint_dir", type=str, default="route/train/checkpoints/t5-large", help="Directory to load checkpoints")
    parser.add_argument("--input_dir", type=str, default="dataset/query", help="Directory containing the input data")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, default="route/results", help="Directory to save results")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(args.checkpoint_dir)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir, device_map="auto", torch_dtype=torch.bfloat16).to(device)

    input_path = [os.path.join(args.input_dir, fname) for fname in os.listdir(args.input_dir) if fname.endswith('.json')]
    model_size = os.path.basename(checkpoint_dir)
    output_path = os.path.join(args.output_dir, model_size)
    os.makedirs(output_path, exist_ok=True)

    main(input_path, output_path, args.batch_size)
