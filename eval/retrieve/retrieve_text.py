import pickle
import numpy as np
import torch

class BGETextRetriever:
    def __init__(self, queryfeats_path: str, textfeats_path: str | list[str]):
        self.queryfeats_path = queryfeats_path
        self.textfeats_path = textfeats_path
        self.queryfeats = None
        self.textids = []
        self.textfeats = []

        self.load_feats(queryfeats_path, textfeats_path)

    def load_feats(self, queryfeats_path, textfeats_path):
        print(f"Loading BGETextRetriever from {textfeats_path}...")
        with open(queryfeats_path, 'rb') as f:
            self.queryfeats = pickle.load(f)
        
        if isinstance(textfeats_path, list):
            textfeats = {}
            for path in textfeats_path:
                with open(path, 'rb') as f:
                    textfeats.update(pickle.load(f))
        else:
            with open(textfeats_path, 'rb') as f:
                textfeats = pickle.load(f)

        for text_id, text_feat in textfeats.items():
            self.textfeats.append(text_feat)
            self.textids.append(text_id)

        self.textfeats = torch.tensor(np.stack(self.textfeats)).to('cuda')

    def retrieve(self, query_id, top_k: int = 5):
        query_feat = self.queryfeats[query_id]
        similarity = torch.matmul(query_feat, self.textfeats.T).unsqueeze(0)
        _, rankings = torch.sort(similarity, dim=1, descending=True)
        top_k_text_ids = [self.textids[rank].rsplit("_part", 1)[0] for rank in rankings[0][:top_k]]
        top_k_scores = similarity[0][rankings[0][:top_k]].cpu().numpy()
        return top_k_text_ids, top_k_scores
    
    def score_recall(self, text_query_ids, gt_ranking, k_values=[1, 5, 10]):
        results = {f"recall@{k}": 0.0 for k in k_values}
        total_queries = len(text_query_ids)
        for query_id in text_query_ids:
            correct_textids = gt_ranking[query_id]
            retrieved_textids, _ = self.retrieve(query_id, max(k_values))
            for k in k_values:
                if any(text in correct_textids for text in retrieved_textids[:k]):
                    results[f"recall@{k}"] += 1
        for k in k_values:
            results[f"recall@{k}"] /= total_queries
        return results


if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="squad", choices=["squad", "natural_questions", "hotpotqa"])
    args = parser.parse_args()

    queryfeats_path = f"eval/features/query/bge-large/{args.target}.pkl"
    textfeats_path = [
        "eval/features/text/squad.pkl",
        "eval/features/text/natural_questions.pkl",
        "eval/features/text/hotpotqa.pkl"
    ]

    retriever = BGETextRetriever(
        queryfeats_path=queryfeats_path,
        textfeats_path=textfeats_path
    )

    gt_ranking_path = f"dataset/query/{args.target}.json"
    with open(gt_ranking_path, 'r') as f:
        gt_ranking_data = json.load(f)
    gt_ranking = {qa['index']: qa['gt_texts'] for qa in gt_ranking_data}

    print(retriever.score_recall(gt_ranking.keys(), gt_ranking))
