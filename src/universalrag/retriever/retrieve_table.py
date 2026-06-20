import pickle
import numpy as np
import torch

class TableRetriever:
    def __init__(self, queryfeats_path: str, tablefeats_path: str | list[str]):
        self.queryfeats_path = queryfeats_path
        self.tablefeats_path = tablefeats_path
        self.queryfeats = None
        self.tableids = []
        self.tablefeats = []

        self.load_feats(queryfeats_path, tablefeats_path)

    def load_feats(self, queryfeats_path, tablefeats_path):
        print(f"Loading TableRetriever from {tablefeats_path}...")
        with open(queryfeats_path, 'rb') as f:
            self.queryfeats = pickle.load(f)
        
        if isinstance(tablefeats_path, list):
            tablefeats = {}
            for path in tablefeats_path:
                with open(path, 'rb') as f:
                    tablefeats.update(pickle.load(f))
        else:
            with open(tablefeats_path, 'rb') as f:
                tablefeats = pickle.load(f)

        for table_id, table_feat in tablefeats.items():
            self.tablefeats.append(table_feat)
            self.tableids.append(table_id)

        self.tablefeats = torch.tensor(np.stack(self.tablefeats)).to('cuda')

    def retrieve(self, query_id, top_k: int = 5):
        query_feat = torch.tensor(self.queryfeats[query_id]).to('cuda')
        similarity = torch.matmul(query_feat, self.tablefeats.T).unsqueeze(0)
        _, rankings = torch.sort(similarity, dim=1, descending=True)
        top_k_table_ids = [self.tableids[rank].rsplit('_row', 1)[0] for rank in rankings[0][:top_k]]
        top_k_scores = similarity[0][rankings[0][:top_k]].cpu().numpy()
        return top_k_table_ids, top_k_scores
    
    def score_recall(self, table_query_ids, gt_ranking, k_values=[1, 5, 10]):
        results = {f"recall@{k}": 0.0 for k in k_values}
        total_queries = len(table_query_ids)
        for query_id in table_query_ids:
            correct_tableids = gt_ranking[query_id]
            retrieved_tableids, _ = self.retrieve(query_id, max(k_values))
            for k in k_values:
                if any(table in correct_tableids for table in retrieved_tableids[:k]):
                    results[f"recall@{k}"] += 1
        for k in k_values:
            results[f"recall@{k}"] /= total_queries
        return results


if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="hybridqa", choices=["hybridqa"])
    args = parser.parse_args()

    queryfeats_path = f"eval/features/query/qwen3/{args.target}.pkl"
    tablefeats_path = ["eval/features/table.pkl"]

    retriever = TableRetriever(
        queryfeats_path=queryfeats_path,
        tablefeats_path=tablefeats_path
    )

    gt_ranking_path = f"dataset/query/{args.target}.json"
    with open(gt_ranking_path, 'r') as f:
        gt_ranking_data = json.load(f)
    gt_ranking = {qa['index']: qa['gt_tables'] for qa in gt_ranking_data}

    print(retriever.score_recall(gt_ranking.keys(), gt_ranking))
