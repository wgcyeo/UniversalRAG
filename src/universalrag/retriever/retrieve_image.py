import pickle
import numpy as np
import torch

class ImgRetriever:
    def __init__(self, queryfeats_path: str, imgfeats_path):
        self.queryfeats_path = queryfeats_path
        self.imgfeats_path = imgfeats_path
        self.queryfeats = None
        self.imgids = []
        self.imgfeats = []

        self.load_feats(queryfeats_path, imgfeats_path)

    def load_feats(self, queryfeats_path: str, imgfeats_path):
        print(f"Loading ImgRetriever from {imgfeats_path}...")
        with open(queryfeats_path, 'rb') as f:
            self.queryfeats = pickle.load(f)
        
        imgfeats = {}
        if isinstance(imgfeats_path, list):
            for path in imgfeats_path:
                with open(path, 'rb') as f:
                    imgfeats.update(pickle.load(f))
        else:
            with open(imgfeats_path, 'rb') as f:
                imgfeats = pickle.load(f)

        for img_id, img_feat in imgfeats.items():
            self.imgfeats.append(img_feat)
            self.imgids.append(img_id)

        self.imgfeats = torch.tensor(np.stack(self.imgfeats)).to('cuda')

    def retrieve(self, query_id, top_k: int = 5):
        query_feat = torch.tensor(self.queryfeats[query_id]).to('cuda')
        similarity = torch.matmul(query_feat, self.imgfeats.T).unsqueeze(0)
        _, rankings = torch.sort(similarity, dim=1, descending=True)
        top_k_img_ids = [self.imgids[rank] for rank in rankings[0][:top_k]]
        top_k_scores = similarity[0][rankings[0][:top_k]].cpu().numpy()
        return top_k_img_ids, top_k_scores
    
    def score_recall(self, text_query_ids, gt_ranking, k_values=[1, 5, 10]):
        results = {f"recall@{k}": 0.0 for k in k_values}
        total_queries = len(text_query_ids)
        for query_id in text_query_ids:
            correct_imgids = gt_ranking[query_id]
            retrieved_imgids, _ = self.retrieve(query_id, max(k_values))
            for k in k_values:
                if any(img in correct_imgids for img in retrieved_imgids[:k]):
                    results[f"recall@{k}"] += 1
        for k in k_values:
            results[f"recall@{k}"] /= total_queries
        return results


if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="webqa", choices=["webqa"])
    args = parser.parse_args()

    queryfeats_path = f"eval/features/query/vlm2vec/{args.target}.pkl"
    imgfeats_path = ["eval/features/image.pkl"]

    retriever = ImgRetriever(
        queryfeats_path=queryfeats_path,
        imgfeats_path=imgfeats_path,
    )

    gt_ranking_path = f"dataset/query/{args.target}.json"
    with open(gt_ranking_path, 'r') as f:
        gt_ranking_data = json.load(f)
    gt_ranking = {qa['index']: qa['gt_images'] for qa in gt_ranking_data}

    print(retriever.score_recall(gt_ranking.keys(), gt_ranking))
