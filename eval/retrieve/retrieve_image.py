import pickle
import numpy as np
import torch

class InternImgRetriever:
    def __init__(self, queryfeats_path: str, imgfeats_path, imgcapfeats_path = None, alpha: float = 0.2):
        self.queryfeats_path = queryfeats_path
        self.imgfeats_path = imgfeats_path
        self.imgcapfeats_path = imgcapfeats_path if alpha != 0 else None
        self.queryfeats = None
        self.imgids = []
        self.imgfeats = []

        assert alpha >= 0 and alpha <= 1; f"alpha should be in [0, 1], but got {alpha}"
        self.load_feats(queryfeats_path, imgfeats_path, self.imgcapfeats_path, alpha)

    def load_feats(self, queryfeats_path: str, imgfeats_path, imgcapfeats_path = None, alpha: float = 0.2):
        print(f"Loading InternImgRetriever from {imgfeats_path}...")
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
        
        imgcapfeats = {}
        if imgcapfeats_path:
            if isinstance(imgcapfeats_path, list):
                for path in imgcapfeats_path:
                    with open(path, 'rb') as f:
                        imgcapfeats.update(pickle.load(f))
            else:
                with open(imgcapfeats_path, 'rb') as f:
                    imgcapfeats = pickle.load(f)

        for img_id, img_feat in imgfeats.items():
            if imgcapfeats_path:
                imgcap_feat = imgcapfeats[img_id]
                self.imgfeats.append(alpha * imgcap_feat + (1-alpha) * img_feat)
            else:
                self.imgfeats.append(img_feat)
            self.imgids.append(img_id)

        self.imgfeats = torch.tensor(np.stack(self.imgfeats)).to('cuda')

    def retrieve(self, query_id, top_k: int = 5):
        query_feat = self.queryfeats[query_id]
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
    parser.add_argument("--alpha", type=float, default=0.2, help="Weight for image caption features (0 to 1)")
    args = parser.parse_args()

    queryfeats_path = f"eval/features/query/internvideo/{args.target}.pkl"
    imgfeats_path = [
        "eval/features/image/webqa.pkl"
    ]
    imgcapfeats_path = [
        "eval/features/image/webqa_imgcap.pkl"
    ]

    retriever = InternImgRetriever(
        queryfeats_path=queryfeats_path,
        imgfeats_path=imgfeats_path,
        imgcapfeats_path=imgcapfeats_path,
        alpha=args.alpha,
    )

    gt_ranking_path = f"dataset/query/{args.target}.json"
    with open(gt_ranking_path, 'r') as f:
        gt_ranking_data = json.load(f)
    gt_ranking = {qa['index']: qa['gt_images'] for qa in gt_ranking_data}

    print(retriever.score_recall(gt_ranking.keys(), gt_ranking))
