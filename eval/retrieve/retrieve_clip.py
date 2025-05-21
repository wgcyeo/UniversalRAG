import pickle
import numpy as np
import torch

class InternClipRetriever:
    def __init__(self, queryfeats_path: str, clipfeats_path, clipscriptfeats_path = None, alpha: float = 0.2):
        self.queryfeats_path = queryfeats_path
        self.clipfeats_path = clipfeats_path
        self.clipscriptfeats_path = clipscriptfeats_path
        self.queryfeats = None
        self.clipids = []
        self.clipfeats = []
        
        assert alpha >= 0 and alpha <= 1; f"alpha should be in [0, 1], but got {alpha}"
        self.load_feats(queryfeats_path, clipfeats_path, clipscriptfeats_path, alpha)

    def load_feats(self, queryfeats_path: str, clipfeats_path, clipscriptfeats_path = None, alpha: float = 0.2):
        print(f"Loading InternClipRetriever from {clipfeats_path}...")
        with open(queryfeats_path, 'rb') as f:
            self.queryfeats = pickle.load(f)
        
        clipfeats = {}
        if isinstance(clipfeats_path, list):
            for path in clipfeats_path:
                with open(path, 'rb') as f:
                    clipfeats.update(pickle.load(f))
        else:
            with open(clipfeats_path, 'rb') as f:
                clipfeats = pickle.load(f)
        
        clipscriptfeats = {}
        if clipscriptfeats_path:
            if isinstance(clipscriptfeats_path, list):
                for path in clipscriptfeats_path:
                    with open(path, 'rb') as f:
                        clipscriptfeats.update(pickle.load(f))
            else:
                with open(clipscriptfeats_path, 'rb') as f:
                    clipscriptfeats = pickle.load(f)

        for clip_id, clip_feat in clipfeats.items():
            original_clip_id = clip_id.rsplit('_', 1)[0]
            if clipscriptfeats_path and original_clip_id in clipscriptfeats:
                script_feat = clipscriptfeats[original_clip_id]
                self.clipfeats.append(alpha * script_feat + (1-alpha) * clip_feat)
            elif clipscriptfeats_path:
                script_feat = clipscriptfeats[clip_id]
                self.clipfeats.append(alpha * script_feat + (1-alpha) * clip_feat)
            else:
                self.clipfeats.append(clip_feat)
            self.clipids.append(clip_id)

        self.clipfeats = torch.tensor(np.stack(self.clipfeats)).to('cuda')

    def retrieve(self, query_id, top_k: int = 5):
        query_feat = self.queryfeats[query_id]
        similarity = torch.matmul(query_feat, self.clipfeats.T).unsqueeze(0)
        _, rankings = torch.sort(similarity, dim=1, descending=True)
        top_k_clip_ids = [self.clipids[rank] for rank in rankings[0][:top_k]]
        top_k_scores = similarity[0][rankings[0][:top_k]].cpu().numpy()
        return top_k_clip_ids, top_k_scores
    
    def score_recall(self, text_query_ids, gt_ranking, k_values=[1, 5, 10]):
        results = {f"recall@{k}": 0.0 for k in k_values}
        total_queries = len(text_query_ids)
        for query_id in text_query_ids:
            correct_clipids = gt_ranking[query_id]
            retrieved_clipids, _ = self.retrieve(query_id, max(k_values))
            retrieved_clipids = [vid.rsplit('_', 1)[0] for vid in retrieved_clipids]
            for k in k_values:
                if any(vid in correct_clipids for vid in retrieved_clipids[:k]):
                    results[f"recall@{k}"] += 1
        for k in k_values:
            results[f"recall@{k}"] /= total_queries
        return results


if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="lvbench", choices=["lvbench", "videorag_wikihow", "videorag_synth"])
    parser.add_argument("--alpha", type=float, default=0.2, help="Weight for video script features (0 to 1)")
    args = parser.parse_args()

    queryfeats_path = f'eval/features/query/internvideo/{args.target}.pkl'
    clipfeats_path=[
        'eval/features/clip/howto100m.pkl',
        'eval/features/clip/lvbench.pkl'
    ]
    clipscriptfeats_path=[
        'eval/features/video/howto100m_vidscript.pkl',
        'eval/features/clip/lvbench_clipscript.pkl'
    ]

    retriever = InternClipRetriever(
        queryfeats_path=queryfeats_path,
        clipfeats_path=clipfeats_path,
        clipscriptfeats_path=clipscriptfeats_path,
        alpha=args.alpha,
    )

    gt_ranking_path = f"dataset/query/{args.target}.json"
    with open(gt_ranking_path, 'r') as f:
        gt_ranking_data = json.load(f)
    gt_ranking = {qa['index']: qa['gt_videos'] for qa in gt_ranking_data}

    print(retriever.score_recall(gt_ranking.keys(), gt_ranking))
