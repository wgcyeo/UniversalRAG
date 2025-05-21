import pickle
import numpy as np
import torch

class InternVidRetriever:
    def __init__(self, queryfeats_path: str, vidfeats_path: str | list[str], vidscriptfeats_path = None, alpha: float = 0.2):
        self.queryfeats_path = queryfeats_path
        self.vidfeats_path = vidfeats_path
        self.vidscriptfeats_path = vidscriptfeats_path if alpha != 0 else None
        self.queryfeats = None
        self.videoids = []
        self.videofeats = []
        
        assert alpha >= 0 and alpha <= 1; f"alpha should be in [0, 1], but got {alpha}"
        self.load_feats(queryfeats_path, vidfeats_path, vidscriptfeats_path, alpha)

    def load_feats(self, queryfeats_path: str, vidfeats_path, vidscriptfeats_path = None, alpha: float = 0.2):
        print(f"Loading InternVidRetriever from {vidfeats_path}...")
        with open(queryfeats_path, 'rb') as f:
            self.queryfeats = pickle.load(f)
        
        vidfeats = {}
        if isinstance(vidfeats_path, list):
            for path in vidfeats_path:
                with open(path, 'rb') as f:
                    vidfeats.update(pickle.load(f))
        else:
            with open(vidfeats_path, 'rb') as f:
                vidfeats = pickle.load(f)
        
        vidscriptfeats = {}
        if vidscriptfeats_path:
            if isinstance(vidscriptfeats_path, list):
                for path in vidscriptfeats_path:
                    with open(path, 'rb') as f:
                        vidscriptfeats.update(pickle.load(f))
            else:
                with open(vidscriptfeats_path, 'rb') as f:
                    vidscriptfeats = pickle.load(f)

        for vid_id, vid_feat in vidfeats.items():
            if vidscriptfeats_path and vid_id in vidscriptfeats:
                vidscript_feat = vidscriptfeats[vid_id]
                self.videofeats.append(alpha * vidscript_feat + (1-alpha) * vid_feat)
            else:
                self.videofeats.append(vid_feat)
            self.videoids.append(vid_id)
        
        self.videofeats = torch.tensor(np.stack(self.videofeats)).to('cuda')

    def retrieve(self, query_id, top_k: int = 5):
        query_feat = self.queryfeats[query_id]
        similarity = torch.matmul(query_feat, self.videofeats.T).unsqueeze(0)
        _, rankings = torch.sort(similarity, dim=1, descending=True)
        top_k_vid_ids = [self.videoids[rank] for rank in rankings[0][:top_k]]
        top_k_scores = similarity[0][rankings[0][:top_k]].cpu().numpy()
        return top_k_vid_ids, top_k_scores
    
    def score_recall(self, text_query_ids, gt_ranking, k_values=[1, 5, 10]):
        results = {f"recall@{k}": 0.0 for k in k_values}
        total_queries = len(text_query_ids)
        for query_id in text_query_ids:
            correct_vids = gt_ranking[query_id]
            retrieved_vids, _ = self.retrieve(query_id, max(k_values))
            for k in k_values:
                if any(vid in correct_vids for vid in retrieved_vids[:k]):
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
    vidfeats_path=[
        'eval/features/video/howto100m.pkl',
        'eval/features/video/lvbench.pkl',
    ]
    vidscriptfeats_path=[
        'eval/features/video/howto100m_vidscript.pkl',
        'eval/features/video/lvbench_vidscript.pkl',
    ]

    retriever = InternVidRetriever(
        queryfeats_path=queryfeats_path,
        vidfeats_path=vidfeats_path,
        vidscriptfeats_path=vidscriptfeats_path,
        alpha=args.alpha,
    )

    gt_ranking_path = f"dataset/query/{args.target}.json"
    with open(gt_ranking_path, 'r') as f:
        gt_ranking_data = json.load(f)
    gt_ranking = {qa['index']: qa['gt_videos'] for qa in gt_ranking_data}

    print(retriever.score_recall(gt_ranking.keys(), gt_ranking))
