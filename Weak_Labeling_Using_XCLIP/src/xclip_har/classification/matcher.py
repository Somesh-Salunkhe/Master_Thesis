import torch
import numpy as np
import json
from transformers import AutoTokenizer, XCLIPModel
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

class XCLIPMatcher:
    def __init__(
        self, 
        model_name: str = "microsoft/xclip-base-patch16-16-frames", 
        device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Loading X-CLIP model '{model_name}' on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = XCLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed_text_prompts(
        self, 
        prompts_dict: Dict[str, List[str]]
    ) -> Tuple[List[str], np.ndarray]:
        """
        Converts activity prompts into normalized text embeddings.
        """
        all_names = []
        all_embs = []

        for activity, prompts in prompts_dict.items():
            if not prompts: continue
            
            inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            text_feat = self.model.get_text_features(**inputs)
            
            # Handle different output types
            if hasattr(text_feat, "pooler_output"):
                text_feat = text_feat.pooler_output
            
            text_feat = text_feat.detach().float().cpu().numpy()
            
            # Normalize and average
            norm = np.linalg.norm(text_feat, axis=1, keepdims=True) + 1e-12
            text_feat = text_feat / norm
            
            activity_emb = text_feat.mean(axis=0, keepdims=True)
            activity_emb = activity_emb / (np.linalg.norm(activity_emb) + 1e-12)
            
            all_names.append(activity)
            all_embs.append(activity_emb[0])

        return all_names, np.stack(all_embs, axis=0)

    def match_clusters(
        self, 
        centroids: np.ndarray, 
        text_embs: np.ndarray, 
        activity_names: List[str],
        exclude_rest: bool = True
    ) -> Tuple[Dict[int, str], np.ndarray]:
        """
        Matches cluster centroids to the nearest text activity using cosine similarity.
        """
        # (K x A)
        sim_matrix = cosine_similarity(centroids, text_embs)
        
        sim_filtered = sim_matrix.copy()
        if exclude_rest and "Rest" in activity_names:
            rest_idx = activity_names.index("Rest")
            sim_filtered[:, rest_idx] = -1.0
            
        best_idx = sim_filtered.argmax(axis=1)
        mapping = {int(i): activity_names[best_idx[i]] for i in range(len(centroids))}
        
        return mapping, sim_matrix
