import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xclip_har.utils.paths import get_project_root, get_output_dir, ensure_dir
from xclip_har.utils.data_io import load_npy, save_npy, save_json
from xclip_har.classification.matcher import XCLIPMatcher

def main():
    parser = argparse.ArgumentParser(description="Match X-CLIP clusters to text activities.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, default=100)
    parser.add_argument('--model_name', type=str, default="microsoft/xclip-base-patch16-16-frames")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    root = get_project_root()
    cluster_dir = get_output_dir() / f"clusters_xclip_{args.n_clusters}" / args.dataset
    out_dir = get_output_dir() / f"classification_xclip_{args.n_clusters}" / args.dataset
    ensure_dir(out_dir)

    # Load prompts
    prompts_path = root / "configs" / "prompts" / f"{args.dataset}_prompts.json"
    if not prompts_path.exists():
        prompts_path = root / "text_embedding_prompts.json" # Legacy fallback

    if not prompts_path.exists():
        print(f"Error: Prompts file not found.")
        return

    with open(prompts_path, "r") as f:
        prompts_dict = json.load(f)

    # Initialize Matcher
    matcher = XCLIPMatcher(model_name=args.model_name, device=f"cuda:{args.gpu}")
    
    print("Embedding text prompts...")
    activity_names, text_embs = matcher.embed_text_prompts(prompts_dict)
    
    # Process subjects
    centroid_files = list(cluster_dir.glob("*_centroids*.npy"))
    print(f"Matching {len(centroid_files)} subjects...")

    for cf in tqdm(centroid_files):
        subject_id = cf.name.split('_')[0]
        centroids = load_npy(cf)
        
        mapping, sim_matrix = matcher.match_clusters(centroids, text_embs, activity_names)
        
        # Save
        result = {
            "subject_id": subject_id,
            "dataset": args.dataset,
            "mapping": mapping
        }
        
        save_json(out_dir / f"{subject_id}_text_mapping.json", result)
        save_npy(out_dir / f"{subject_id}_cosine_matrix.npy", sim_matrix)

    print(f"Classification complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
