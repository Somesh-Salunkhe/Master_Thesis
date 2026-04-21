import sys
import argparse
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xclip_har.utils.paths import get_project_root, get_output_dir, ensure_dir
from xclip_har.utils.data_io import load_npy, load_json
from xclip_har.visualization.heatmaps import plot_cosine_heatmap
from xclip_har.visualization.projections import perform_tsne, plot_combined_projection
from xclip_har.visualization.timeline import plot_activity_timeline

def main():
    parser = argparse.ArgumentParser(description="Unified Visualization Tool for X-CLIP Pipeline")
    parser.add_argument('--type', type=str, required=True, 
                        choices=['timeline', 'heatmap', 'projection', 'all'], 
                        help='Type of visualization to generate')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subject_id', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, default=100)
    args = parser.parse_args()

    root = get_project_root()
    cluster_dir = get_output_dir() / f"clusters_xclip_{args.n_clusters}" / args.dataset
    match_dir = get_output_dir() / f"classification_xclip_{args.n_clusters}" / args.dataset
    
    vis_dir = get_output_dir() / "plots" / args.dataset / args.subject_id
    ensure_dir(vis_dir)

    # 1. Heatmap
    if args.type in ['heatmap', 'all']:
        print(f"Generating Heatmap for {args.subject_id}...")
        cosine_path = match_dir / f"{args.subject_id}_cosine_matrix.npy"
        names_path = get_output_dir() / "text_embeddings_xclip" / "activity_names.json"
        
        if cosine_path.exists() and names_path.exists():
            S = load_npy(cosine_path)
            activity_names = load_json(names_path)
            fig = plot_cosine_heatmap(S, activity_names, title=f"Heatmap - {args.subject_id}")
            fig.write_html(vis_dir / "heatmap.html")
            print(f"  Saved to {vis_dir / 'heatmap.html'}")

    # 2. Projection (t-SNE)
    if args.type in ['projection', 'all']:
        print(f"Generating t-SNE for {args.subject_id}...")
        emb_path = root / "data" / args.dataset / "processed" / "xclip_embeddings" / f"{args.subject_id}.npy"
        labels_path = cluster_dir / f"{args.subject_id}_labels{args.n_clusters}.npy"
        centroids_path = cluster_dir / f"{args.subject_id}_centroids{args.n_clusters}.npy"
        match_path = match_dir / f"{args.subject_id}_text_mapping.json"
        
        if all(p.exists() for p in [emb_path, labels_path, centroids_path, match_path]):
            X = load_npy(emb_path)
            labels = load_npy(labels_path)
            centroids = load_npy(centroids_path)
            mapping = load_json(match_path)['mapping']
            
            # Subsample for speed if needed
            if len(X) > 5000:
                idx = np.random.choice(len(X), 5000, replace=False)
                X = X[idx]
                labels = labels[idx]
                
            pts2d, cen2d = perform_tsne(X, centroids)
            text_labels = np.array([mapping.get(str(int(c)), f"C{c}") for c in labels])
            
            fig = plot_combined_projection(pts2d, cen2d, labels, text_labels, title=f"t-SNE - {args.subject_id}")
            fig.write_html(vis_dir / "projection.html")
            print(f"  Saved to {vis_dir / 'projection.html'}")

    # 3. Timeline
    if args.type in ['timeline', 'all']:
        print(f"Generating Timeline for {args.subject_id}...")
        # (Assuming ground truth is available in similar structure as other projects)
        labels_path = cluster_dir / f"{args.subject_id}_labels{args.n_clusters}.npy"
        match_path = match_dir / f"{args.subject_id}_text_mapping.json"
        
        if labels_path.exists() and match_path.exists():
            labels = load_npy(labels_path)
            mapping = load_json(match_path)['mapping']
            pred_labels = [mapping.get(str(int(c)), "Background") for c in labels]
            
            # Placeholder for GT
            gt_labels = ["Background"] * len(pred_labels)
            
            fig = plot_activity_timeline(gt_labels, pred_labels, title=f"Timeline - {args.subject_id}")
            fig.write_html(vis_dir / "timeline.html")
            print(f"  Saved to {vis_dir / 'timeline.html'}")

if __name__ == "__main__":
    main()
