import sys
import argparse
import numpy as np
import yaml
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weak_annotation.utils.paths import get_project_root, get_output_dir, get_config_dir, ensure_dir
from weak_annotation.utils.data_io import load_npy, load_json, get_ground_truth_labels
from weak_annotation.visualization.timeline import plot_activity_timeline
from weak_annotation.visualization.heatmaps import plot_cosine_heatmap, plot_probability_heatmap
from weak_annotation.visualization.projections import perform_tsne, plot_combined_projection

def main():
    parser = argparse.ArgumentParser(description="Unified Visualization Tool for Weak Annotation Pipeline")
    parser.add_argument('--type', type=str, required=True, 
                        choices=['timeline', 'heatmap', 'projection', 'all'], 
                        help='Type of visualization to generate')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--subject_id', type=str, required=True, help='Subject ID (e.g., sbj_0)')
    parser.add_argument('--features', type=str, default='clip', help='Feature type used for clustering')
    parser.add_argument('--clip_length', type=int, default=4, help='Clip length in seconds')
    
    args = parser.parse_args()
    root = get_project_root()
    
    # Paths
    cluster_base = get_output_dir() / f"{args.features}_clusters" / args.dataset
    label_path = cluster_base / f"{args.subject_id}_labels.npy"
    centroid_path = cluster_base / f"{args.subject_id}_centroids.npy"
    mapping_path = cluster_base / "text_classifier_results" / f"{args.subject_id}_text_mapping.json"
    probs_path = cluster_base / "text_classifier_results" / f"{args.subject_id}_probs.npy"
    
    vis_dir = get_output_dir() / "visualizations" / args.dataset / args.subject_id
    ensure_dir(vis_dir)

    # 1. Timeline Visualization
    if args.type in ['timeline', 'all']:
        print(f"Generating Timeline for {args.subject_id}...")
        if not label_path.exists() or not mapping_path.exists():
            print("  Error: Missing labels or mapping for timeline.")
        else:
            cluster_labels = load_npy(label_path)
            mapping = load_json(mapping_path)['mapping']
            pred_labels = [mapping.get(str(int(c)), "Background") for c in cluster_labels]
            
            # Load Ground Truth
            config_path = get_config_dir() / args.dataset / f'annotation_pipeline_{args.clip_length}sec.yaml'
            csv_path = None
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                csv_rel = config['dataset'].get('ground_truth_csv')
                if csv_rel:
                    csv_path = root / csv_rel
            
            gt_labels = ["Background"] * len(pred_labels)
            if csv_path and csv_path.exists():
                gt_labels = get_ground_truth_labels(csv_path, args.subject_id, args.dataset, len(pred_labels))
            
            fig = plot_activity_timeline(
                gt_labels, pred_labels, 
                title=f"Activity Timeline - {args.subject_id} ({args.dataset})"
            )
            fig.write_html(vis_dir / "timeline.html")
            print(f"  Saved to {vis_dir / 'timeline.html'}")

    # 2. Heatmap Visualization
    if args.type in ['heatmap', 'all']:
        print(f"Generating Heatmaps for {args.subject_id}...")
        if not probs_path.exists():
            print("  Error: Missing probability file for heatmap.")
        else:
            probs = load_npy(probs_path)
            class_names_path = cluster_base / "text_classifier_results" / f"{args.dataset}_class_names.json"
            if class_names_path.exists():
                class_names = load_json(class_names_path)
                fig = plot_probability_heatmap(
                    probs, class_names, 
                    title=f"Probability Heatmap - {args.subject_id}"
                )
                fig.write_html(vis_dir / "heatmap_probs.html")
                print(f"  Saved to {vis_dir / 'heatmap_probs.html'}")
            else:
                print("  Error: Missing class names for heatmap.")

    # 3. Projection Visualization
    if args.type in ['projection', 'all']:
        print(f"Generating t-SNE Projection for {args.subject_id}...")
        if not centroid_path.exists() or not mapping_path.exists():
            print("  Error: Missing centroids or mapping for projection.")
        else:
            # Need raw features for projection
            config_path = get_config_dir() / args.dataset / f'annotation_pipeline_{args.clip_length}sec.yaml'
            feature_path = None
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                feat_folder = root / config['dataset'][f'{args.features}_folder']
                feature_path = feat_folder / f"{args.subject_id}.npy"
                # Fallback
                if not feature_path.exists():
                    try:
                        num = int(args.subject_id.split('_')[1])
                        feature_path = feat_folder / f"sbj_{num:02d}.npy"
                    except: pass

            if feature_path and feature_path.exists():
                features = load_npy(feature_path)
                centroids = load_npy(centroid_path)
                mapping = load_json(mapping_path)['mapping']
                
                # Run t-SNE
                pts2d, cen2d, indices = perform_tsne(features, centroids)
                
                # Nearest centroids for subsampled points
                from sklearn.metrics import pairwise_distances_argmin
                X_sub = features[indices]
                cluster_ids = pairwise_distances_argmin(X_sub, centroids)
                text_labels = np.array([mapping.get(str(int(c)), f"C{c}") for c in cluster_ids])
                
                fig = plot_combined_projection(
                    pts2d, cen2d, cluster_ids, text_labels,
                    title=f"t-SNE Projection - {args.subject_id}"
                )
                fig.write_html(vis_dir / "projection.html")
                print(f"  Saved to {vis_dir / 'projection.html'}")
            else:
                print(f"  Error: Feature file not found at {feature_path}")

if __name__ == "__main__":
    main()
