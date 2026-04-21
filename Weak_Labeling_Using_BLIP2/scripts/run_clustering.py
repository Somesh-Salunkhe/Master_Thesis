import sys
import argparse
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from blip2_har.utils.paths import get_project_root, get_output_dir, ensure_dir
from blip2_har.utils.data_io import load_npy, save_npy, save_json
from blip2_har.clustering.gmm import apply_gmm

def get_nearest_samples(labels, distances, n_samples, n_clusters):
    """Helper to get nearest indices per cluster."""
    sample_indices = {}
    for c in range(n_clusters):
        cluster_mask = labels == c
        cluster_indices = np.where(cluster_mask)[0]
        cluster_distances = distances[cluster_mask]
        sorted_order = np.argsort(cluster_distances)
        sorted_indices = cluster_indices[sorted_order]
        sample_indices[c] = sorted_indices[:min(n_samples, len(sorted_indices))].tolist()
    return sample_indices

def process_clustering(features, subject_info, output_dir, n_clusters, seed):
    """Core clustering and saving logic."""
    ensure_dir(output_dir)
    labels, distances, gmm = apply_gmm(features, n_clusters, seed=seed)
    
    # Generate centers (highest density points)
    from sklearn.metrics import pairwise_distances
    import scipy.stats
    centers = np.empty((n_clusters, features.shape[1]))
    center_indices = np.empty(n_clusters, dtype=int)
    for i in range(n_clusters):
        try:
            density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(features)
            center_idx = np.argmax(density)
            centers[i, :] = features[center_idx]
            center_indices[i] = center_idx
        except:
            centers[i, :] = gmm.means_[i]
            center_indices[i] = 0

    # Staged sampling
    stages = {'stage_A': 1, 'stage_B': 11, 'stage_C': 51, 'stage_D': 101}
    staged_samples = {name: get_nearest_samples(labels, distances, count, n_clusters) for name, count in stages.items()}
    
    save_npy(output_dir / 'cluster_labels.npy', labels)
    save_npy(output_dir / 'cluster_distances.npy', distances)
    save_npy(output_dir / 'cluster_centers.npy', centers)
    save_json(output_dir / 'staged_samples.json', staged_samples)
    save_json(output_dir / 'subject_mapping.json', subject_info)

def main():
    parser = argparse.ArgumentParser(description="Cluster pooled features using GMM.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--input_dir', type=str, help='Input directory (defaults to processed data)')
    parser.add_argument('--n_clusters', type=int, default=100)
    parser.add_argument('--separate', action='store_true', help='Cluster each subject separately')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    root = get_project_root()
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        # Default pooled dir: 48 frames, 12 stride
        input_dir = root / "data" / args.dataset / "processed" / "blip2_features" / "48_frames_12_stride_avg_pool"

    output_base = get_output_dir() / "clustering" / args.dataset
    if args.separate:
        output_base = get_output_dir() / "clustering" / f"{args.dataset}_separate"

    feat_files = sorted(list(input_dir.glob("*.npy")))
    
    if args.separate:
        for ff in tqdm(feat_files):
            sbj_id = ff.stem
            features = load_npy(ff)
            subject_info = [(sbj_id, i) for i in range(len(features))]
            process_clustering(features, subject_info, output_base / sbj_id, args.n_clusters, args.seed)
    else:
        all_features = []
        subject_info = []
        for ff in tqdm(feat_files):
            features = load_npy(ff)
            all_features.append(features)
            subject_info.extend([(ff.stem, i) for i in range(len(features))])
        
        combined_features = np.vstack(all_features)
        process_clustering(combined_features, subject_info, output_base, args.n_clusters, args.seed)

if __name__ == "__main__":
    main()
