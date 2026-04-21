import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xclip_har.utils.paths import get_data_dir, get_output_dir, ensure_dir
from xclip_har.utils.data_io import load_npy, save_npy
from xclip_har.clustering.gmm import apply_gmm

def main():
    parser = argparse.ArgumentParser(description="Cluster X-CLIP embeddings using GMM.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--subjects', type=str, default='all')
    args = parser.parse_args()

    input_dir = get_data_dir() / args.dataset / "processed" / "xclip_embeddings"
    output_dir = get_output_dir() / f"clusters_xclip_{args.n_clusters}" / args.dataset
    ensure_dir(output_dir)

    # Detect subjects
    if args.subjects == 'all':
        feat_files = sorted(list(input_dir.glob("*.npy")))
    else:
        feat_files = [input_dir / f"{s}.npy" for s in args.subjects.split(',')]

    print(f"Clustering {len(feat_files)} subjects...")

    for ff in tqdm(feat_files):
        if not ff.exists(): continue
        
        subject_id = ff.stem
        features = load_npy(ff)
        
        labels, distances, gmm = apply_gmm(features, args.n_clusters, seed=args.seed)
        
        # Save results (legacy naming convention for compatibility)
        save_npy(output_dir / f"{subject_id}_labels{args.n_clusters}.npy", labels)
        save_npy(output_dir / f"{subject_id}_distances{args.n_clusters}.npy", distances)
        
        # Calculate centroids (mean of points in cluster)
        centroids = np.stack([features[labels == i].mean(axis=0) for i in range(args.n_clusters)])
        save_npy(output_dir / f"{subject_id}_centroids{args.n_clusters}.npy", centroids)

    print(f"Clustering complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
