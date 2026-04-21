import sys
import os
import argparse
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weak_annotation.utils.paths import get_project_root, get_config_dir, get_output_dir, ensure_dir
from weak_annotation.utils.data_io import load_npy, save_npy
from weak_annotation.clustering.gmm import apply_gmm, get_centroids

def main():
    parser = argparse.ArgumentParser(description="Run GMM clustering on feature embeddings.")
    parser.add_argument('--dataset', type=str, default='wear', help='Dataset name (e.g., wear, wetlab)')
    parser.add_argument('--features', type=str, default='clip', choices=['clip', 'dino', 'raft', 'conv3d'], help='Features to use')
    parser.add_argument('--clusters', type=int, default=100, help='Number of clusters')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--clip_length', type=int, default=4, help='Clip length in seconds')
    parser.add_argument('--subjects', type=str, default='all', help='Subject IDs (comma separated or "all")')
    args = parser.parse_args()

    root = get_project_root()
    config_path = get_config_dir() / args.dataset / f'annotation_pipeline_{args.clip_length}sec.yaml'
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = get_output_dir() / f'{args.features}_clusters' / args.dataset
    ensure_dir(output_dir)

    # Determine subjects to process
    if args.subjects == 'all':
        # Automatically detect subjects from feature folder
        feature_folder = root / config['dataset'][f'{args.features}_folder']
        subject_files = list(feature_folder.glob("*.npy"))
        subject_ids = sorted([f.stem for f in subject_files])
    else:
        subject_ids = args.subjects.split(',')

    print(f"Clustering {len(subject_ids)} subjects from {args.dataset} using {args.features} features...")

    for sbj_id in tqdm(subject_ids):
        # Loading features
        feature_path = root / config['dataset'][f'{args.features}_folder'] / f"{sbj_id}.npy"
        
        # Fallback for naming inconsistencies (e.g. sbj_1 vs sbj_01)
        if not feature_path.exists():
            try:
                num = int(sbj_id.split('_')[1])
                alt_name = f"sbj_{num:02d}.npy"
                feature_path = root / config['dataset'][f'{args.features}_folder'] / alt_name
            except (IndexError, ValueError):
                pass
                
        if not feature_path.exists():
            print(f"Warning: Feature file not found for {sbj_id}")
            continue

        try:
            features = load_npy(feature_path)
            
            # Run Clustering
            labels, distances, gmm = apply_gmm(
                features, 
                k=args.clusters, 
                seed=args.seed
            )
            
            centroids = get_centroids(gmm)
            
            # Save results
            save_npy(output_dir / f'{sbj_id}_centroids.npy', centroids)
            save_npy(output_dir / f'{sbj_id}_labels.npy', labels)
            
        except Exception as e:
            print(f"Error processing {sbj_id}: {e}")

    print(f"Clustering complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
