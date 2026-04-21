import sys
import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weak_annotation.utils.paths import get_project_root, get_output_dir, ensure_dir
from weak_annotation.utils.data_io import load_npy, save_json, save_npy
from weak_annotation.classification.clip_classifier import CLIPClassifier

def main():
    parser = argparse.ArgumentParser(description="Map clusters to semantic labels using CLIP.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., wear, wetlab)')
    parser.add_argument('--model', type=str, required=True, help='Model/Cluster folder name (e.g., clip_clusters)')
    parser.add_argument('--prompts', type=str, default=None, help='Path to prompts JSON (optional)')
    parser.add_argument('--clip_model', type=str, default="openai/clip-vit-large-patch14", help='CLIP model version')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()

    root = get_project_root()
    cluster_dir = get_output_dir() / args.model / args.dataset
    
    if not cluster_dir.exists():
        print(f"Error: Cluster directory not found at {cluster_dir}")
        return

    # Load prompts
    if args.prompts:
        prompts_path = Path(args.prompts)
    else:
        # Try to find in the default location
        prompts_path = root / "configs" / "prompts" / f"{args.dataset}_prompts.json"
        if not prompts_path.exists():
            # Fallback to old location for compatibility during transition
            prompts_path = root / "experiments" / "text_embeddings_and_classifier" / f"{args.dataset}_prompts.json"

    if not prompts_path.exists():
        print(f"Error: Prompts file not found at {prompts_path}")
        return

    with open(prompts_path, 'r') as f:
        class_prompts = json.load(f)

    # Initialize Classifier
    classifier = CLIPClassifier(model_name=args.clip_model, device=args.device)
    
    # Build Prototypes
    print(f"Building text prototypes for {args.dataset}...")
    class_names, prototypes = classifier.build_text_prototypes(class_prompts)
    
    # Output Directory
    output_dir = cluster_dir / "text_classifier_results"
    ensure_dir(output_dir)

    # Save Class Metadata
    save_npy(output_dir / f"{args.dataset}_text_prototypes.npy", prototypes.cpu().numpy())
    save_json(output_dir / f"{args.dataset}_class_names.json", class_names)

    # Process subjects
    centroid_files = sorted(list(cluster_dir.glob("*_centroids.npy")))
    print(f"Classifying clusters for {len(centroid_files)} subjects...")

    for cf in tqdm(centroid_files):
        subject_id = cf.name.replace("_centroids.npy", "")
        
        # Load centroids
        centroids = load_npy(cf)
        feats = torch.from_numpy(centroids).to(classifier.device).float()

        # Classify
        top_idx, top_probs, all_probs = classifier.classify_embeddings(feats, prototypes)
        
        # Create mapping
        mapping = {int(i): class_names[top_idx[i, 0]] for i in range(len(centroids))}
        
        # Save results
        save_json(output_dir / f"{subject_id}_text_mapping.json", {
            "subject_id": subject_id,
            "dataset": args.dataset,
            "mapping": mapping
        })
        save_npy(output_dir / f"{subject_id}_probs.npy", all_probs)
        
        # Save summary text
        with open(output_dir / f"{subject_id}_classification_summary.txt", "w") as f:
            f.write(f"Subject: {subject_id}\nDataset: {args.dataset}\n\n")
            for i in range(len(centroids)):
                f.write(f"Cluster {i:02d}: {mapping[i]:30s} (p={top_probs[i, 0]:.4f})\n")

    print(f"Classification complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
