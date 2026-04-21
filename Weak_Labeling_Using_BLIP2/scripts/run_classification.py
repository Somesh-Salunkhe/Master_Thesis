import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from blip2_har.utils.paths import get_project_root, get_output_dir, ensure_dir
from blip2_har.utils.data_io import load_npy, save_npy, save_json
from blip2_har.classification.itc_classifier import ITCClassifier

def main():
    parser = argparse.ArgumentParser(description="Map clusters to semantic labels using BLIP-2 ITC.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--cluster_dir', type=str, help='Direct path to clustering results')
    parser.add_argument('--model_name', type=str, default="Salesforce/blip2-itm-vit-g")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    root = get_project_root()
    if args.cluster_dir:
        cluster_base = Path(args.cluster_dir)
    else:
        cluster_base = get_output_dir() / "clustering" / args.dataset

    # Load prompts
    prompts_path = root / "configs" / "prompts" / f"{args.dataset}_prompts.json"
    if not prompts_path.exists():
        # Check fallback
        prompts_path = root / "Text_classifier" / f"{args.dataset}_prompts.json"
    
    if not prompts_path.exists():
        print(f"Error: Prompts file not found at {prompts_path}")
        return

    with open(prompts_path, "r") as f:
        class_prompts = json.load(f)

    # Initialize Classifier
    classifier = ITCClassifier(model_name=args.model_name, device=f"cuda:{args.gpu}")
    
    print("Building text prototypes...")
    class_names, text_prototypes = classifier.build_text_prototypes(class_prompts)

    # Identify subjects
    subjects = sorted([d for d in cluster_base.iterdir() if d.is_dir() and d.name.startswith("sbj_")])
    is_separate = len(subjects) > 0

    if not is_separate:
        subjects = [cluster_base] # Combined mode

    print(f"Classifying {len(subjects)} subject/cluster directories...")

    for sbj_dir in tqdm(subjects):
        cf_path = sbj_dir / "cluster_centers.npy"
        if not cf_path.exists():
            continue

        centroids = load_npy(cf_path)
        vision_feats = classifier.project_vision_features(centroids)
        top_idx, top_probs, all_probs = classifier.classify(vision_feats, text_prototypes)

        mapping = {int(i): class_names[top_idx[i, 0]] for i in range(len(centroids))}
        
        # Save
        result = {
            "subject_id": sbj_dir.name if is_separate else "combined",
            "dataset": args.dataset,
            "mapping": mapping
        }
        
        save_json(sbj_dir / "text_mapping.json", result)
        save_npy(sbj_dir / "probs.npy", all_probs)
        
        with open(sbj_dir / "classification_summary.txt", "w") as f:
            f.write(f"Dataset: {args.dataset}\n\n")
            for i, (k, v) in enumerate(sorted(mapping.items())):
                f.write(f"Cluster {int(k):03d}: {v:35s} (p={top_probs[i, 0]:.4f})\n")

    print(f"Classification complete.")

if __name__ == "__main__":
    main()
