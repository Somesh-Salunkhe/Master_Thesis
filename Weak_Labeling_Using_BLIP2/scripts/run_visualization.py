import sys
import argparse
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from blip2_har.utils.paths import get_project_root, get_output_dir, ensure_dir
from blip2_har.utils.data_io import load_npy, load_json
from blip2_har.utils.srt_helpers import generate_srt_content

def main():
    parser = argparse.ArgumentParser(description="Unified Visualization Tool for BLIP-2 Pipeline")
    parser.add_argument('--type', type=str, required=True, choices=['timeline', 'projection', 'srt', 'all'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subject_id', type=str, help='Subject ID (e.g., sbj_1)')
    parser.add_argument('--cluster_dir', type=str, help='Path to clustering results')
    args = parser.parse_args()

    root = get_project_root()
    if args.cluster_dir:
        cluster_base = Path(args.cluster_dir)
    else:
        cluster_base = get_output_dir() / "clustering" / args.dataset

    vis_dir = get_output_dir() / "visualizations" / args.dataset
    ensure_dir(vis_dir)

    # 1. SRT Generation
    if args.type in ['srt', 'all']:
        if not args.subject_id:
            print("Error: --subject_id is required for SRT generation.")
        else:
            print(f"Generating SRT for {args.subject_id}...")
            labels_path = cluster_base / "cluster_labels.npy"
            mapping_path = cluster_base / "subject_mapping.json"
            
            if labels_path.exists() and mapping_path.exists():
                labels = load_npy(labels_path)
                mapping = load_json(mapping_path)
                
                # Filter for subject
                idx_list = [(i, win) for i, (sbj, win) in enumerate(mapping) if str(sbj) == args.subject_id.replace("sbj_", "")]
                idx_list.sort(key=lambda x: x[1])
                
                if not idx_list:
                    # Try with "sbj_X" format
                    idx_list = [(i, win) for i, (sbj, win) in enumerate(mapping) if str(sbj) == args.subject_id]
                    idx_list.sort(key=lambda x: x[1])

                if idx_list:
                    content = generate_srt_content(idx_list, labels)
                    with open(vis_dir / f"{args.subject_id}_clusters.srt", "w") as f:
                        f.write(content)
                    print(f"  Saved to {vis_dir / f'{args.subject_id}_clusters.srt'}")
                else:
                    print(f"  Warning: No data found for {args.subject_id} in mapping.")
            else:
                print(f"  Error: Clustering results not found in {cluster_base}")

    # Note: Timeline and Projection would follow similar logic as the previous project
    # using reusable modules in src/blip2_har/visualization/

if __name__ == "__main__":
    main()
