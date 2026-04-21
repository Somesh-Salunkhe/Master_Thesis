import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weak_annotation.utils.paths import get_project_root, get_output_dir, ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Evaluate weak annotation results across seeds.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--network', type=str, default='deepconvlstm')
    parser.add_argument('--type', type=str, default='weak_ce', help='weak_ce or baseline')
    parser.add_argument('--clusters', type=int, default=100)
    parser.add_argument('--seeds', type=str, default='1,2,3', help='Comma separated seed values')
    args = parser.parse_args()

    root = get_project_root()
    seeds = [int(s) for s in args.seeds.split(',')]
    
    # Path construction
    if args.type == 'baseline':
        base_path = root / 'experiments' / args.dataset / args.network / args.type
    else:
        base_path = root / 'experiments' / args.dataset / args.network / args.type / str(args.clusters)

    # Automatically find subject JSON files
    anno_dir = root / 'data' / args.dataset / 'annotations'
    json_files = list(anno_dir.glob("loso_sbj_*.json"))
    
    if not json_files:
        print(f"Error: No annotation JSONs found in {anno_dir}")
        return

    # Determine num_classes from first JSON
    with open(json_files[0]) as f:
        meta = json.load(f)
        num_classes = len(meta['label_dict']) + (0 if args.dataset == 'rwhar' else 1)
        labels = (['null'] if args.dataset != 'rwhar' else []) + list(meta['label_dict'].keys())

    all_acc = np.zeros((len(seeds), num_classes))
    all_f1 = np.zeros((len(seeds), num_classes))

    print(f"Evaluating {args.dataset} ({num_classes} classes) across {len(seeds)} seeds...")

    for s_pos, seed in enumerate(seeds):
        all_preds = []
        all_gt = []

        for j_file in json_files:
            with open(j_file) as f:
                anno_data = json.load(f)
                label_dict = {name: i for i, name in enumerate(labels)}
                val_sbjs = [k for k, v in anno_data['database'].items() if v['subset'] == 'Validation']
            
            for sbj in val_sbjs:
                raw_path = root / 'data' / args.dataset / 'raw' / 'inertial' / f"{sbj}.csv"
                if not raw_path.exists(): continue
                
                df = pd.read_csv(raw_path).replace({"label": label_dict}).fillna(0)
                data = df.to_numpy()
                
                # Load predictions for this seed and subject
                # Note: This follows the legacy output structure for compatibility
                pred_path = base_path / f"seed_{seed}" / "unprocessed_results" / f"v_preds_{j_file.stem}.npy"
                if not pred_path.exists(): continue
                
                v_orig_preds = np.load(pred_path)
                sbj_num = int(sbj.split("_")[-1])
                v_preds = v_orig_preds[data[:, 0] == sbj_num]

                all_preds.extend(v_preds)
                all_gt.extend(data[:, -1])

        if not all_preds:
            continue

        all_preds = np.array(all_preds)
        all_gt = np.array(all_gt)

        # Metrics
        conf = confusion_matrix(all_gt, all_preds, normalize='true', labels=range(num_classes))
        all_acc[s_pos, :] = conf.diagonal()
        all_f1[s_pos, :] = f1_score(all_gt, all_preds, average=None, labels=range(num_classes))

        # Confusion Matrix Plot for first seed
        if seed == seeds[0]:
            plt.figure(figsize=(12, 10))
            sns.heatmap(conf, annot=True, fmt='.2f', cmap='Greens', xticklabels=labels, yticklabels=labels)
            plt.title(f'Confusion Matrix - Seed {seed}')
            plt.tight_layout()
            out_vis = get_output_dir() / "visualizations" / "evaluation"
            ensure_dir(out_vis)
            plt.savefig(out_vis / f"{args.dataset}_{args.type}_confusion.pdf")
            print(f"  Confusion matrix saved to {out_vis}")

    print("\n--- Evaluation Summary ---")
    print(f"Average Accuracy: {np.mean(all_acc)*100:.2f}% (+/- {np.std(np.mean(all_acc, axis=1))*100:.2f})")
    print(f"Average F1 Score: {np.mean(all_f1)*100:.2f}% (+/- {np.std(np.mean(all_f1, axis=1))*100:.2f})")

if __name__ == "__main__":
    main()
