import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from blip2_har.utils.paths import get_project_root, get_output_dir, get_data_dir, ensure_dir
from blip2_har.utils.data_io import load_json, save_json
from blip2_har.captions.generator import BLIP2Captioner

def main():
    parser = argparse.ArgumentParser(description="Generate staged captions for clusters.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--cluster_dir', type=str, help='Path to clustering results')
    parser.add_argument('--stages', type=str, default='stage_A,stage_B,stage_C,stage_D')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    root = get_project_root()
    video_dir = get_data_dir() / args.dataset / "raw" / "videos"
    
    if args.cluster_dir:
        cluster_base = Path(args.cluster_dir)
    else:
        cluster_base = get_output_dir() / "clustering" / args.dataset

    staged_samples_path = cluster_base / 'staged_samples.json'
    mapping_path = cluster_base / 'subject_mapping.json'
    
    if not staged_samples_path.exists():
        print(f"Error: Missing {staged_samples_path}")
        return

    staged_samples = load_json(staged_samples_path)
    subject_mapping = load_json(mapping_path)

    captioner = BLIP2Captioner(device=f"cuda:{args.gpu}")
    output_dir = get_output_dir() / "captions" / args.dataset
    ensure_dir(output_dir)

    stages_to_process = args.stages.split(',')
    
    for stage_name in stages_to_process:
        if stage_name not in staged_samples: continue
        
        print(f"Processing {stage_name}...")
        stage_samples = staged_samples[stage_name]
        stage_captions = {}
        
        for cluster_id, sample_indices in tqdm(stage_samples.items()):
            cluster_captions = []
            for idx in sample_indices:
                sbj_id, window_idx = subject_mapping[idx]
                video_path = video_dir / f"{sbj_id}-12fps.mp4"
                if not video_path.exists(): video_path = video_dir / f"{sbj_id}.mp4"
                
                if not video_path.exists(): continue
                
                try:
                    frames = captioner.get_window_frames(str(video_path), window_idx)
                    frame_results = []
                    for i, frame in enumerate(frames):
                        cap = captioner.generate_caption(frame)
                        frame_results.append({'frame_idx': i, 'caption': cap})
                    
                    cluster_captions.append({
                        'sample_idx': idx,
                        'subject': sbj_id,
                        'window': window_idx,
                        'captions': frame_results
                    })
                except Exception as e:
                    print(f"Error at sample {idx}: {e}")
            
            stage_captions[cluster_id] = cluster_captions
        
        save_json(output_dir / f"{stage_name}_captions.json", stage_captions)

    print("Captioning complete.")

if __name__ == "__main__":
    main()
