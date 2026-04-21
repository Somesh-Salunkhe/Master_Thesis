import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from blip2_har.utils.paths import get_project_root, get_data_dir, ensure_dir
from blip2_har.utils.data_io import save_npy
from blip2_har.features.extractor import BLIP2Extractor

def main():
    parser = argparse.ArgumentParser(description="Extract BLIP-2 features from videos.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model_name', type=str, default='Salesforce/blip2-opt-2.7b')
    parser.add_argument('--feature_type', type=str, default='qformer', choices=['qformer', 'vision'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--subjects', type=str, default='all', help='Subject IDs (comma separated or "all")')
    args = parser.parse_args()

    root = get_project_root()
    video_dir = get_data_dir() / args.dataset / "raw" / "videos"
    output_dir = get_data_dir() / args.dataset / "processed" / "blip2_features" / "12fps_framewise"
    ensure_dir(output_dir)

    extractor = BLIP2Extractor(model_name=args.model_name, device=f"cuda:{args.gpu}")

    # Detect subjects
    if args.subjects == 'all':
        video_files = list(video_dir.glob("*.mp4"))
        subject_ids = sorted([f.stem.split('-')[0].split('.')[0] for f in video_files])
        # Filter for unique IDs (e.g. sbj_0)
        subject_ids = sorted(list(set(subject_ids)))
    else:
        subject_ids = args.subjects.split(',')

    print(f"Extracting features for {len(subject_ids)} subjects...")

    for sbj_id in subject_ids:
        # Check for various naming conventions
        video_path = video_dir / f"{sbj_id}-12fps.mp4"
        if not video_path.exists():
            video_path = video_dir / f"{sbj_id}.mp4"
            
        if not video_path.exists():
            print(f"Warning: Video not found for {sbj_id}")
            continue

        print(f"Processing {sbj_id}...")
        features = extractor.extract_from_video(
            str(video_path), 
            target_fps=12, 
            feature_type=args.feature_type
        )
        
        save_npy(output_dir / f"{sbj_id}.npy", features)

    print(f"Extraction complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
