import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xclip_har.utils.paths import get_data_dir, ensure_dir
from xclip_har.features.preprocessor import preprocess_video

def main():
    parser = argparse.ArgumentParser(description="Preprocess videos using FFmpeg.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--resize', type=int, default=224)
    args = parser.parse_args()

    data_dir = get_data_dir() / args.dataset
    raw_vid_dir = data_dir / "raw" / "videos"
    
    if not raw_vid_dir.exists():
        print(f"Error: Raw video directory not found at {raw_vid_dir}")
        return

    out_dir = data_dir / "processed" / f"videos_fps{args.fps}"
    ensure_dir(out_dir)

    video_files = list(raw_vid_dir.glob("*.mp4"))
    print(f"Preprocessing {len(video_files)} videos...")

    for vid in tqdm(video_files):
        out_path = out_dir / vid.name
        preprocess_video(vid, out_path, fps=args.fps, resize_shorter_side=args.resize)

    print(f"Preprocessing complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
