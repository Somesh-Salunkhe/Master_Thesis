import sys
import argparse
import numpy as np
import moviepy.editor as mp
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xclip_har.utils.paths import get_data_dir, ensure_dir
from xclip_har.utils.data_io import save_npy
from xclip_har.features.extractor import XCLIPExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract X-CLIP embeddings from videos.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="microsoft/xclip-base-patch16-16-frames")
    parser.add_argument('--window_s', type=int, default=4)
    parser.add_argument('--stride_s', type=int, default=1)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    data_dir = get_data_dir() / args.dataset
    video_dir = data_dir / "processed" / f"videos_fps{args.fps}"
    
    if not video_dir.exists():
        # Fallback to raw if processed doesn't exist
        video_dir = data_dir / "raw" / "videos"

    output_dir = data_dir / "processed" / "xclip_embeddings"
    ensure_dir(output_dir)

    extractor = XCLIPExtractor(model_name=args.model_name, device=f"cuda:{args.gpu}")

    window_f = args.window_s * args.fps
    stride_f = args.stride_s * args.fps

    video_files = list(video_dir.glob("*.mp4"))
    print(f"Extracting features for {len(video_files)} videos...")

    for vid_path in video_files:
        subject_id = vid_path.stem
        print(f"Processing {subject_id}...")
        
        video = mp.VideoFileClip(str(vid_path))
        embs = []
        
        # Sliding window iteration
        num_frames = int(video.duration * args.fps)
        for start in tqdm(range(0, num_frames - window_f + 1, stride_f), leave=False):
            # Sample frames from window
            frames = []
            for f_idx in np.linspace(start, start + window_f - 1, extractor.num_frames, dtype=int):
                t = f_idx / args.fps
                if t < video.duration:
                    frames.append(video.get_frame(t))
            
            if len(frames) == extractor.num_frames:
                emb = extractor.embed_window(frames)
                embs.append(emb)
        
        video.close()
        
        if embs:
            embs_arr = np.stack(embs, axis=0)
            save_npy(output_dir / f"{subject_id}.npy", embs_arr)
            print(f"  Saved {embs_arr.shape} to {output_dir}")

if __name__ == "__main__":
    main()
