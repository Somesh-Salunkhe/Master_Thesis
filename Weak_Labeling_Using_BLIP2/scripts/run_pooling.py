import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from blip2_har.utils.paths import get_data_dir, ensure_dir
from blip2_har.utils.data_io import load_npy, save_npy
from blip2_har.features.pooling import average_pool_features

def main():
    parser = argparse.ArgumentParser(description="Average pool framewise features into sliding windows.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--window_s', type=int, default=4, help='Window size in seconds')
    parser.add_argument('--stride_s', type=int, default=1, help='Stride in seconds')
    parser.add_argument('--fps', type=int, default=12)
    args = parser.parse_args()

    input_dir = get_data_dir() / args.dataset / "processed" / "blip2_features" / f"{args.fps}fps_framewise"
    
    window_f = args.window_s * args.fps
    stride_f = args.stride_s * args.fps
    
    output_dir = get_data_dir() / args.dataset / "processed" / "blip2_features" / f"{window_f}_frames_{stride_f}_stride_avg_pool"
    ensure_dir(output_dir)

    feature_files = sorted(list(input_dir.glob("*.npy")))
    print(f"Pooling {len(feature_files)} feature files...")

    for ff in tqdm(feature_files):
        features = load_npy(ff)
        pooled = average_pool_features(features, window_f, stride_f)
        save_npy(output_dir / ff.name, pooled)

    print(f"Pooling complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
