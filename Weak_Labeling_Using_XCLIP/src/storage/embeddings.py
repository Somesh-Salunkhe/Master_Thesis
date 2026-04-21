# Storage of Embeddings

from pathlib import Path
import csv
import numpy as np
from src.video.sliding_windows import WindowMeta

def save_embeddings_npy(out_path: Path, embeddings: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings.astype(np.float32))

def save_window_index_csv(out_path: Path, metas: list[WindowMeta]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_index", "start_frame", "end_frame", "start_time_s", "end_time_s"])
        for m in metas:
            w.writerow([m.window_index, m.start_frame, m.end_frame, m.start_time_s, m.end_time_s])
