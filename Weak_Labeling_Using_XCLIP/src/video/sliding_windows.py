# Sliding Windows

from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class WindowMeta:
    window_index: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float

def uniform_sample_indices(total: int, num: int) -> np.ndarray:
    if total <= 0:
        raise ValueError("total must be > 0")
    if num <= 0:
        raise ValueError("num must be > 0")
    if total == 1:
        return np.zeros((num,), dtype=np.int64)
    return np.linspace(0, total - 1, num=num).astype(np.int64)

def iter_video_windows(
    video_path: str,
    clipsize_frames: int,
    stride_frames: int,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_idx = 0
    win_idx = 0
    next_emit_at = clipsize_frames

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frames.append(frame_bgr)
        frame_idx += 1

        if frame_idx == next_emit_at:
            start = frame_idx - clipsize_frames
            end = frame_idx
            meta = WindowMeta(
                window_index=win_idx,
                start_frame=start,
                end_frame=end,
                start_time_s=start / fps,
                end_time_s=end / fps,
            )
            yield frames[-clipsize_frames:], meta

            win_idx += 1
            next_emit_at += stride_frames

    cap.release()
