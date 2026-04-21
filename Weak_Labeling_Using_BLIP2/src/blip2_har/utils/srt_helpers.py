import datetime
from typing import List, Tuple, Optional

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds_rem = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds_rem:02},{millis:03}"

def generate_srt_content(
    indices: List[Tuple[int, int]], 
    cluster_labels: List[int], 
    stride: float = 12.0, 
    fps: float = 12.0,
    activity_labels: Optional[List[int]] = None,
    activity_names: Optional[List[str]] = None
) -> str:
    """
    Generates the content for an SRT file based on cluster labels and window indices.
    
    Args:
        indices: List of (global_index, window_index) sorted by window_index.
        cluster_labels: Array of cluster IDs.
        stride: Stride in frames.
        fps: Frames per second.
        
    Returns:
        SRT formatted string.
    """
    lines = []
    seconds_per_step = stride / fps
    
    for seq_idx, (global_idx, win_idx) in enumerate(indices):
        cluster_id = cluster_labels[global_idx]
        
        start_seconds = (win_idx * stride) / fps
        end_seconds = start_seconds + seconds_per_step
        
        start_str = seconds_to_srt_time(start_seconds)
        end_str = seconds_to_srt_time(end_seconds)
        
        info = f"Cluster: {cluster_id}"
        
        if activity_labels is not None and activity_names is not None:
            gt_idx = activity_labels[global_idx]
            gt_name = activity_names[gt_idx] if gt_idx < len(activity_names) else "Unknown"
            info += f" | GT: {gt_name}"
            
        lines.append(f"{seq_idx + 1}")
        lines.append(f"{start_str} --> {end_str}")
        lines.append(f"{info}")
        lines.append("")
        
    return "\n".join(lines)
