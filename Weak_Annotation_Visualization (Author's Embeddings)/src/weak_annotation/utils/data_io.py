import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_npy(path: str | Path) -> np.ndarray:
    """Loads a numpy file and converts to float32."""
    return np.load(path).astype(np.float32)

def save_npy(path: str | Path, data: np.ndarray):
    """Saves data to a numpy file, ensuring parent directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)

def load_json(path: str | Path) -> Dict[str, Any]:
    """Loads a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path: str | Path, data: Dict[str, Any], indent: int = 2):
    """Saves data to a JSON file, ensuring parent directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def time_to_seconds(t_str: str) -> int:
    """Converts a time string (HH:MM:SS or MM:SS) to total seconds."""
    try:
        if ':' not in str(t_str): return 0
        parts = str(t_str).split(':')
        if len(parts) == 2: # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3: # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        return 0
    return 0

def load_ground_truth(csv_path: Path, subject_id: str, dataset: str) -> Optional[pd.DataFrame]:
    """Loads ground truth for a specific subject from a CSV file."""
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    
    # Handle different subject ID formats in CSV (int vs string)
    try:
        subject_num = int(subject_id.split('_')[1])
    except (IndexError, ValueError):
        subject_num = subject_id
        
    sub_df = df[df['subject'] == subject_num]
    return sub_df

def get_ground_truth_labels(csv_path: Path, subject_id: str, dataset: str, n_samples: int) -> List[str]:
    """Generates a list of labels per second based on ground truth CSV."""
    gt_labels = ["Background"] * n_samples
    sub_df = load_ground_truth(csv_path, subject_id, dataset)
    
    if sub_df is not None:
        for _, row in sub_df.iterrows():
            s = time_to_seconds(row['start_time'])
            e = time_to_seconds(row['end_time'])
            activity = row.get('activity', row.get('label', 'Unknown'))
            for sec in range(s, e + 1):
                if sec < n_samples:
                    gt_labels[sec] = activity
    
    return gt_labels
