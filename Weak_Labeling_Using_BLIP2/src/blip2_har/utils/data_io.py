import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_npy(path: str | Path) -> np.ndarray:
    """Loads a numpy file."""
    return np.load(path)

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
