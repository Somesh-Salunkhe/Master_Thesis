import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path)

def save_npy(path: str | Path, data: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)

def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path: str | Path, data: Dict[str, Any], indent: int = 2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)
