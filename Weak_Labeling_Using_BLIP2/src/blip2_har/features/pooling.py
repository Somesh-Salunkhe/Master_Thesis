import numpy as np
from typing import Tuple

def average_pool_features(
    features: np.ndarray, 
    window_size_frames: int, 
    stride_frames: int
) -> np.ndarray:
    """
    Applies sliding window average pooling to framewise features.
    
    Args:
        features: Framewise features (N, D).
        window_size_frames: Size of the window in frames.
        stride_frames: Stride in frames.
        
    Returns:
        pooled_features: Window-wise features (M, D).
    """
    num_frames = features.shape[0]
    dim = features.shape[1]
    
    pooled = []
    for start in range(0, num_frames - window_size_frames + 1, stride_frames):
        end = start + window_size_frames
        window_feat = features[start:end]
        pooled.append(window_feat.mean(axis=0))
        
    return np.array(pooled)
