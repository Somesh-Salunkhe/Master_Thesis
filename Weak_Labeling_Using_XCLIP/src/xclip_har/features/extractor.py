import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from typing import List, Optional

class XCLIPExtractor:
    """
    Extracts video features using the X-CLIP model.
    """
    def __init__(
        self, 
        model_name: str = "microsoft/xclip-base-patch16-16-frames", 
        device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Loading X-CLIP model '{model_name}' on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        self.num_frames = int(self.model.config.vision_config.num_frames)

    @torch.no_grad()
    def embed_window(self, frames_list: List[np.ndarray]) -> np.ndarray:
        """
        Extracts a feature vector for a single window of frames.
        
        Args:
            frames_list: List of numpy arrays (H, W, 3) in BGR or RGB.
        """
        # Sampling frames to match model's expected num_frames
        idx = np.linspace(0, len(frames_list) - 1, num=self.num_frames).astype(np.int64)
        
        frames_to_process = []
        for i in idx:
            frame = frames_list[int(i)]
            # Assume BGR input and convert to RGB if needed (standard for OpenCV/MoviePy)
            # Actually, MoviePy gives RGB, but if using OpenCV it's BGR.
            # I'll provide a toggle or assume RGB for standard pipeline consistency.
            frames_to_process.append(frame)

        inputs = self.processor(videos=frames_to_process, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        video_feat = self.model.get_video_features(**inputs)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        
        return video_feat.detach().float().cpu().numpy()[0]
