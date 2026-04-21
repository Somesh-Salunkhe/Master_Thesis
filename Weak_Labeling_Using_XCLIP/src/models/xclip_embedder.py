# X-CLIP Embedder

import numpy as np
import torch
from transformers import AutoProcessor, AutoModel

class XClipEmbedder:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        self.num_frames = int(self.model.config.vision_config.num_frames)

    @torch.no_grad()
    def embed_window(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        idx = np.linspace(0, len(frames_bgr) - 1, num=self.num_frames).astype(np.int64)
        frames_rgb = []
        for i in idx:
            bgr = frames_bgr[int(i)]
            rgb = bgr[:, :, ::-1]
            frames_rgb.append(rgb)

        inputs = self.processor(videos=frames_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        video_feat = self.model.get_video_features(**inputs)
        video_feat = video_feat.detach().float().cpu().numpy()
        return video_feat[0]
