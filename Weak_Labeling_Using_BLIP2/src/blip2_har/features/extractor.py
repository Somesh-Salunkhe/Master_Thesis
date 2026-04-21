import torch
import numpy as np
import moviepy.editor as mp
from transformers import Blip2Processor, Blip2Model
from tqdm import tqdm
from typing import Optional, List, Tuple

class BLIP2Extractor:
    def __init__(
        self, 
        model_name: str = "Salesforce/blip2-opt-2.7b", 
        device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Loading BLIP-2 model '{model_name}' on {device}...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.model.eval()

    @torch.no_grad()
    def extract_from_video(
        self, 
        video_path: str, 
        target_fps: int = 12, 
        feature_type: str = 'qformer'
    ) -> np.ndarray:
        """Extracts framewise features from a video file."""
        video = mp.VideoFileClip(video_path)
        duration = video.duration
        length_video = int(duration * target_fps)
        
        features = []
        
        for i, frame in enumerate(tqdm(video.iter_frames(fps=target_fps), total=length_video, desc="Extracting features")):
            inputs = self.processor(images=frame, return_tensors="pt").to(self.device, torch.float16)
            
            if feature_type == 'qformer':
                # Get Q-Former output (768-dim)
                outputs = self.model.get_qformer_features(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    feature = outputs.last_hidden_state.mean(dim=1).cpu()
                else:
                    feature = outputs.mean(dim=1).cpu()
            else:
                # Get vision encoder output
                outputs = self.model.vision_model(**inputs)
                feature = outputs.pooler_output.cpu()
            
            features.append(feature.squeeze().numpy())
            
            if i % 500 == 0:
                torch.cuda.empty_cache()

        video.close()
        return np.array(features)
