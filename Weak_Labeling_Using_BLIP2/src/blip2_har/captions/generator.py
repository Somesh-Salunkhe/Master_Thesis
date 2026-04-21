import torch
import numpy as np
from PIL import Image
import moviepy.editor as mp
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
from typing import List, Dict, Any, Optional

class BLIP2Captioner:
    def __init__(
        self, 
        model_name: str = "Salesforce/blip2-opt-2.7b", 
        device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Loading BLIP-2 generation model '{model_name}' on {device}...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16
        ).to(device)
        self.model.eval()

    def get_window_frames(
        self, 
        video_path: str, 
        window_idx: int, 
        window_size: int = 48, 
        stride: int = 12, 
        fps: int = 12, 
        num_samples: int = 5
    ) -> List[Image.Image]:
        """Extracts representative frames from a sliding window in a video."""
        start_frame = window_idx * stride
        end_frame = start_frame + window_size
        
        frame_indices = np.linspace(start_frame, end_frame - 1, num_samples, dtype=int)
        
        video = mp.VideoFileClip(video_path)
        frames = []
        
        for idx in frame_indices:
            time = idx / fps
            if time < video.duration:
                frame = video.get_frame(time)
                frames.append(Image.fromarray(frame))
        
        video.close()
        return frames

    @torch.no_grad()
    def generate_caption(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """Generates a caption for a single image."""
        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        else:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            early_stopping=True
        )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption
