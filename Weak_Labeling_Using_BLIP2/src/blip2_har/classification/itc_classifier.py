import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import Blip2ForImageTextRetrieval, AutoProcessor
from tqdm import tqdm

class ITCClassifier:
    """
    Classifies BLIP-2 Q-Former features against text prototypes in the 
    shared 256-D ITC (Image-Text Contrastive) space.
    """
    def __init__(
        self, 
        model_name: str = "Salesforce/blip2-itm-vit-g", 
        device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Loading ITC retrieval model '{model_name}' on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def build_text_prototypes(
        self, 
        class_prompts: Dict[str, List[str]]
    ) -> Tuple[List[str], torch.Tensor]:
        """Builds normalized 256-D text prototypes for each activity class."""
        class_names: List[str] = []
        proto_list: List[torch.Tensor] = []

        for cls_name, prompts in tqdm(class_prompts.items(), desc="Building prototypes"):
            if not prompts:
                continue
            class_names.append(cls_name)

            # Replicating the ITC text path directly
            inputs = self.processor.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(self.device)

            text_embeds = self.model.embeddings(input_ids=inputs.input_ids)
            text_outputs = self.model.qformer(
                query_embeds=text_embeds,
                query_length=0,
                attention_mask=inputs.attention_mask,
                return_dict=True,
            )

            # CLS token output
            cls_output = text_outputs.last_hidden_state[:, 0, :]
            
            # Project to shared 256-D space
            text_feat = self.model.text_projection(cls_output.to(self.model.text_projection.weight.dtype))
            text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
            
            # Average across prompts
            proto = text_feat.float().mean(dim=0)
            proto = torch.nn.functional.normalize(proto, dim=-1)
            proto_list.append(proto)

        prototypes = torch.stack(proto_list, dim=0)
        return class_names, prototypes

    @torch.no_grad()
    def project_vision_features(self, centroids: np.ndarray) -> torch.Tensor:
        """Projects 768-D vision centroids into the shared 256-D ITC space."""
        feats = torch.from_numpy(centroids).to(self.device).to(self.model.vision_projection.weight.dtype)
        projected = self.model.vision_projection(feats)
        projected = torch.nn.functional.normalize(projected.float(), dim=-1)
        return projected

    @torch.no_grad()
    def classify(
        self, 
        vision_feats: torch.Tensor, 
        text_prototypes: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cosine-similarity classification in the shared ITC space."""
        logits = 100.0 * vision_feats @ text_prototypes.T
        probs = logits.softmax(dim=-1)
        top_probs, top_idx = probs.topk(1, dim=-1)
        return top_idx.cpu().numpy(), top_probs.cpu().numpy(), probs.cpu().numpy()
