import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

class CLIPClassifier:
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-large-patch14", 
        device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Loading CLIP model '{model_name}' on {device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def build_text_prototypes(
        self, 
        class_prompts: Dict[str, List[str]]
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Builds normalized text prototypes (embeddings) for each class.
        
        Args:
            class_prompts: Dictionary mapping class names to lists of prompts.
            
        Returns:
            class_names: List of class names.
            prototypes: Tensor of shape (num_classes, embedding_dim).
        """
        class_names: List[str] = []
        proto_list: List[torch.Tensor] = []

        for cls_name, prompts in class_prompts.items():
            if not prompts:
                continue

            class_names.append(cls_name)

            # Tokenizing and embedding prompts
            inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            outputs = self.model.get_text_features(**inputs)
            
            # Handle different output formats
            if hasattr(outputs, "text_embeds"):
                text_features = outputs.text_embeds
            elif hasattr(outputs, "pooler_output"):
                text_features = outputs.pooler_output
            else:
                text_features = outputs

            # Normalizing and averaging
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
            proto = text_features.mean(dim=0)
            proto = proto / (proto.norm() + 1e-8)
            proto_list.append(proto)

        prototypes = torch.stack(proto_list, dim=0)
        return class_names, prototypes

    @torch.no_grad()
    def classify_embeddings(
        self, 
        embeddings: torch.Tensor, 
        prototypes: torch.Tensor, 
        topk: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classifies input embeddings against text prototypes.
        
        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim).
            prototypes: Tensor of shape (num_classes, embedding_dim).
            topk: Number of top classes to return.
            
        Returns:
            top_idx: Indices of top-k classes.
            top_probs: Probabilities of top-k classes.
            all_probs: Probabilities for all classes.
        """
        # Ensure normalization
        embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        
        logits = 100.0 * embeddings @ prototypes.T
        probs = logits.softmax(dim=-1)
        top_probs, top_idx = probs.topk(topk, dim=-1)

        return (
            top_idx.cpu().numpy(), 
            top_probs.cpu().numpy(), 
            probs.cpu().numpy()
        )
