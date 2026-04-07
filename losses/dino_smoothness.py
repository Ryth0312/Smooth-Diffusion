import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import List, Union, Optional, Dict
import numpy as np
from transformers import AutoModel
import os

class DinoFeatureExtractor:
    def __init__(
        self, 
        model_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m', 
        #model_name: str = "facebook/dinov2-base",
        device: str = 'cuda',
        layers: Optional[List[int]] = None
    ):
        self.device = device
        self.model_name = model_name
        self.layers = layers

        try:
            self.model = AutoModel.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
            self.model.to(device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")
        
        # Image preprocessing for ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Feature Extractor initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Layers: {layers}")
    
    def preprocess_images(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: List of PIL Images or tensor [B, 3, H, W] in [0, 1] or [-1, 1]
        Returns:
            Preprocessed tensor [B, 3, 224, 224] with ImageNet normalization
        """
        if isinstance(images, torch.Tensor):
            # If tensor, assume it's in [-1, 1] or [0, 1]
            # Convert to [0, 1] if needed
            if images.min() < 0:
                images = (images + 1) / 2  # [-1, 1] -> [0, 1]
            
            # Resize to 224x224 and apply ImageNet normalization
            images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)
            # Apply normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
            images = (images - mean) / std
            return images
        else:
            # List of PIL Images
            batch = []
            for img in images:
                img_tensor = self.preprocess(img)
                batch.append(img_tensor)
            return torch.stack(batch).to(self.device)
    
    def extract_features(
        self, 
        images: Union[List[Image.Image], torch.Tensor],
        pool: str = 'mean',
        l2norm: bool = True
    ) -> torch.Tensor:
        """
        Args:
            images: List of PIL Images or tensor [B, 3, H, W]
            pool: Pooling method ('mean', 'cls', or 'none')
                  - 'mean': Average pool over all tokens (recommended)
                  - 'cls': Use only CLS token
                  - 'none': Keep all tokens [B, N_tokens, D]
            l2norm: Whether to L2-normalize features
        
        Returns:
            Features tensor [B, D] if pooled, [B, N_tokens, D] if not pooled
        """
        # Preprocess
        x = self.preprocess_images(images)
        
        # NOTE: No torch.no_grad() here! We need gradients to flow to images for optimization
        # The DINO model parameters are frozen (not in optimizer), so they won't be updated

        outputs = self.model(pixel_values=x)
        features = outputs.last_hidden_state  # [B, 1+N, D]
        
        # Pooling
        if pool == 'mean':
            # Average over all tokens
            features = features[:, 1:, :].mean(dim=1)  # [B, D]
        elif pool == 'cls':
            # Use only CLS token (first token)
            features = features[:, 0, :]  # [B, D]
        elif pool == 'none':
            # Keep all tokens
            pass  # [B, N_tokens, D]
        else:
            raise ValueError(f"Invalid pool method: {pool}")
        
        # L2 normalization
        if l2norm and len(features.shape) == 2:
            features = F.normalize(features, p=2, dim=1)
        elif l2norm and len(features.shape) == 3:
            features = F.normalize(features, p=2, dim=2)
        
        return features


def path_feature_smoothness_loss(
    features: torch.Tensor,
    lam_smooth: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Formula of Velocity (first-order smoothness): mean(||f[t+1] - f[t]||^2)
    
    Args:
        features: Feature tensor [T, D] where T is number of frames
        lam_smooth: Weight for velocity loss
    
    Returns:
        Dictionary with 'total' and 'velocity' losses
    """
    # First-order differences (velocity)
    # f[t+1] - f[t] for t in [0, T-2]
    vel = features[1:] - features[:-1]  # [T-1, D]
    vel_squared = (vel ** 2).mean()  # Mean over all elements (normalize by D)
    
    # Total loss
    total = lam_smooth * vel_squared
    
    return {
        'total': total,
        'velocity': vel_squared,
    }


def dino_smoothness_loss(
    path_images: Union[List[Image.Image], torch.Tensor],
    dino_extractor: DinoFeatureExtractor,
    lam_smooth: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute DINOv3 feature-based smoothness loss for an image sequence)
    
    Args:
        path_images: Sequence of images [T, 3, H, W] or List of T PIL Images
        lam_smooth: Weight for velocity loss
    
    Returns:
        Dictionary with 'velocity' losses
        Use loss['total'].backward() in optimization
    """
    # Extract features
    features = dino_extractor.extract_features(path_images, pool='none', l2norm=False)
    
    # Compute path smoothness
    losses = path_feature_smoothness_loss(features, lam_smooth=lam_smooth)
    return losses


def compute_dino_metrics(
    images: Union[List[Image.Image], torch.Tensor],
    dino_extractor: DinoFeatureExtractor
) -> Dict[str, float]:
    """
    Compute DINOv3 feature-based metrics for a saved image sequence (offline evaluation).
    
    Args:
        images: Sequence of images [T, 3, H, W] or List of T PIL Images
    Returns:
        Dictionary with metric values:
        - 'dino_vel_mean': Mean velocity (smoothness)
    """
    with torch.no_grad():
        # Extract features
        features = dino_extractor.extract_features(images, pool='none', l2norm=False)
        # Compute losses
        losses = path_feature_smoothness_loss(features, lam_smooth=1.0)
    
    return {
        'dino_vel_mean': losses['velocity'].item(),
    }