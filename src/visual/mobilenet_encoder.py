"""
MobileNetV2 Visual Encoder Implementation.
Extracts 1280-dimensional feature vector embeddings from standard torchvision MobileNetV2 backbone.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from src.visual.base_visual_encoder import (
    BaseVisualEncoder,
    ModelLoadError,
    PreprocessingError,
    InferenceError,
)

DEFAULT_MOBILENET_NAME = "mobilenet_v2_default"


class MobileNetV2Encoder(BaseVisualEncoder):
    """
    MobileNetV2 visual feature extractor using torchvision.

    Exposes 1280-dimensional feature vector embeddings (post-pooling).
    Does NOT output ImageNet category probabilities. A visual prototype
    or classifier head can consume these 1280-dim feature vectors.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MOBILENET_NAME,
        device: str = "cpu",
        embedding_dim: int = 1280
    ):
        super().__init__(model_name=model_name, device=device)
        self._embedding_dim = embedding_dim
        self._model: Optional[nn.Module] = None
        self._transform: Optional[Any] = None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def load_model(self) -> None:
        """
        Lazily loads torchvision MobileNetV2 model and weights.
        """
        if self._is_loaded and self._model is not None and self._transform is not None:
            return

        try:
            weights = MobileNet_V2_Weights.DEFAULT
            base_model = mobilenet_v2(weights=weights)

            # Replace classifier with Identity to expose 1280-dim features after pooling
            base_model.classifier = nn.Identity()

            base_model.to(self._device)
            base_model.eval()

            self._model = base_model
            self._transform = weights.transforms()
            self._is_loaded = True

        except Exception as e:
            self._is_loaded = False
            self._model = None
            self._transform = None
            raise ModelLoadError(
                f"Failed to load MobileNetV2 torchvision model: {e}"
            ) from e

    def unload_model(self) -> None:
        """Frees MobileNetV2 model weights from memory."""
        self._model = None
        self._transform = None
        super().unload_model()

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocesses image into PyTorch tensor ready for MobileNetV2 forward pass.
        Returns tensor of shape (1, 3, 224, 224).
        """
        if not self._is_loaded:
            self.load_model()

        pil_img = self._load_pil_image(image_input)
        try:
            tensor_img = self._transform(pil_img)
            tensor_img = tensor_img.unsqueeze(0).to(self._device)  # (1, 3, 224, 224)
            return tensor_img
        except Exception as e:
            raise PreprocessingError(f"MobileNetV2 image preprocessing failed: {e}") from e

    def get_image_embedding(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extracts 1D (1280,) float32 feature embedding vector.
        """
        if not self._is_loaded:
            self.load_model()

        tensor_img = self.preprocess_image(image_input)
        try:
            with torch.no_grad():
                features = self._model(tensor_img)  # (1, 1280)
                emb = features.detach().cpu().numpy().squeeze(0).astype(np.float32)

            if not np.all(np.isfinite(emb)):
                raise InferenceError("MobileNetV2 produced non-finite (NaN/Inf) feature values.")

            if normalize:
                emb = self._l2_normalize(emb)

            return emb
        except Exception as e:
            if isinstance(e, (PreprocessingError, InferenceError)):
                raise
            raise InferenceError(f"Error during MobileNetV2 feature extraction inference: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        if self._is_loaded and self._model is not None:
            total_params = sum(p.numel() for p in self._model.parameters())
            info["num_parameters"] = total_params
            info["model_size_mb"] = round(total_params * 4 / (1024 * 1024), 2)
        return info
