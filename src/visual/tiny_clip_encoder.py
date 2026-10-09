"""
TinyCLIP-8M Visual Encoder Implementation.
Provides zero-shot visual and textual embeddings in a joint 512-dimensional embedding space.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import gc
import numpy as np
from PIL import Image
import torch

from src.visual.base_visual_encoder import (
    BaseVisualEncoder,
    ModelLoadError,
    PreprocessingError,
    InferenceError,
)

DEFAULT_TINYCLIP_CHECKPOINT = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"


class TinyCLIPEncoder(BaseVisualEncoder):
    """
    TinyCLIP-8M visual encoder using Hugging Face transformers.

    Produces 512-dimensional embeddings for images and text in a shared multimodal space.
    Enables direct zero-shot cosine similarity calculation between screenshot images
    and canonical category prompt descriptions.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TINYCLIP_CHECKPOINT,
        device: str = "cpu",
        embedding_dim: int = 512
    ):
        super().__init__(model_name=model_name, device=device)
        self._embedding_dim = embedding_dim
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def load_model(self) -> None:
        """
        Lazily loads TinyCLIP CLIPModel and AutoProcessor.
        """
        if self._is_loaded and self._model is not None and self._processor is not None:
            return

        try:
            from transformers import CLIPModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = CLIPModel.from_pretrained(self._model_name)
            self._model.to(self._device)
            self._model.eval()

            # Confirm embedding dimension from model config if available
            if hasattr(self._model.config, "projection_dim"):
                self._embedding_dim = self._model.config.projection_dim
            elif hasattr(self._model.config, "text_config") and hasattr(self._model.config.text_config, "projection_dim"):
                self._embedding_dim = self._model.config.text_config.projection_dim

            self._is_loaded = True

        except Exception as e:
            self._is_loaded = False
            self._model = None
            self._processor = None
            raise ModelLoadError(
                f"Failed to load TinyCLIP model from '{self._model_name}': {e}. "
                "Ensure internet connectivity or pre-downloaded weights."
            ) from e

    def unload_model(self) -> None:
        """Frees TinyCLIP model weights and processor from memory."""
        self._model = None
        self._processor = None
        super().unload_model()

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Preprocesses image into pixel_values tensor expected by TinyCLIP.
        """
        if not self._is_loaded:
            self.load_model()

        pil_img = self._load_pil_image(image_input)
        try:
            inputs = self._processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            return inputs
        except Exception as e:
            raise PreprocessingError(f"TinyCLIP image preprocessing failed: {e}") from e

    def get_image_embedding(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extracts 1D (512,) float32 image embedding vector.
        """
        if not self._is_loaded:
            self.load_model()

        inputs = self.preprocess_image(image_input)
        try:
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    if hasattr(features, "image_embeds") and features.image_embeds is not None:
                        features = features.image_embeds
                    elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                        features = features.pooler_output
                    else:
                        features = features[0]
                # Convert to float32 numpy
                emb = features.detach().cpu().numpy().squeeze(0).astype(np.float32)

            if not np.all(np.isfinite(emb)):
                raise InferenceError("TinyCLIP produced non-finite (NaN/Inf) image embedding values.")

            if normalize:
                emb = self._l2_normalize(emb)

            return emb
        except Exception as e:
            if isinstance(e, (PreprocessingError, InferenceError)):
                raise
            raise InferenceError(f"Error during TinyCLIP image embedding inference: {e}") from e

    def get_text_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extracts 1D (512,) float32 text embedding vector.
        """
        if not self._is_loaded:
            self.load_model()

        if not text or not text.strip():
            raise PreprocessingError("Empty or whitespace-only text provided for text embedding.")

        try:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self._model.get_text_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    if hasattr(features, "text_embeds") and features.text_embeds is not None:
                        features = features.text_embeds
                    elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                        features = features.pooler_output
                    else:
                        features = features[0]
                emb = features.detach().cpu().numpy().squeeze(0).astype(np.float32)

            if not np.all(np.isfinite(emb)):
                raise InferenceError("TinyCLIP produced non-finite (NaN/Inf) text embedding values.")

            if normalize:
                emb = self._l2_normalize(emb)

            return emb
        except Exception as e:
            if isinstance(e, (PreprocessingError, InferenceError)):
                raise
            raise InferenceError(f"Error during TinyCLIP text embedding inference: {e}") from e

    def get_text_embeddings(
        self,
        text_list: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extracts 2D (M, 512) float32 text embedding matrix for a list of text prompts.
        """
        if not self._is_loaded:
            self.load_model()

        if not text_list:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        try:
            inputs = self._processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self._model.get_text_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    if hasattr(features, "text_embeds") and features.text_embeds is not None:
                        features = features.text_embeds
                    elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                        features = features.pooler_output
                    else:
                        features = features[0]
                matrix = features.detach().cpu().numpy().astype(np.float32)

            if not np.all(np.isfinite(matrix)):
                raise InferenceError("TinyCLIP produced non-finite text embeddings.")

            if normalize:
                matrix = self._l2_normalize(matrix, axis=1)

            return matrix
        except Exception as e:
            if isinstance(e, (PreprocessingError, InferenceError)):
                raise
            raise InferenceError(f"Error computing batch text embeddings: {e}") from e

    def compute_similarity(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        text_list: List[str]
    ) -> np.ndarray:
        """
        Computes zero-shot cosine similarities between a screenshot and a list of text prompts.
        Returns 1D array of floats representing similarity scores.
        """
        img_emb = self.get_image_embedding(image_input, normalize=True)  # (512,)
        txt_embs = self.get_text_embeddings(text_list, normalize=True)   # (M, 512)

        # Dot product of normalized vectors = Cosine Similarity
        similarities = np.dot(txt_embs, img_emb)
        return similarities.astype(np.float32)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        if self._is_loaded and self._model is not None:
            total_params = sum(p.numel() for p in self._model.parameters())
            info["num_parameters"] = total_params
            info["model_size_mb"] = round(total_params * 4 / (1024 * 1024), 2)  # fp32 estimate
        return info
