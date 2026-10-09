"""
Base Visual Encoder Interface for Screenshot Classification.
Defines abstract contract, common preprocessing helpers, and exception hierarchy.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import gc
import numpy as np
from PIL import Image
import torch


class VisualEncoderError(Exception):
    """Base exception for all visual encoder errors."""
    pass


class ModelLoadError(VisualEncoderError):
    """Raised when model weights or processor fail to load."""
    pass


class PreprocessingError(VisualEncoderError):
    """Raised when image preprocessing fails or input format is invalid."""
    pass


class InferenceError(VisualEncoderError):
    """Raised when forward pass or embedding extraction fails."""
    pass


class BaseVisualEncoder(ABC):
    """
    Abstract base class for all visual encoders in the Visual Intelligence Branch.

    Supports:
    - Lazy loading
    - Standardized image preprocessing
    - Single & batch embedding extraction
    - CPU/GPU device selection
    - Model metadata reporting
    - Graceful error handling
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self._model_name: str = model_name
        self._device: str = device
        self._is_loaded: bool = False

    @property
    def model_name(self) -> str:
        """Returns model identifier or weight path."""
        return self._model_name

    @property
    def device(self) -> str:
        """Returns current computation device (e.g., 'cpu')."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded into memory."""
        return self._is_loaded

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of feature embedding output vector."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Lazily loads model weights and preprocessor into memory.
        Must set self._is_loaded = True upon success.
        Raises ModelLoadError on failure.
        """
        pass

    def unload_model(self) -> None:
        """
        Unloads model weights from memory to reclaim resources.
        Sets self._is_loaded = False.
        """
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_pil_image(self, image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """
        Utility to convert various image input formats into a PIL RGB Image.
        Raises PreprocessingError on invalid input or file missing.
        """
        try:
            if isinstance(image_input, Image.Image):
                return image_input.convert("RGB")
            if isinstance(image_input, (str, Path)):
                path = Path(image_input)
                if not path.exists():
                    raise PreprocessingError(f"Image path does not exist: {path}")
                if not path.is_file():
                    raise PreprocessingError(f"Image path is not a file: {path}")
                try:
                    img = Image.open(path)
                    img.verify()  # verify integrity
                    img = Image.open(path)  # re-open after verify
                    return img.convert("RGB")
                except Exception as e:
                    raise PreprocessingError(f"Failed to open or decode image file '{path}': {e}") from e

            if isinstance(image_input, np.ndarray):
                if image_input.size == 0:
                    raise PreprocessingError("Provided numpy array image is empty.")
                if image_input.ndim == 2:  # Grayscale
                    return Image.fromarray(image_input).convert("RGB")
                elif image_input.ndim == 3:
                    if image_input.shape[2] in (1, 3, 4):
                        return Image.fromarray(image_input).convert("RGB")
                    elif image_input.shape[0] in (1, 3, 4):  # CHW to HWC
                        arr = np.transpose(image_input, (1, 2, 0))
                        return Image.fromarray(arr).convert("RGB")
                raise PreprocessingError(f"Unsupported numpy image shape: {image_input.shape}")

            if isinstance(image_input, torch.Tensor):
                if image_input.numel() == 0:
                    raise PreprocessingError("Provided PyTorch tensor image is empty.")
                arr = image_input.detach().cpu().numpy()
                return self._load_pil_image(arr)

            raise PreprocessingError(f"Unsupported image input type: {type(image_input)}")

        except PreprocessingError:
            raise
        except Exception as e:
            raise PreprocessingError(f"Error converting image input to PIL Image: {e}") from e

    @abstractmethod
    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]) -> Any:
        """
        Preprocesses an image input into tensor format expected by model.
        Must raise PreprocessingError on invalid input.
        """
        pass

    @abstractmethod
    def get_image_embedding(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extracts 1D embedding float32 vector for a single image.
        If normalize is True, vector is L2-normalized to unit length.
        Must raise InferenceError or PreprocessingError on failure.
        """
        pass

    def get_image_embeddings(
        self,
        image_inputs: List[Union[str, Path, Image.Image, np.ndarray, torch.Tensor]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extracts 2D embedding float32 matrix (N, embedding_dim) for a list of images.
        Default implementation iterates over single embeddings or batching.
        """
        if not image_inputs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = []
        for idx, img in enumerate(image_inputs):
            try:
                emb = self.get_image_embedding(img, normalize=normalize)
                embeddings.append(emb)
            except Exception as e:
                raise InferenceError(f"Failed to compute embedding for image at index {idx}: {e}") from e

        return np.vstack(embeddings).astype(np.float32)

    def _l2_normalize(self, v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
        """Normalizes numpy array along specified axis using L2 norm."""
        norm = np.linalg.norm(v, ord=2, axis=axis, keepdims=True)
        norm = np.maximum(norm, eps)
        return v / norm

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns model metadata, parameter counts, memory status, and setup details.
        """
        info = {
            "model_name": self._model_name,
            "encoder_type": self.__class__.__name__,
            "embedding_dim": self.embedding_dim,
            "is_loaded": self._is_loaded,
            "device": self._device,
            "num_parameters": None,
            "model_size_mb": None
        }
        return info
