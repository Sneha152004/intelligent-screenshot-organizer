"""
EasyOCR Engine Module
=====================
Production implementation of BaseOCREngine using EasyOCR for deep-learning-based
text extraction from screenshot images.
"""

from pathlib import Path
from typing import List, Optional

from .mock_ocr import BaseOCREngine, MockOCREngine


class EasyOCREngine(BaseOCREngine):
    """
    Production OCR engine leveraging EasyOCR (PyTorch-based CRAFT + ResNet/LSTM).
    Implements lazy initialization of the EasyOCR Reader instance to avoid premature
    model downloads or memory allocations during module imports and testing.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize EasyOCR Engine parameters.

        Args:
            languages: List of language codes to recognize (default: ['en']).
            gpu: Whether to enable GPU acceleration (default: False for CPU compatibility).
            verbose: Whether EasyOCR should print verbose initialization/processing logs.
        """
        self.languages = languages if languages is not None else ["en"]
        self.gpu = gpu
        self.verbose = verbose
        self._reader = None

    @property
    def reader(self):
        """
        Lazily initialize and return the cached EasyOCR Reader instance.
        """
        if self._reader is None:
            try:
                import easyocr
            except ImportError as e:
                raise ImportError(
                    "EasyOCR package is not installed. Please install 'easyocr' to use EasyOCREngine."
                ) from e
            try:
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    verbose=self.verbose,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize EasyOCR Reader with languages={self.languages}, gpu={self.gpu}: {e}"
                ) from e
        return self._reader

    def extract_text(self, image_path: str) -> str:
        """
        Extract text content from an image using EasyOCR.

        Args:
            image_path: Absolute or relative path to target image file.

        Returns:
            Extracted text content as a newline-delimited string.

        Raises:
            FileNotFoundError: If the specified image_path does not exist.
            RuntimeError: If EasyOCR processing encounters an error.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Screenshot image file not found: {image_path}")

        try:
            results = self.reader.readtext(str(path), detail=0)
            return "\n".join(results)
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ImportError)):
                raise
            raise RuntimeError(
                f"Error extracting OCR text from '{image_path}' using EasyOCR: {e}"
            ) from e


def get_ocr_engine(engine_type: str = "mock", **kwargs) -> BaseOCREngine:
    """
    Factory function to instantiate and return a swappable BaseOCREngine.

    Args:
        engine_type: Type identifier ('mock' or 'easyocr').
        **kwargs: Optional keyword arguments passed to the engine constructor.

    Returns:
        Instance of BaseOCREngine (MockOCREngine or EasyOCREngine).

    Raises:
        ValueError: If engine_type is unrecognized.
    """
    engine_type_clean = engine_type.strip().lower()
    if engine_type_clean == "mock":
        return MockOCREngine(**kwargs)
    elif engine_type_clean in ("easyocr", "easy_ocr"):
        return EasyOCREngine(**kwargs)
    else:
        raise ValueError(
            f"Unsupported OCR engine type: '{engine_type}'. Supported engines: 'mock', 'easyocr'."
        )
