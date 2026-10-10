"""
Unit tests and smoke tests for Visual Intelligence Branch encoders (BaseVisualEncoder, TinyCLIPEncoder, MobileNetV2Encoder).
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import torch

from src.visual.base_visual_encoder import (
    BaseVisualEncoder,
    VisualEncoderError,
    ModelLoadError,
    PreprocessingError,
    InferenceError,
)
from src.visual.tiny_clip_encoder import TinyCLIPEncoder
from src.visual.mobilenet_encoder import MobileNetV2Encoder


class DummyEncoder(BaseVisualEncoder):
    """Concrete dummy class for testing BaseVisualEncoder methods."""

    def __init__(self, model_name: str = "dummy", device: str = "cpu", dim: int = 128):
        super().__init__(model_name=model_name, device=device)
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def load_model(self) -> None:
        self._is_loaded = True

    def preprocess_image(self, image_input):
        return self._load_pil_image(image_input)

    def get_image_embedding(self, image_input, normalize: bool = True) -> np.ndarray:
        if not self._is_loaded:
            self.load_model()
        self._load_pil_image(image_input)
        vec = np.ones(self._dim, dtype=np.float32)
        if normalize:
            vec = self._l2_normalize(vec)
        return vec


class TestBaseVisualEncoder(unittest.TestCase):
    """Tests for BaseVisualEncoder abstract interface and helper methods."""

    def test_lazy_loading_initial_state(self):
        encoder = DummyEncoder(model_name="dummy_test", device="cpu")
        self.assertFalse(encoder.is_loaded)
        self.assertEqual(encoder.model_name, "dummy_test")
        self.assertEqual(encoder.device, "cpu")
        self.assertEqual(encoder.embedding_dim, 128)

    def test_load_and_unload_model(self):
        encoder = DummyEncoder()
        encoder.load_model()
        self.assertTrue(encoder.is_loaded)
        encoder.unload_model()
        self.assertFalse(encoder.is_loaded)

    def test_load_pil_image_valid_inputs(self):
        encoder = DummyEncoder()

        # PIL Image input
        img = Image.new("RGB", (100, 100), color="red")
        res1 = encoder._load_pil_image(img)
        self.assertIsInstance(res1, Image.Image)
        self.assertEqual(res1.size, (100, 100))

        # Numpy HWC array input
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        res2 = encoder._load_pil_image(arr)
        self.assertIsInstance(res2, Image.Image)
        self.assertEqual(res2.size, (50, 50))

        # PyTorch CHW tensor input
        tensor = torch.zeros((3, 60, 60), dtype=torch.uint8)
        res3 = encoder._load_pil_image(tensor)
        self.assertIsInstance(res3, Image.Image)
        self.assertEqual(res3.size, (60, 60))

    def test_load_pil_image_invalid_path(self):
        encoder = DummyEncoder()
        non_existent_path = Path(tempfile.gettempdir()) / "non_existent_ss_12345.png"
        with self.assertRaises(PreprocessingError):
            encoder._load_pil_image(non_existent_path)

    def test_load_pil_image_unsupported_type(self):
        encoder = DummyEncoder()
        with self.assertRaises(PreprocessingError):
            encoder._load_pil_image(12345)  # int is not a valid image input

    def test_l2_normalize(self):
        encoder = DummyEncoder()
        v = np.array([3.0, 4.0], dtype=np.float32)
        norm_v = encoder._l2_normalize(v)
        self.assertAlmostEqual(np.linalg.norm(norm_v), 1.0, places=5)
        self.assertTrue(np.allclose(norm_v, np.array([0.6, 0.8], dtype=np.float32)))

    def test_batch_embeddings(self):
        encoder = DummyEncoder()
        imgs = [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]
        matrix = encoder.get_image_embeddings(imgs, normalize=True)
        self.assertEqual(matrix.shape, (2, 128))
        self.assertEqual(matrix.dtype, np.float32)


class TestTinyCLIPEncoderMock(unittest.TestCase):
    """Unit tests for TinyCLIPEncoder using mocks."""

    def test_lazy_loading(self):
        encoder = TinyCLIPEncoder(model_name="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
        self.assertFalse(encoder.is_loaded)
        self.assertEqual(encoder.embedding_dim, 512)

    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.CLIPModel.from_pretrained")
    def test_successful_load_and_inference(self, mock_clip, mock_proc):
        mock_model_instance = MagicMock()
        mock_model_instance.config.projection_dim = 512

        # Mock get_image_features output
        mock_tensor = torch.ones((1, 512), dtype=torch.float32)
        mock_model_instance.get_image_features.return_value = mock_tensor
        mock_clip.return_value = mock_model_instance

        mock_proc_instance = MagicMock()
        mock_proc_instance.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}
        mock_proc.return_value = mock_proc_instance

        encoder = TinyCLIPEncoder(model_name="mock_tiny_clip")
        img = Image.new("RGB", (224, 224))

        emb = encoder.get_image_embedding(img, normalize=True)

        self.assertTrue(encoder.is_loaded)
        self.assertEqual(emb.shape, (512,))
        self.assertEqual(emb.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(emb)))
        self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=5)

    @patch("transformers.CLIPModel.from_pretrained", side_effect=Exception("Network error download failed"))
    def test_model_load_failure(self, mock_clip):
        encoder = TinyCLIPEncoder(model_name="invalid_model_path")
        with self.assertRaises(ModelLoadError):
            encoder.load_model()
        self.assertFalse(encoder.is_loaded)


class TestMobileNetV2EncoderMock(unittest.TestCase):
    """Unit tests for MobileNetV2Encoder using mocks."""

    def test_lazy_loading(self):
        encoder = MobileNetV2Encoder()
        self.assertFalse(encoder.is_loaded)
        self.assertEqual(encoder.embedding_dim, 1280)

    @patch("src.visual.mobilenet_encoder.mobilenet_v2")
    def test_successful_inference(self, mock_mobilenet):
        mock_model = MagicMock()
        mock_model.return_value = torch.ones((1, 1280), dtype=torch.float32)
        mock_mobilenet.return_value = mock_model

        encoder = MobileNetV2Encoder()
        img = Image.new("RGB", (224, 224))

        emb = encoder.get_image_embedding(img, normalize=True)
        self.assertTrue(encoder.is_loaded)
        self.assertEqual(emb.shape, (1280,))
        self.assertEqual(emb.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(emb)))
        self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=5)

    @patch("src.visual.mobilenet_encoder.mobilenet_v2", side_effect=Exception("Weights download failed"))
    def test_load_failure(self, mock_mobilenet):
        encoder = MobileNetV2Encoder()
        with self.assertRaises(ModelLoadError):
            encoder.load_model()
        self.assertFalse(encoder.is_loaded)


class TestVisualEncodersSmoke(unittest.TestCase):
    """Real weights end-to-end smoke tests on synthetic images (CPU)."""

    def test_mobilenet_real_weights_smoke(self):
        encoder = MobileNetV2Encoder(device="cpu")
        img = Image.new("RGB", (224, 224), color="blue")

        # Test lazy load
        self.assertFalse(encoder.is_loaded)
        emb1 = encoder.get_image_embedding(img, normalize=True)
        self.assertTrue(encoder.is_loaded)

        # Dimension & finiteness
        self.assertEqual(emb1.shape, (1280,))
        self.assertEqual(emb1.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(emb1)))
        self.assertAlmostEqual(np.linalg.norm(emb1), 1.0, places=4)

        # Repeated inference determinism check
        emb2 = encoder.get_image_embedding(img, normalize=True)
        self.assertTrue(np.allclose(emb1, emb2, atol=1e-5))

        # Model metadata
        info = encoder.get_model_info()
        self.assertEqual(info["encoder_type"], "MobileNetV2Encoder")
        self.assertEqual(info["embedding_dim"], 1280)
        self.assertTrue(info["num_parameters"] > 0)

        # Clean up
        encoder.unload_model()
        self.assertFalse(encoder.is_loaded)

    def test_tiny_clip_real_weights_smoke(self):
        encoder = TinyCLIPEncoder(model_name="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", device="cpu")
        img = Image.new("RGB", (224, 224), color="green")

        # Test lazy load
        self.assertFalse(encoder.is_loaded)
        img_emb1 = encoder.get_image_embedding(img, normalize=True)
        self.assertTrue(encoder.is_loaded)

        # Image Embedding shape & finiteness
        self.assertEqual(img_emb1.shape, (512,))
        self.assertEqual(img_emb1.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(img_emb1)))
        self.assertAlmostEqual(np.linalg.norm(img_emb1), 1.0, places=4)

        # Text Embedding shape & finiteness
        txt_emb = encoder.get_text_embedding("a screenshot of a payment receipt", normalize=True)
        self.assertEqual(txt_emb.shape, (512,))
        self.assertEqual(txt_emb.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(txt_emb)))
        self.assertAlmostEqual(np.linalg.norm(txt_emb), 1.0, places=4)

        # Joint space zero-shot similarity calculation
        prompts = [
            "a screenshot of a payment receipt",
            "a screenshot of a programming code editor",
            "a screenshot of a chat conversation"
        ]
        sims = encoder.compute_similarity(img, prompts)
        self.assertEqual(sims.shape, (3,))
        self.assertTrue(np.all(np.isfinite(sims)))

        # Repeated inference determinism check
        img_emb2 = encoder.get_image_embedding(img, normalize=True)
        self.assertTrue(np.allclose(img_emb1, img_emb2, atol=1e-5))

        # Model metadata
        info = encoder.get_model_info()
        self.assertEqual(info["encoder_type"], "TinyCLIPEncoder")
        self.assertEqual(info["embedding_dim"], 512)
        self.assertTrue(info["num_parameters"] > 0)

        # Clean up
        encoder.unload_model()
        self.assertFalse(encoder.is_loaded)


if __name__ == "__main__":
    unittest.main()
