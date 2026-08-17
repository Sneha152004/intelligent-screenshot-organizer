"""
Unit Tests for Embedder Module
==============================
Tests the BaseEmbedder interface and SentenceTransformerEmbedder implementation.
"""

import unittest
import numpy as np

from src.embedder import SentenceTransformerEmbedder, BaseEmbedder


class TestSentenceTransformerEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load embedder model once for all tests
        cls.embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    def test_single_input(self):
        """Test 1: One input text returns one non-empty vector."""
        texts = ["Airtel Wi-Fi password"]
        vectors = self.embedder.embed(texts)
        self.assertEqual(len(vectors), 1)
        self.assertGreater(len(vectors[0]), 0)
        self.assertIsInstance(vectors[0][0], float)

    def test_multiple_inputs(self):
        """Test 2: Multiple inputs return matching number of vectors with equal dimensionality."""
        texts = [
            "Airtel Wi-Fi password",
            "UPI payment of ₹689",
            "Deep Learning syllabus",
        ]
        vectors = self.embedder.embed(texts)
        self.assertEqual(len(vectors), 3)

        dim = len(vectors[0])
        self.assertGreater(dim, 0)
        for v in vectors:
            self.assertEqual(len(v), dim)

    def test_empty_input(self):
        """Test 3: Empty list input returns an empty list."""
        vectors = self.embedder.embed([])
        self.assertEqual(vectors, [])

    def test_determinism(self):
        """Test 4: Embedding the same text twice produces numerically identical vectors."""
        text = "Deep Learning course syllabus and outcomes"
        vec1 = self.embedder.embed([text])[0]
        vec2 = self.embedder.embed([text])[0]

        self.assertEqual(len(vec1), len(vec2))
        np.testing.assert_allclose(vec1, vec2, rtol=1e-6, atol=1e-6)

    def test_invalid_input(self):
        """Test 5: Non-list input or non-string elements raise TypeError."""
        with self.assertRaises(TypeError):
            self.embedder.embed("Not a list")  # type: ignore

        with self.assertRaises(TypeError):
            self.embedder.embed([123, "valid string"])  # type: ignore

    def test_base_embedder_interface(self):
        """Verify subclassing BaseEmbedder works for alternative models."""
        class CustomEmbedder(BaseEmbedder):
            def embed(self, texts):
                return [[1.0, 2.0, 3.0] for _ in texts]

        custom = CustomEmbedder()
        res = custom.embed(["test1", "test2"])
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
