"""
Integration Tests for OCR Pipeline + Embedding Layer
====================================================
Verifies that real processed data from the screenshot dataset flows end-to-end
from metadata loading -> mock OCR -> chunking -> SentenceTransformer embedding.
"""

import unittest
from pathlib import Path

from src.pipeline import OCRPipeline
from src.embedder import SentenceTransformerEmbedder


class TestPipelineEmbeddingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = OCRPipeline()
        cls.embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    def test_end_to_end_embedding_pipeline(self):
        """
        Tests end-to-end processing of sample dataset and verifies embedding properties.
        """
        # Step 1: Run pipeline on actual sample metadata
        processed_docs = self.pipeline.process(metadata_path="sample/metadata.csv")
        self.assertGreater(len(processed_docs), 0)

        # Step 2: Collect non-empty chunks and trace back to screenshot filenames
        all_chunks = []
        chunk_sources = []
        for doc in processed_docs:
            filename = Path(doc["image_path"]).name
            for chunk in doc["chunks"]:
                cleaned = chunk.strip()
                if cleaned:
                    all_chunks.append(cleaned)
                    chunk_sources.append(filename)

        self.assertGreater(len(all_chunks), 0, "Pipeline produced no non-empty chunks.")

        # Step 3: Generate embeddings
        embeddings = self.embedder.embed(all_chunks)

        # Step 4: Verify chunk count equals embedding count
        self.assertEqual(
            len(embeddings),
            len(all_chunks),
            "Number of embeddings must match number of chunks.",
        )

        # Step 5: Verify embedding dimensionality > 0 and consistent across all chunks
        expected_dim = len(embeddings[0])
        self.assertGreater(expected_dim, 0, "Embedding dimension must be greater than 0.")

        for idx, emb in enumerate(embeddings):
            self.assertEqual(
                len(emb),
                expected_dim,
                f"Embedding at index {idx} has dimension {len(emb)}, expected {expected_dim}.",
            )

        # Step 6: Verify traceability mapping exists for every embedding
        self.assertEqual(len(chunk_sources), len(embeddings))


if __name__ == "__main__":
    unittest.main()
