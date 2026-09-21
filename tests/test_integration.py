"""
Integration Tests for 3-Layer Storage Architecture Pipeline
============================================================
Verifies end-to-end dataset flow:
Screenshot File Storage (Layer 1)
  -> Structured Metadata Storage (Layer 2)
  -> Mock OCR & Text Chunking
  -> SentenceTransformer Embeddings
  -> ChromaDB Vector Index (Layer 3)
  -> Semantic Query & screenshot_id Resolution.
"""

import os
import shutil
import tempfile
import unittest

from src.config import AppConfig
from src.embedder import SentenceTransformerEmbedder
from src.pipeline import OCRPipeline
from src.storage.file_store import LocalFileStore
from src.storage.metadata_store import SQLiteMetadataStore
from src.vector_store import VectorStoreManager


class TestThreeLayerArchitectureIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_3layer_")
        self.screenshots_dir = os.path.join(self.temp_dir, "screenshots")
        self.metadata_db_path = os.path.join(self.temp_dir, "metadata.sqlite")
        self.chroma_db_dir = os.path.join(self.temp_dir, "chroma_db")

        self.file_store = LocalFileStore(storage_dir=self.screenshots_dir)
        self.metadata_store = SQLiteMetadataStore(db_path=self.metadata_db_path)
        self.vector_store = VectorStoreManager(db_dir=self.chroma_db_dir, collection_name="test_collection")
        self.embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

        self.pipeline = OCRPipeline(
            file_store=self.file_store,
            metadata_store=self.metadata_store,
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_three_layer_pipeline(self):
        """
        Tests end-to-end processing of sample dataset through all 3 storage layers.
        """
        # Step 1: Process dataset
        processed_docs = self.pipeline.process(metadata_path="sample/metadata.csv")
        self.assertGreater(len(processed_docs), 0)

        # Verify Layer 1 & Layer 2 registration
        all_metadata = self.metadata_store.list_all_metadata()
        self.assertEqual(len(all_metadata), len(processed_docs))

        # Step 2: Index chunks in Layer 3 (ChromaDB)
        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []

        for doc in processed_docs:
            sid = doc["screenshot_id"]
            cat = doc["metadata"]["category"]
            for idx, chunk in enumerate(doc["chunks"]):
                cleaned = chunk.strip()
                if cleaned:
                    chunk_ids.append(f"{sid}_chunk_{idx}")
                    chunk_texts.append(cleaned)
                    chunk_metadatas.append({
                        "screenshot_id": sid,
                        "chunk_index": idx,
                        "category": cat,
                    })

        embeddings = self.embedder.embed(chunk_texts)
        self.vector_store.add_documents(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
        )

        self.assertEqual(self.vector_store.count_documents(), len(chunk_texts))

        # Step 3: Perform 2-stage retrieval (Vector Search + Metadata Resolution)
        query = "wifi password"
        results = self.pipeline.search_and_enrich(
            query_text=query,
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=3,
        )

        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match["screenshot_id"], "ss_001")
        self.assertEqual(top_match["filename"], "ss_001.jpg")
        self.assertEqual(top_match["category"], "Connectivity")
        self.assertTrue(os.path.exists(top_match["image_uri"]))


if __name__ == "__main__":
    unittest.main()
