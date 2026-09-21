"""
Unit & Integration Tests for ChromaDB Vector Store Module (Layer 3)
====================================================================
Tests collection creation, lightweight document & embedding insertion,
semantic similarity search, top-K retrieval, document count, collection deletion,
and cross-session persistence without requiring raw image paths in ChromaDB.
"""

import os
import shutil
import tempfile
import unittest

from src.config import config
from src.embedder import SentenceTransformerEmbedder
from src.vector_store import VectorStoreManager


class TestVectorStoreManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for isolated database tests
        self.temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
        self.collection_name = "test_screenshot_chunks"
        self.store = VectorStoreManager(
            db_dir=self.temp_dir,
            collection_name=self.collection_name,
        )
        self.embedder = SentenceTransformerEmbedder(model_name=config.embedding_model)

    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_collection_creation(self):
        """Test 1: ChromaDB collection creation."""
        self.assertIsNotNone(self.store.collection)
        self.assertEqual(self.store.collection_name, self.collection_name)
        self.assertEqual(self.store.count_documents(), 0)

    def test_insert_documents_and_count(self):
        """Test 2 & 6: Insert documents with lightweight metadata and verify document count."""
        texts = [
            "Airtel_runu_7550",
            "Password Air@53054",
            "UPI payment receipt of ₹689 to Braingrow",
        ]
        embeddings = self.embedder.embed(texts)
        ids = [f"ss_001_chunk_{i}" for i in range(len(texts))]
        metadatas = [
            {
                "screenshot_id": "ss_001",
                "chunk_index": i,
                "category": "Connectivity" if i < 2 else "Payment",
            }
            for i in range(len(texts))
        ]

        self.store.add_documents(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self.assertEqual(self.store.count_documents(), 3)

    def test_metadata_persistence_and_get_document(self):
        """Test 3: Lightweight metadata persistence and document retrieval by ID."""
        text = "IndiGo Boarding Pass Guwahati to Bhubaneswar"
        embedding = self.embedder.embed([text])[0]
        doc_id = "ss_011_chunk_0"
        meta = {
            "screenshot_id": "ss_011",
            "chunk_index": 0,
            "category": "Travel",
        }

        self.store.add_documents(
            ids=[doc_id],
            documents=[text],
            embeddings=[[float(x) for x in embedding]],
            metadatas=[meta],
        )

        record = self.store.get_document(doc_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["id"], doc_id)
        self.assertEqual(record["document"], text)
        self.assertEqual(record["metadata"]["screenshot_id"], "ss_011")
        self.assertEqual(record["metadata"]["category"], "Travel")

    def test_similarity_search_and_top_k(self):
        """Test 4 & 5: Retrieve by semantic similarity and Top-K retrieval."""
        texts = [
            "Airtel Wi-Fi network password Air@53054",
            "Payment receipt ₹1,800 paid to Ipsita Dilip",
            "Deep Learning course curriculum and syllabus",
            "Flight boarding pass IndiGo seat 16E",
            "VS Code source code Express.js controller",
        ]
        embeddings = self.embedder.embed(texts)
        ids = [f"doc_{i}" for i in range(len(texts))]
        metadatas = [
            {"screenshot_id": f"ss_{i:03d}", "chunk_index": 0, "category": cat}
            for i, cat in enumerate(["Connectivity", "Payment", "Education", "Travel", "Coding"])
        ]

        self.store.add_documents(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Query for Wi-Fi credentials
        query_text = "wifi password credentials"
        query_vec = self.embedder.embed([query_text])[0]

        top_k = 3
        results = self.store.similarity_search(query_embedding=query_vec, top_k=top_k)

        self.assertEqual(len(results), top_k)
        # Top result should be the Connectivity chunk
        top_match = results[0]
        self.assertEqual(top_match["category"], "Connectivity")
        self.assertIn("Airtel", top_match["document"])
        self.assertGreaterEqual(top_match["similarity_score"], 0.0)

    def test_delete_collection(self):
        """Test 7: Delete collection resets storage."""
        texts = ["Sample text chunk"]
        embeddings = self.embedder.embed(texts)
        self.store.add_documents(
            ids=["chunk_0"],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"screenshot_id": "ss_999", "chunk_index": 0, "category": "Test"}],
        )

        self.assertEqual(self.store.count_documents(), 1)
        self.store.delete_collection()
        self.assertEqual(self.store.count_documents(), 0)

    def test_cross_session_persistence(self):
        """Test 8: Database persistence across sessions."""
        texts = ["Persistent chunk data test"]
        embeddings = self.embedder.embed(texts)
        doc_id = "persistent_chunk_0"

        # Session 1: Insert document
        self.store.add_documents(
            ids=[doc_id],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"screenshot_id": "ss_888", "chunk_index": 0, "category": "Persistence"}],
        )
        self.assertEqual(self.store.count_documents(), 1)

        # Session 2: Re-instantiate VectorStoreManager pointing to same directory
        store_session_2 = VectorStoreManager(
            db_dir=self.temp_dir,
            collection_name=self.collection_name,
        )
        self.assertEqual(store_session_2.count_documents(), 1)
        retrieved = store_session_2.get_document(doc_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["document"], texts[0])
        self.assertEqual(retrieved["metadata"]["category"], "Persistence")


if __name__ == "__main__":
    unittest.main()
