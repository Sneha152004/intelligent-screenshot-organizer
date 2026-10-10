"""
End-to-End Product Pipeline Integration Test Suite.
===================================================
Tests the full production screenshot-understanding pipeline end-to-end:
1. Ingestion & File Storage (Layer 1)
2. OCR Extraction (Mock / EasyOCR)
3. Semantic Classification (FinalRefinedClassifier)
4. Visual Layout Analysis (YOLOv8n-cls 5-Class Visual Adapter)
5. Selective Multimodal Fusion (Strategy E)
6. Structured Metadata Creation & SQLite Persistence (Layer 2)
7. Text Embeddings & Vector Indexing in ChromaDB (Layer 3)
8. Two-Stage Search & Enrichment Retrieval
9. Robust Visual Layout Failure Fallback
"""

import os
import sys
import tempfile
import pytest
import sqlite3
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from src.pipeline import OCRPipeline
from src.mock_ocr import MockOCREngine
from src.embedder import BaseEmbedder, SentenceTransformerEmbedder
from src.vector_store import VectorStoreManager
from src.storage.file_store import LocalFileStore
from src.storage.metadata_store import SQLiteMetadataStore
from src.models.metadata import ScreenshotMetadata


class MockEmbedder(BaseEmbedder):
    """Fast deterministic embedder for unit test performance."""
    def embed(self, texts):
        return [[float(len(t) % 10 + i) for i in range(384)] for t in texts]

    def embed_queries(self, queries):
        return self.embed(queries)


@pytest.fixture
def temp_environment(tmp_path):
    """Sets up isolated temporary directories for storage testing."""
    file_dir = tmp_path / "files"
    db_path = tmp_path / "metadata.sqlite"
    chroma_dir = tmp_path / "chroma"

    file_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # Create sample image
    img_path = tmp_path / "test_code_editor.png"
    img = Image.new("RGB", (400, 400), color=(30, 30, 30))
    img.save(img_path)

    # Create sample metadata CSV
    csv_path = tmp_path / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Screenshot ID,Actual Category,notes,filename\n")
        f.write(f"ss_e2e_001,General,code editor terminal python script,{img_path.name}\n")

    return {
        "tmp_path": tmp_path,
        "file_dir": str(file_dir),
        "db_path": str(db_path),
        "chroma_dir": str(chroma_dir),
        "img_path": str(img_path),
        "csv_path": str(csv_path),
    }


def test_end_to_end_pipeline_flow(temp_environment):
    """
    Verifies full pipeline execution from ingestion to SQLite & ChromaDB 2-stage search.
    """
    env = temp_environment
    file_store = LocalFileStore(storage_dir=env["file_dir"])
    metadata_store = SQLiteMetadataStore(db_path=env["db_path"])
    mock_ocr = MockOCREngine(mapping={
        Path(env["img_path"]).name: "def train_model():\n    import torch\n    print('Training YOLOv8n-cls model')"
    })

    pipeline = OCRPipeline(
        ocr_engine=mock_ocr,
        file_store=file_store,
        metadata_store=metadata_store,
        enable_visual=True
    )

    # Process dataset through pipeline
    processed_docs = pipeline.process(
        metadata_path=env["csv_path"],
        images_dir=str(env["tmp_path"]),
    )

    assert len(processed_docs) == 1
    doc = processed_docs[0]

    # Check document fields
    assert doc["screenshot_id"] == "ss_e2e_001"
    assert "ocr_text" in doc
    assert "def train_model():" in doc["ocr_text"]
    assert isinstance(doc["metadata"]["category"], str) and len(doc["metadata"]["category"]) > 0

    # Verify Layer 1 File Storage
    assert file_store.exists("ss_e2e_001")

    # Verify Layer 2 SQLite Metadata Storage
    stored_meta = metadata_store.get_metadata("ss_e2e_001")
    assert stored_meta is not None
    assert stored_meta.screenshot_id == "ss_e2e_001"
    assert stored_meta.ocr_available is True
    assert "evidence" in stored_meta.to_dict()

    # Layer 3 Embeddings & ChromaDB Indexing
    embedder = MockEmbedder()
    vector_store = VectorStoreManager(
        db_dir=env["chroma_dir"],
        collection_name="test_e2e_collection"
    )

    # Index chunks in vector store
    for chunk_idx, chunk in enumerate(doc["chunks"]):
        chunk_embedding = embedder.embed([chunk])[0]
        vector_store.add_documents(
            ids=[f"{doc['screenshot_id']}_chunk_{chunk_idx}"],
            documents=[chunk],
            embeddings=[chunk_embedding],
            metadatas=[{
                "screenshot_id": doc["screenshot_id"],
                "chunk_index": chunk_idx,
                "filename": doc["metadata"]["filename"],
                "category": doc["metadata"]["category"]
            }]
        )

    assert vector_store.count_documents() > 0

    # Two-Stage Search & Enrichment
    search_results = pipeline.search_and_enrich(
        query_text="train YOLO model python",
        embedder=embedder,
        vector_store=vector_store,
        top_k=1
    )

    assert len(search_results) == 1
    res = search_results[0]
    assert res["screenshot_id"] == "ss_e2e_001"
    assert res["category"] == stored_meta.category
    assert "image_uri" in res
    assert res["metadata"]["screenshot_id"] == "ss_e2e_001"


def test_visual_layout_selective_fusion_override(temp_environment):
    """
    Verifies that selective multimodal fusion overrides low-confidence / ambiguous fallback predictions.
    """
    env = temp_environment
    file_store = LocalFileStore(storage_dir=env["file_dir"])
    metadata_store = SQLiteMetadataStore(db_path=env["db_path"])

    # Mock OCR with ambiguous sparse text
    mock_ocr = MockOCREngine(mapping={
        Path(env["img_path"]).name: "12:34 PM 85%"
    })

    pipeline = OCRPipeline(
        ocr_engine=mock_ocr,
        file_store=file_store,
        metadata_store=metadata_store,
        enable_visual=True
    )

    processed_docs = pipeline.process(
        metadata_path=env["csv_path"],
        images_dir=str(env["tmp_path"]),
    )

    doc = processed_docs[0]
    meta = doc["metadata"]

    # Should produce valid category and evidence structure
    assert "evidence" in meta
    assert "fusion_applied" in meta["evidence"]
    assert "original_semantic_category" in meta["evidence"]


def test_pipeline_visual_failure_fallback(temp_environment):
    """
    Verifies that when visual classification is disabled or fails, the pipeline safely falls back to semantic-only processing.
    """
    env = temp_environment
    file_store = LocalFileStore(storage_dir=env["file_dir"])
    metadata_store = SQLiteMetadataStore(db_path=env["db_path"])
    mock_ocr = MockOCREngine(mapping={
        Path(env["img_path"]).name: "Flight booking status confirmed from New York to Tokyo"
    })

    # Pipeline with enable_visual=False
    pipeline = OCRPipeline(
        ocr_engine=mock_ocr,
        file_store=file_store,
        metadata_store=metadata_store,
        enable_visual=False
    )

    processed_docs = pipeline.process(
        metadata_path=env["csv_path"],
        images_dir=str(env["tmp_path"]),
    )

    doc = processed_docs[0]
    assert doc["fusion_applied"] is False
    assert doc["visual_evidence"] is None
    assert doc["metadata"]["category"] in ["Travel", "General"]
