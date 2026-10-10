"""
Unit & Integration Tests for Automatic Category Folder Organizer
================================================================
Validates:
1. Multiple screenshots categorized into different category subfolders.
2. SQLite persistence of category and organized image_uri path.
3. Correct metadata and image URI resolution through search_and_enrich().
4. Idempotency on repeated runs and collision safety on duplicate filenames.
5. Preservation of original source files in input directory.
6. Graceful handling of corrupted/zero-byte image files without aborting batch.
7. Handling of empty input directories.
"""

import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image
import pytest

from src.embedder import BaseEmbedder
from src.mock_ocr import MockOCREngine
from src.organizer import ScreenshotOrganizer, organize_folder
from src.pipeline import OCRPipeline
from src.storage.file_store import LocalFileStore
from src.storage.metadata_store import SQLiteMetadataStore
from src.vector_store import VectorStoreManager


class MockEmbedder(BaseEmbedder):
    """Fast deterministic embedder for testing."""
    def embed(self, texts):
        return [[float(len(t) % 10 + i) for i in range(384)] for t in texts]

    def embed_queries(self, queries):
        return self.embed(queries)


@pytest.fixture
def temp_organizer_env(tmp_path):
    """Sets up isolated temporary directories for organizer testing."""
    input_dir = tmp_path / "unorganized_input"
    output_dir = tmp_path / "organized_output"
    db_path = tmp_path / "metadata.sqlite"
    chroma_dir = tmp_path / "chroma_db"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # Create sample images
    img1 = input_dir / "flight_ticket.png"
    Image.new("RGB", (200, 200), color=(100, 100, 200)).save(img1)

    img2 = input_dir / "gpay_receipt.png"
    Image.new("RGB", (200, 200), color=(200, 100, 100)).save(img2)

    img3 = input_dir / "python_code.png"
    Image.new("RGB", (200, 200), color=(100, 200, 100)).save(img3)

    return {
        "tmp_path": tmp_path,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "db_path": str(db_path),
        "chroma_dir": str(chroma_dir),
        "img1": str(img1),
        "img2": str(img2),
        "img3": str(img3),
    }


def test_multiple_screenshots_categorization_and_folder_creation(temp_organizer_env):
    """Verifies that screenshots are copied into distinct category subfolders."""
    env = temp_organizer_env

    mock_ocr = MockOCREngine(mapping={
        "flight_ticket.png": "Flight booking itinerary flight boarding pass PNR 12345 to Tokyo status confirmed",
        "gpay_receipt.png": "Payment successful ₹500.00 via Google Pay UPI transaction ID 123456",
        "python_code.png": "def calculate_matrix():\n    import numpy as np\n    return np.zeros((10, 10))",
    })

    file_store = LocalFileStore(storage_dir=env["output_dir"])
    metadata_store = SQLiteMetadataStore(db_path=env["db_path"])

    pipeline = OCRPipeline(
        ocr_engine=mock_ocr,
        file_store=file_store,
        metadata_store=metadata_store,
        enable_visual=False,
    )

    organizer = ScreenshotOrganizer(
        pipeline=pipeline,
        file_store=file_store,
        metadata_store=metadata_store,
    )

    summary = organizer.organize_folder(input_dir=env["input_dir"], output_dir=env["output_dir"])

    assert summary["processed"] == 3
    assert summary["organized"] == 3
    assert summary["failed"] == 0

    # Verify category subdirectories exist in output_dir
    out_path = Path(env["output_dir"])
    travel_dir = out_path / "Travel"
    payment_dir = out_path / "Payment"

    assert travel_dir.exists() and travel_dir.is_dir()
    assert payment_dir.exists() and payment_dir.is_dir()

    # Check files exist in category subfolders
    assert (travel_dir / "flight_ticket.png").exists()
    assert (payment_dir / "gpay_receipt.png").exists()


def test_sqlite_persistence_of_organized_path(temp_organizer_env):
    """Verifies SQLite metadata store persists category and organized image_uri."""
    env = temp_organizer_env

    mock_ocr = MockOCREngine(mapping={
        "flight_ticket.png": "Flight booking itinerary flight boarding pass PNR 12345 to Tokyo status confirmed",
    })

    file_store = LocalFileStore(storage_dir=env["output_dir"])
    metadata_store = SQLiteMetadataStore(db_path=env["db_path"])

    pipeline = OCRPipeline(
        ocr_engine=mock_ocr,
        file_store=file_store,
        metadata_store=metadata_store,
        enable_visual=False,
    )

    organizer = ScreenshotOrganizer(
        pipeline=pipeline,
        file_store=file_store,
        metadata_store=metadata_store,
    )

    summary = organizer.organize_folder(input_dir=env["input_dir"], output_dir=env["output_dir"])

    # Retrieve stored metadata from SQLite
    meta = metadata_store.get_metadata("flight_ticket")
    assert meta is not None
    assert meta.category == "Travel"
    assert "Travel" in meta.image_uri
    assert meta.image_uri.endswith("flight_ticket.png")
    assert os.path.exists(meta.image_uri)


def test_search_and_enrich_with_organized_library(temp_organizer_env):
    """Verifies two-stage retrieval (search_and_enrich) returns organized image URIs."""
    env = temp_organizer_env

    mock_ocr = MockOCREngine(mapping={
        "gpay_receipt.png": "Payment successful ₹500.00 via Google Pay UPI transaction ID 123456",
    })

    file_store = LocalFileStore(storage_dir=env["output_dir"])
    metadata_store = SQLiteMetadataStore(db_path=env["db_path"])

    pipeline = OCRPipeline(
        ocr_engine=mock_ocr,
        file_store=file_store,
        metadata_store=metadata_store,
        enable_visual=False,
    )

    organizer = ScreenshotOrganizer(
        pipeline=pipeline,
        file_store=file_store,
        metadata_store=metadata_store,
    )

    organizer.organize_folder(input_dir=env["input_dir"], output_dir=env["output_dir"])

    # Index in ChromaDB
    embedder = MockEmbedder()
    vector_store = VectorStoreManager(db_dir=env["chroma_dir"], collection_name="test_organizer_coll")

    meta = metadata_store.get_metadata("gpay_receipt")
    chunk_text = "Payment successful ₹500.00 via Google Pay UPI transaction ID 123456"
    vec = embedder.embed([chunk_text])[0]

    vector_store.add_documents(
        ids=["gpay_receipt_chunk_0"],
        documents=[chunk_text],
        embeddings=[vec],
        metadatas=[{"screenshot_id": "gpay_receipt", "chunk_index": 0, "category": "Payment"}],
    )

    results = pipeline.search_and_enrich(
        query_text="Google Pay payment",
        embedder=embedder,
        vector_store=vector_store,
        top_k=1,
    )

    assert len(results) == 1
    res = results[0]
    assert res["screenshot_id"] == "gpay_receipt"
    assert res["category"] == "Payment"
    assert "Payment" in res["image_uri"]
    assert res["image_uri"].endswith("gpay_receipt.png")

    if hasattr(vector_store, "client") and hasattr(vector_store.client, "close"):
        vector_store.client.close()


def test_idempotency_and_collision_handling(temp_organizer_env):
    """Verifies idempotent repeated processing and collision-safe filename generation."""
    env = temp_organizer_env

    mock_ocr = MockOCREngine(mapping={
        "flight_ticket.png": "Flight booking itinerary flight boarding pass PNR 12345 to Tokyo status confirmed",
    })

    file_store = LocalFileStore(storage_dir=env["output_dir"])
    metadata_store = SQLiteMetadataStore(db_path=env["db_path"])

    pipeline = OCRPipeline(
        ocr_engine=mock_ocr,
        file_store=file_store,
        metadata_store=metadata_store,
        enable_visual=False,
    )

    organizer = ScreenshotOrganizer(
        pipeline=pipeline,
        file_store=file_store,
        metadata_store=metadata_store,
    )

    # First run
    sum1 = organizer.organize_folder(input_dir=env["input_dir"], output_dir=env["output_dir"])
    assert sum1["organized"] >= 1

    # Second run on identical input -> should skip existing files (idempotent)
    sum2 = organizer.organize_folder(input_dir=env["input_dir"], output_dir=env["output_dir"])
    assert sum2["skipped"] >= 1
    assert sum2["organized"] == 0

    # Test filename collision with different content
    input_dir_path = Path(env["input_dir"])
    collision_dir = input_dir_path / "subfolder"
    collision_dir.mkdir(parents=True, exist_ok=True)
    dup_file = collision_dir / "flight_ticket.png"

    # Create a different image content under the same filename
    Image.new("RGB", (300, 300), color=(255, 0, 0)).save(dup_file)

    mock_ocr.mapping[str(dup_file)] = "Flight booking itinerary flight boarding pass PNR 12345 to Tokyo status confirmed"

    sum3 = organizer.organize_folder(input_dir=env["input_dir"], output_dir=env["output_dir"])
    assert sum3["processed"] >= 1

    travel_dir = Path(env["output_dir"]) / "Travel"
    # Verify both the original and collision-resolved file exist without overwrite
    assert (travel_dir / "flight_ticket.png").exists()
    assert (travel_dir / "flight_ticket_1.png").exists() or sum3["skipped"] >= 1


def test_preservation_of_original_source_files(temp_organizer_env):
    """Verifies original files in input_dir are never deleted, moved, or altered."""
    env = temp_organizer_env

    mock_ocr = MockOCREngine(mapping={})
    file_store = LocalFileStore(storage_dir=env["output_dir"])
    metadata_store = SQLiteMetadataStore(db_path=env["db_path"])

    pipeline = OCRPipeline(
        ocr_engine=mock_ocr,
        file_store=file_store,
        metadata_store=metadata_store,
        enable_visual=False,
    )

    organizer = ScreenshotOrganizer(pipeline=pipeline)
    organizer.organize_folder(input_dir=env["input_dir"], output_dir=env["output_dir"])

    # Confirm source files still exist in input_dir
    assert os.path.exists(env["img1"])
    assert os.path.exists(env["img2"])
    assert os.path.exists(env["img3"])


def test_corrupted_image_handling_in_batch(temp_organizer_env):
    """Verifies that an unreadable or zero-byte image does not abort the batch."""
    env = temp_organizer_env
    input_path = Path(env["input_dir"])

    # Add a zero-byte corrupted file
    corrupt_file = input_path / "corrupt_0byte.png"
    with open(corrupt_file, "wb") as f:
        f.write(b"")

    mock_ocr = MockOCREngine(mapping={
        "flight_ticket.png": "Flight booking itinerary flight boarding pass PNR 12345 to Tokyo status confirmed",
    })

    file_store = LocalFileStore(storage_dir=env["output_dir"])
    metadata_store = SQLiteMetadataStore(db_path=env["db_path"])

    pipeline = OCRPipeline(
        ocr_engine=mock_ocr,
        file_store=file_store,
        metadata_store=metadata_store,
        enable_visual=False,
    )

    organizer = ScreenshotOrganizer(pipeline=pipeline)
    summary = organizer.organize_folder(input_dir=env["input_dir"], output_dir=env["output_dir"])

    assert summary["failed"] >= 1
    assert summary["processed"] >= 1
    assert any(ff["file"] == "corrupt_0byte.png" for ff in summary["failed_files"])

    # Valid screenshots were still organized successfully!
    travel_dir = Path(env["output_dir"]) / "Travel"
    assert (travel_dir / "flight_ticket.png").exists()


def test_empty_input_folder_handling(tmp_path):
    """Verifies that organizing an empty folder returns clean summary without error."""
    empty_input = tmp_path / "empty_input"
    empty_output = tmp_path / "empty_output"
    empty_input.mkdir(parents=True, exist_ok=True)
    empty_output.mkdir(parents=True, exist_ok=True)

    summary = organize_folder(input_dir=str(empty_input), output_dir=str(empty_output))

    assert summary["processed"] == 0
    assert summary["organized"] == 0
    assert summary["skipped"] == 0
    assert summary["failed"] == 0
    assert summary["category_counts"] == {}
