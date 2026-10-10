"""
Repeatable End-to-End Multimodal Integration Validation Script
================================================================
Validates 5 core production pipeline integration scenarios:
  1. Semantic-visual agreement (High-confidence semantic prediction authoritative).
  2. Low-confidence ambiguity resolution (Strategy E selective visual layout override).
  3. Sparse/unavailable OCR (Fallback to notes/filename; safe semantic-only operation).
  4. Corrupted or unreadable images (Graceful exception handling without pipeline crash).
  5. SQLite metadata persistence & two-stage retrieval (Layer 1-3 resolution via search_and_enrich).

Generates markdown report: end_to_end_validation_report.md
Returns exit code 0 if all required scenarios PASS, or 1 if any required scenario FAILS/SKIPS.
"""

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Tuple

from PIL import Image
import pandas as pd

# Path setup
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.embedder import SentenceTransformerEmbedder
from src.mock_ocr import MockOCREngine
from src.models.metadata import ScreenshotMetadata
from src.pipeline import OCRPipeline
from src.storage.file_store import LocalFileStore
from src.storage.metadata_store import SQLiteMetadataStore
from src.vector_store import VectorStoreManager


class EndToEndValidator:
    """Repeatable Validator for the 5 End-to-End Multimodal Integration Scenarios."""

    def __init__(self, output_dir: Path = BASE_DIR):
        self.output_dir = output_dir
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.results: List[Dict[str, Any]] = []

    def log_scenario_result(
        self,
        scenario_num: int,
        title: str,
        status: str,
        details: str,
        is_mocked: bool = False,
        error_msg: str = "",
    ):
        result = {
            "scenario_number": scenario_num,
            "title": title,
            "status": status,  # PASS, FAIL, or SKIP
            "details": details,
            "is_mocked": is_mocked,
            "error_msg": error_msg,
        }
        self.results.append(result)
        print(f"Scenario {scenario_num} [{status}] - {title}: {details}")

    def run_scenario_1_agreement(self, temp_dir: Path) -> Dict[str, Any]:
        """Scenario 1: Semantic-Visual Agreement."""
        title = "Semantic-Visual Agreement"
        try:
            shots_dir = temp_dir / "shots_s1"
            shots_dir.mkdir(parents=True, exist_ok=True)
            db_path = temp_dir / "s1.sqlite"

            img_path = shots_dir / "val_agree.png"
            Image.new("RGB", (224, 224), color=(50, 50, 50)).save(img_path)

            csv_path = temp_dir / "meta_s1.csv"
            pd.DataFrame([
                {
                    "Screenshot ID": "ss_val_001",
                    "filename": img_path.name,
                    "Actual Category": "Travel",
                    "notes": "Flight booking itinerary flight boarding pass PNR 12345 to Tokyo status confirmed",
                }
            ]).to_csv(csv_path, index=False)

            mock_ocr = MockOCREngine(mapping={
                img_path.name: "Flight booking itinerary flight boarding pass PNR 12345 to Tokyo status confirmed"
            })

            file_store = LocalFileStore(storage_dir=str(shots_dir))
            metadata_store = SQLiteMetadataStore(db_path=str(db_path))

            pipeline = OCRPipeline(
                ocr_engine=mock_ocr,
                file_store=file_store,
                metadata_store=metadata_store,
                enable_visual=True,
            )

            processed = pipeline.process(metadata_path=str(csv_path), images_dir=str(shots_dir))
            doc = processed[0]
            cat = doc["metadata"]["category"]
            fused = doc["fusion_applied"]
            orig_cat = doc["metadata"]["evidence"].get("original_semantic_category")

            # Explicit Assertions
            assert cat == "Travel", f"Expected category 'Travel', got '{cat}'"
            assert fused is False, f"Expected fusion_applied=False, got {fused}"
            assert orig_cat == "Travel", f"Expected original_semantic_category 'Travel', got '{orig_cat}'"
            assert doc["metadata"]["ocr_available"] is True, "Expected ocr_available=True"

            details = "High-confidence semantic prediction ('Travel') remained authoritative without visual override."
            self.log_scenario_result(1, title, "PASS", details, is_mocked=False)
        except Exception as e:
            self.log_scenario_result(1, title, "FAIL", f"Assertion/Exception failure in scenario: {e}", error_msg=str(e))

    def run_scenario_2_ambiguity_resolution(self, temp_dir: Path) -> Dict[str, Any]:
        """Scenario 2: Low-Confidence Ambiguity Resolution (Selective Fusion Strategy E)."""
        title = "Low-Confidence Ambiguity Resolution (Strategy E Override)"
        try:
            shots_dir = temp_dir / "shots_s2"
            shots_dir.mkdir(parents=True, exist_ok=True)
            db_path = temp_dir / "s2.sqlite"

            img_path = shots_dir / "val_ambig.png"
            Image.new("RGB", (224, 224), color=(30, 30, 30)).save(img_path)

            csv_path = temp_dir / "meta_s2.csv"
            pd.DataFrame([
                {
                    "Screenshot ID": "ss_val_002",
                    "filename": img_path.name,
                    "Actual Category": "General",
                    "notes": "12:34 PM 85%",
                }
            ]).to_csv(csv_path, index=False)

            mock_ocr = MockOCREngine(mapping={img_path.name: "12:34 PM 85%"})
            file_store = LocalFileStore(storage_dir=str(shots_dir))
            metadata_store = SQLiteMetadataStore(db_path=str(db_path))

            # Test using custom adapter mock predicting layout code_editor_terminal (conf 0.85)
            class MockYOLOAdapter:
                def predict_image(self, path: str):
                    return {"screen_type": "code_editor_terminal", "confidence": 0.85}

            pipeline = OCRPipeline(
                ocr_engine=mock_ocr,
                file_store=file_store,
                metadata_store=metadata_store,
                visual_adapter=MockYOLOAdapter(),
                enable_visual=True,
            )

            processed = pipeline.process(metadata_path=str(csv_path), images_dir=str(shots_dir))
            doc = processed[0]
            cat = doc["metadata"]["category"]
            fused = doc["fusion_applied"]
            reason = doc["metadata"]["evidence"].get("fusion_reason", "")
            vis_st = doc["visual_evidence"]["screen_type"]
            orig_cat = doc["metadata"]["evidence"].get("original_semantic_category")

            # Explicit Assertions
            assert cat == "Coding", f"Expected category 'Coding', got '{cat}'"
            assert fused is True, f"Expected fusion_applied=True, got {fused}"
            assert "overrode" in reason, f"Expected override explanation in reason, got '{reason}'"
            assert vis_st == "code_editor_terminal", f"Expected screen_type 'code_editor_terminal', got '{vis_st}'"
            assert orig_cat == "General", f"Expected original_semantic_category 'General', got '{orig_cat}'"

            details = f"Strategy E layout override successfully applied: low-confidence General overrode to '{cat}' via layout 'code_editor_terminal'."
            self.log_scenario_result(2, title, "PASS", details, is_mocked=True)
        except Exception as e:
            self.log_scenario_result(2, title, "FAIL", f"Assertion/Exception failure in scenario: {e}", error_msg=str(e))

    def run_scenario_3_sparse_or_unavailable_ocr(self, temp_dir: Path) -> Dict[str, Any]:
        """Scenario 3: Sparse or Unavailable OCR Context Fallback."""
        title = "Sparse or Unavailable OCR Context Fallback"
        try:
            shots_dir = temp_dir / "shots_s3"
            shots_dir.mkdir(parents=True, exist_ok=True)
            db_path = temp_dir / "s3.sqlite"

            img_path = shots_dir / "val_sparse.png"
            Image.new("RGB", (224, 224), color=(100, 100, 100)).save(img_path)

            csv_path = temp_dir / "meta_s3.csv"
            pd.DataFrame([
                {
                    "Screenshot ID": "ss_val_003",
                    "filename": img_path.name,
                    "Actual Category": "General",
                    "notes": "Flight booking itinerary to Tokyo status confirmed",
                }
            ]).to_csv(csv_path, index=False)

            # OCR returns empty string -> pipeline falls back to notes context
            mock_ocr = MockOCREngine(mapping={img_path.name: ""})
            file_store = LocalFileStore(storage_dir=str(shots_dir))
            metadata_store = SQLiteMetadataStore(db_path=str(db_path))

            pipeline = OCRPipeline(
                ocr_engine=mock_ocr,
                file_store=file_store,
                metadata_store=metadata_store,
                enable_visual=False,  # Visual branch disabled
            )

            processed = pipeline.process(metadata_path=str(csv_path), images_dir=str(shots_dir))
            doc = processed[0]
            cat = doc["metadata"]["category"]
            vis_ev = doc["visual_evidence"]
            fused = doc["fusion_applied"]

            # Explicit Assertions
            assert cat == "Travel", f"Expected notes fallback category 'Travel', got '{cat}'"
            assert vis_ev is None, f"Expected visual_evidence=None, got {vis_ev}"
            assert fused is False, f"Expected fusion_applied=False, got {fused}"

            details = "Empty OCR fell back cleanly to notes context ('Travel'), and disabled visual branch produced visual_evidence=None without error."
            self.log_scenario_result(3, title, "PASS", details, is_mocked=False)
        except Exception as e:
            self.log_scenario_result(3, title, "FAIL", f"Assertion/Exception failure in scenario: {e}", error_msg=str(e))

    def run_scenario_4_corrupted_images(self, temp_dir: Path) -> Dict[str, Any]:
        """Scenario 4: Graceful Handling of Unreadable or Corrupted Image References."""
        title = "Graceful Handling of Corrupted or Unreadable Images"
        try:
            shots_dir = temp_dir / "shots_s4"
            shots_dir.mkdir(parents=True, exist_ok=True)
            db_path = temp_dir / "s4.sqlite"

            # Create a zero-byte invalid image file
            corrupt_img = shots_dir / "corrupt_0byte.png"
            with open(corrupt_img, "wb") as f:
                f.write(b"")

            csv_path = temp_dir / "meta_s4.csv"
            pd.DataFrame([
                {
                    "Screenshot ID": "ss_val_004",
                    "filename": corrupt_img.name,
                    "Actual Category": "General",
                    "notes": "Python script code syntax error traceback",
                }
            ]).to_csv(csv_path, index=False)

            mock_ocr = MockOCREngine(mapping={corrupt_img.name: "Python script code syntax error traceback"})
            file_store = LocalFileStore(storage_dir=str(shots_dir))
            metadata_store = SQLiteMetadataStore(db_path=str(db_path))

            pipeline = OCRPipeline(
                ocr_engine=mock_ocr,
                file_store=file_store,
                metadata_store=metadata_store,
                enable_visual=True,
            )

            processed = pipeline.process(metadata_path=str(csv_path), images_dir=str(shots_dir))
            doc = processed[0]
            cat = doc["metadata"]["category"]
            vis_ev = doc["visual_evidence"]
            fused = doc["fusion_applied"]

            # Explicit Assertions
            assert vis_ev is None, f"Expected visual_evidence=None for corrupted image, got {vis_ev}"
            assert fused is False, f"Expected fusion_applied=False, got {fused}"
            assert cat in ["Coding", "General"], f"Expected fallback prediction, got '{cat}'"

            details = f"Zero-byte corrupted image file was handled gracefully during visual inference. Visual branch caught exception safely, set visual_evidence=None, and predicted category='{cat}' without crashing."
            self.log_scenario_result(4, title, "PASS", details, is_mocked=False)
        except Exception as e:
            self.log_scenario_result(4, title, "FAIL", f"Assertion/Exception failure in scenario: {e}", error_msg=str(e))

    def run_scenario_5_metadata_persistence_and_retrieval(self, temp_dir: Path) -> Dict[str, Any]:
        """Scenario 5: SQLite Metadata Persistence and Two-Stage Retrieval Resolution."""
        title = "SQLite Metadata Persistence and Two-Stage Retrieval (search_and_enrich)"
        vector_store = None
        try:
            shots_dir = temp_dir / "shots_s5"
            shots_dir.mkdir(parents=True, exist_ok=True)
            db_path = temp_dir / "s5.sqlite"
            chroma_dir = temp_dir / "chroma_s5"
            chroma_dir.mkdir(parents=True, exist_ok=True)

            img_path = shots_dir / "val_search.png"
            Image.new("RGB", (224, 224), color=(80, 80, 80)).save(img_path)

            csv_path = temp_dir / "meta_s5.csv"
            pd.DataFrame([
                {
                    "Screenshot ID": "ss_val_005",
                    "filename": img_path.name,
                    "Actual Category": "Payment",
                    "notes": "GPay payment confirmation invoice receipt ₹500",
                }
            ]).to_csv(csv_path, index=False)

            mock_ocr = MockOCREngine(mapping={img_path.name: "Payment successful ₹500.00 via Google Pay UPI transaction ID 123456"})
            file_store = LocalFileStore(storage_dir=str(shots_dir))
            metadata_store = SQLiteMetadataStore(db_path=str(db_path))

            pipeline = OCRPipeline(
                ocr_engine=mock_ocr,
                file_store=file_store,
                metadata_store=metadata_store,
                enable_visual=False,
            )

            processed = pipeline.process(metadata_path=str(csv_path), images_dir=str(shots_dir))
            doc = processed[0]
            sid = doc["screenshot_id"]

            # Verify Layer 2 SQLite Metadata Persistence
            stored_meta = metadata_store.get_metadata(sid)
            assert stored_meta is not None, "Failed to retrieve metadata from SQLite store."
            assert stored_meta.screenshot_id == sid, f"Expected screenshot_id '{sid}', got '{stored_meta.screenshot_id}'"
            assert stored_meta.category == "Payment", f"Expected category 'Payment', got '{stored_meta.category}'"

            # Index in Layer 3 ChromaDB
            embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
            vector_store = VectorStoreManager(db_dir=str(chroma_dir), collection_name="val_collection_s5")

            chunk_text = doc["chunks"][0] if doc["chunks"] else doc["ocr_text"]
            chunk_embedding = embedder.embed([chunk_text])[0]

            vector_store.add_documents(
                ids=[f"{sid}_chunk_0"],
                documents=[chunk_text],
                embeddings=[chunk_embedding],
                metadatas=[{
                    "screenshot_id": sid,
                    "chunk_index": 0,
                    "category": doc["metadata"]["category"],
                }],
            )

            # Execute 2-Stage Retrieval Search
            search_results = pipeline.search_and_enrich(
                query_text="Google Pay payment receipt",
                embedder=embedder,
                vector_store=vector_store,
                top_k=1,
            )

            assert len(search_results) == 1, f"Expected 1 search result, got {len(search_results)}"
            res = search_results[0]
            assert res["screenshot_id"] == sid, f"Expected screenshot_id '{sid}', got '{res['screenshot_id']}'"
            assert res["image_uri"] != "", "Expected non-empty image_uri in 2-stage enriched result."
            assert res["category"] == "Payment", f"Expected category 'Payment', got '{res['category']}'"

            img_name = Path(res["image_uri"]).name if res["image_uri"] else ""
            details = f"2-Stage retrieval resolved vector hit to Layer 2 SQLite & Layer 1 FileStore: screenshot_id='{res['screenshot_id']}', category='{res['category']}', image_uri='{img_name}'."
            self.log_scenario_result(5, title, "PASS", details, is_mocked=False)

        except Exception as e:
            self.log_scenario_result(5, title, "FAIL", f"Assertion/Exception failure in scenario: {e}", error_msg=str(e))
        finally:
            # Proper resource closure for ChromaDB client handles on Windows
            if vector_store is not None and hasattr(vector_store, "client"):
                if hasattr(vector_store.client, "close"):
                    try:
                        vector_store.client.close()
                    except Exception:
                        pass
                elif hasattr(vector_store.client, "_system"):
                    try:
                        vector_store.client._system.stop()
                    except Exception:
                        pass

    def generate_markdown_report(self) -> str:
        all_passed = (len(self.results) == 5) and all(r["status"] == "PASS" for r in self.results)
        overall_status = "PASSED" if all_passed else "FAILED"

        lines = [
            "# End-to-End Multimodal Integration Validation Report",
            "======================================================",
            "",
            "> [!NOTE]",
            "> This report documents the latest local end-to-end multimodal integration validation results.",
            "> Execution uses production pipeline components where available and an isolated deterministic mock layout adapter for ambiguity testing (Scenario 2).",
            "",
            f"**Execution Timestamp**: {self.timestamp}  ",
            f"**Overall Status**: **{overall_status}**  ",
            f"**Total Scenarios**: {len(self.results)}  ",
            "",
            "---",
            "",
            "## Scenario Validation Summary",
            "",
            "| Scenario # | Title | Status | Execution Mode | Details |",
            "| :---: | :--- | :---: | :---: | :--- |",
        ]

        for r in self.results:
            exec_mode = "Mocked Layout" if r["is_mocked"] else "Production Model"
            status_str = f"**{r['status']}**"
            lines.append(
                f"| Scenario {r['scenario_number']} | {r['title']} | {status_str} | {exec_mode} | {r['details']} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Scenario Verification Details",
            "",
        ])

        for r in self.results:
            lines.extend([
                f"### Scenario {r['scenario_number']}: {r['title']}",
                f"- **Status**: **{r['status']}**",
                f"- **Execution Mode**: {'Mocked Layout Adapter' if r['is_mocked'] else 'Production Component'}",
                f"- **Details**: {r['details']}",
            ])
            if r["error_msg"]:
                lines.append(f"- **Error Trace**: `{r['error_msg']}`")
            lines.append("")

        report_content = "\n".join(lines)
        report_path = self.output_dir / "end_to_end_validation_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        json_path = self.output_dir / "end_to_end_validation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": self.timestamp,
                "overall_status": overall_status,
                "results": self.results
            }, f, indent=2)

        print(f"\nGenerated Validation Report: {report_path}")
        return overall_status

    def execute_all_validations(self) -> int:
        print("======================================================================")
        print("      EXECUTING REPEATABLE END-TO-END VALIDATION SUITE                ")
        print("======================================================================")

        with tempfile.TemporaryDirectory(prefix="e2e_val_suite_", ignore_cleanup_errors=True) as tmp_dir_str:
            temp_dir = Path(tmp_dir_str)

            self.run_scenario_1_agreement(temp_dir)
            self.run_scenario_2_ambiguity_resolution(temp_dir)
            self.run_scenario_3_sparse_or_unavailable_ocr(temp_dir)
            self.run_scenario_4_corrupted_images(temp_dir)
            self.run_scenario_5_metadata_persistence_and_retrieval(temp_dir)

            gc.collect()  # Release any unreferenced file handles on Windows before directory cleanup

        overall = self.generate_markdown_report()
        return 0 if overall == "PASSED" else 1


if __name__ == "__main__":
    validator = EndToEndValidator()
    sys.exit(validator.execute_all_validations())
