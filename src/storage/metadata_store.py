"""
Structured Metadata Storage Layer (Layer 2)
=============================================
Abstracts application-level structured metadata storage.
Uses SQLite for local prototype storage; handles non-destructive schema migrations
and deterministic JSON serialization/deserialization for structured evidence fields.
"""

from contextlib import contextmanager
import json
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Optional

from ..models.metadata import ScreenshotMetadata


class BaseMetadataStore(ABC):
    """Abstract Base Class for Structured Application Metadata Storage."""

    @abstractmethod
    def save_metadata(self, metadata: ScreenshotMetadata) -> None:
        """Saves or updates screenshot metadata record."""
        pass

    @abstractmethod
    def get_metadata(self, screenshot_id: str) -> Optional[ScreenshotMetadata]:
        """Retrieves screenshot metadata record by screenshot_id."""
        pass

    @abstractmethod
    def list_all_metadata(self) -> List[ScreenshotMetadata]:
        """Lists all screenshot metadata records."""
        pass

    @abstractmethod
    def delete_metadata(self, screenshot_id: str) -> None:
        """Deletes screenshot metadata record by screenshot_id."""
        pass


class SQLiteMetadataStore(BaseMetadataStore):
    """SQLite implementation of structured metadata storage."""

    def __init__(self, db_path: str = "data/metadata.sqlite"):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Creates the metadata table schema and applies non-destructive migration if needed."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS screenshot_metadata (
                    screenshot_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    image_uri TEXT NOT NULL,
                    category TEXT DEFAULT 'General',
                    content_type TEXT DEFAULT 'unknown',
                    ocr_available INTEGER DEFAULT 1,
                    created_at TEXT,
                    indexed_at TEXT,
                    tags TEXT,
                    intent TEXT,
                    summary TEXT,
                    importance INTEGER,
                    evidence TEXT,
                    events TEXT,
                    financial_info TEXT,
                    contacts TEXT,
                    urls TEXT,
                    text_topics TEXT,
                    entities TEXT,
                    dates TEXT,
                    action_items TEXT,
                    sensitive_info TEXT,
                    confidence_scores TEXT,
                    category_candidates TEXT,
                    category_scores TEXT,
                    classification_confidence REAL DEFAULT 0.0,
                    classification_ambiguous INTEGER DEFAULT 0,
                    classification_margin REAL DEFAULT 0.0,
                    classification_evidence TEXT
                )
                """
            )
            # Safe non-destructive migration for existing SQLite databases
            existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(screenshot_metadata)").fetchall()]
            column_defs = {
                "category": "TEXT DEFAULT 'General'",
                "content_type": "TEXT DEFAULT 'unknown'",
                "tags": "TEXT",
                "intent": "TEXT",
                "summary": "TEXT",
                "importance": "INTEGER",
                "evidence": "TEXT",
                "events": "TEXT",
                "financial_info": "TEXT",
                "contacts": "TEXT",
                "urls": "TEXT",
                "text_topics": "TEXT",
                "entities": "TEXT",
                "dates": "TEXT",
                "action_items": "TEXT",
                "sensitive_info": "TEXT",
                "confidence_scores": "TEXT",
                "category_candidates": "TEXT",
                "category_scores": "TEXT",
                "classification_confidence": "REAL DEFAULT 0.0",
                "classification_ambiguous": "INTEGER DEFAULT 0",
                "classification_margin": "REAL DEFAULT 0.0",
                "classification_evidence": "TEXT",
            }
            for col, col_def in column_defs.items():
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE screenshot_metadata ADD COLUMN {col} {col_def}")

    def save_metadata(self, metadata: ScreenshotMetadata) -> None:
        """Saves or updates screenshot metadata record using standard JSON serialization for complex fields."""
        metadata.validate()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO screenshot_metadata (
                    screenshot_id, filename, image_uri, category, content_type, ocr_available,
                    created_at, indexed_at, tags, intent, summary, importance,
                    evidence, events, financial_info, contacts, urls, text_topics,
                    entities, dates, action_items, sensitive_info, confidence_scores,
                    category_candidates, category_scores, classification_confidence,
                    classification_ambiguous, classification_margin, classification_evidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(screenshot_id) DO UPDATE SET
                    filename=excluded.filename,
                    image_uri=excluded.image_uri,
                    category=excluded.category,
                    content_type=excluded.content_type,
                    ocr_available=excluded.ocr_available,
                    created_at=excluded.created_at,
                    indexed_at=excluded.indexed_at,
                    tags=excluded.tags,
                    intent=excluded.intent,
                    summary=excluded.summary,
                    importance=excluded.importance,
                    evidence=excluded.evidence,
                    events=excluded.events,
                    financial_info=excluded.financial_info,
                    contacts=excluded.contacts,
                    urls=excluded.urls,
                    text_topics=excluded.text_topics,
                    entities=excluded.entities,
                    dates=excluded.dates,
                    action_items=excluded.action_items,
                    sensitive_info=excluded.sensitive_info,
                    confidence_scores=excluded.confidence_scores,
                    category_candidates=excluded.category_candidates,
                    category_scores=excluded.category_scores,
                    classification_confidence=excluded.classification_confidence,
                    classification_ambiguous=excluded.classification_ambiguous,
                    classification_margin=excluded.classification_margin,
                    classification_evidence=excluded.classification_evidence
                """,
                (
                    metadata.screenshot_id,
                    metadata.filename,
                    metadata.image_uri,
                    metadata.category,
                    metadata.content_type,
                    1 if metadata.ocr_available else 0,
                    metadata.created_at,
                    metadata.indexed_at,
                    json.dumps(metadata.tags),
                    metadata.intent,
                    metadata.summary,
                    metadata.importance,
                    json.dumps(metadata.evidence),
                    json.dumps(metadata.events),
                    json.dumps(metadata.financial_info),
                    json.dumps(metadata.contacts),
                    json.dumps(metadata.urls),
                    json.dumps(metadata.text_topics),
                    json.dumps(metadata.entities),
                    json.dumps(metadata.dates),
                    json.dumps(metadata.action_items),
                    json.dumps(metadata.sensitive_info),
                    json.dumps(metadata.confidence_scores),
                    json.dumps(metadata.category_candidates),
                    json.dumps(metadata.category_scores),
                    float(metadata.classification_confidence),
                    1 if metadata.classification_ambiguous else 0,
                    float(metadata.classification_margin),
                    json.dumps(metadata.classification_evidence),
                ),
            )

    def get_metadata(self, screenshot_id: str) -> Optional[ScreenshotMetadata]:
        """Retrieves and deserializes screenshot metadata record into a ScreenshotMetadata object."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM screenshot_metadata WHERE screenshot_id = ?",
                (screenshot_id,),
            ).fetchone()

        if not row:
            return None

        return ScreenshotMetadata.from_dict(dict(row))

    def list_all_metadata(self) -> List[ScreenshotMetadata]:
        """Lists all screenshot metadata records deserialized into ScreenshotMetadata objects."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM screenshot_metadata").fetchall()

        return [ScreenshotMetadata.from_dict(dict(row)) for row in rows]

    def delete_metadata(self, screenshot_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM screenshot_metadata WHERE screenshot_id = ?", (screenshot_id,))

    def clear_all(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM screenshot_metadata")
