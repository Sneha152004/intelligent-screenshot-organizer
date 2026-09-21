"""
Structured Metadata Storage Layer (Layer 2)
=============================================
Abstracts application-level structured metadata storage.
Uses SQLite for local prototype storage; can be replaced with PostgreSQL without changing application logic.
Handles deterministic JSON serialization/deserialization for structured intelligence fields.
"""

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

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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
                    ocr_available INTEGER DEFAULT 1,
                    created_at TEXT,
                    indexed_at TEXT,
                    tags TEXT,
                    intent TEXT,
                    summary TEXT,
                    importance INTEGER,
                    entities TEXT,
                    dates TEXT,
                    action_items TEXT,
                    sensitive_info TEXT
                )
                """
            )
            # Safe migration for existing SQLite databases
            existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(screenshot_metadata)").fetchall()]
            column_defs = {
                "tags": "TEXT",
                "intent": "TEXT",
                "summary": "TEXT",
                "importance": "INTEGER",
                "entities": "TEXT",
                "dates": "TEXT",
                "action_items": "TEXT",
                "sensitive_info": "TEXT",
            }
            for col, col_type in column_defs.items():
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE screenshot_metadata ADD COLUMN {col} {col_type}")

    def save_metadata(self, metadata: ScreenshotMetadata) -> None:
        """Saves or updates screenshot metadata record using standard JSON serialization for complex fields."""
        metadata.validate()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO screenshot_metadata (
                    screenshot_id, filename, image_uri, category, ocr_available,
                    created_at, indexed_at, tags, intent, summary, importance,
                    entities, dates, action_items, sensitive_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(screenshot_id) DO UPDATE SET
                    filename=excluded.filename,
                    image_uri=excluded.image_uri,
                    category=excluded.category,
                    ocr_available=excluded.ocr_available,
                    created_at=excluded.created_at,
                    indexed_at=excluded.indexed_at,
                    tags=excluded.tags,
                    intent=excluded.intent,
                    summary=excluded.summary,
                    importance=excluded.importance,
                    entities=excluded.entities,
                    dates=excluded.dates,
                    action_items=excluded.action_items,
                    sensitive_info=excluded.sensitive_info
                """,
                (
                    metadata.screenshot_id,
                    metadata.filename,
                    metadata.image_uri,
                    metadata.category,
                    1 if metadata.ocr_available else 0,
                    metadata.created_at,
                    metadata.indexed_at,
                    json.dumps(metadata.tags),
                    metadata.intent,
                    metadata.summary,
                    metadata.importance,
                    json.dumps(metadata.entities),
                    json.dumps(metadata.dates),
                    json.dumps(metadata.action_items),
                    json.dumps(metadata.sensitive_info),
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
