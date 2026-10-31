"""
Persistent Conversation Storage Layer
=====================================
Manages persistent conversation sessions and chat messages in SQLite.
Completely decoupled from ScreenshotMetadata records, allowing safe atomic operations,
session management, and message history without modifying screenshot metadata.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SQLiteConversationStore:
    """
    SQLite-backed store for conversation sessions and messages.
    Uses parameterized SQL and strict foreign key integrity.
    """

    def __init__(self, db_path: str = "data/metadata.sqlite"):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create conversations and chat_messages tables non-destructively."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    referenced_ids TEXT DEFAULT '[]',
                    references_json TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_conv_created ON chat_messages(conversation_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC)"
            )

    def create_conversation(
        self,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new conversation session."""
        cid = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        t = title.strip() if title and title.strip() else "New Conversation"
        now = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    updated_at=excluded.updated_at
                """,
                (cid, t, now, now),
            )

        return {
            "conversation_id": cid,
            "title": t,
            "created_at": now,
            "updated_at": now,
        }

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation session by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT conversation_id, title, created_at, updated_at
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "conversation_id": row["conversation_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List conversation sessions ordered by most recent activity."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.conversation_id, c.title, c.created_at, c.updated_at,
                       COUNT(m.message_id) AS message_count
                FROM conversations c
                LEFT JOIN chat_messages m ON c.conversation_id = m.conversation_id
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    "conversation_id": r["conversation_id"],
                    "title": r["title"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "message_count": int(r["message_count"] or 0),
                }
                for r in rows
            ]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation session and all its messages."""
        with self._get_connection() as conn:
            # Explicit delete of messages ensures cleanup even if foreign keys are disabled
            conn.execute(
                "DELETE FROM chat_messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            cur = conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            return cur.rowcount > 0

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (title.strip(), now, conversation_id),
            )
            return cur.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        referenced_ids: Optional[List[str]] = None,
        references: Optional[List[Dict[str, Any]]] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a message to a conversation. Auto-creates conversation if missing,
        and auto-titles conversation from the first user message.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: '{role}'. Must be 'user' or 'assistant'.")

        mid = message_id or f"msg_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        ref_ids_clean = [str(x) for x in (referenced_ids or []) if x]
        refs_clean = references or []

        ref_ids_str = json.dumps(ref_ids_clean)
        refs_str = json.dumps(refs_clean)

        with self._get_connection() as conn:
            # Ensure parent conversation exists
            conv_row = conn.execute(
                "SELECT conversation_id, title FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()

            if not conv_row:
                # Auto-create conversation
                init_title = content.strip().replace("\n", " ")[:36] if role == "user" else "New Conversation"
                if not init_title:
                    init_title = "New Conversation"
                conn.execute(
                    """
                    INSERT INTO conversations (conversation_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (conversation_id, init_title, now, now),
                )
            elif role == "user" and conv_row["title"] == "New Conversation":
                # Auto-update placeholder title on first substantive user message
                derived_title = content.strip().replace("\n", " ")
                if len(derived_title) > 36:
                    derived_title = derived_title[:33] + "..."
                if derived_title:
                    conn.execute(
                        "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                        (derived_title, conversation_id),
                    )

            # Insert message
            conn.execute(
                """
                INSERT INTO chat_messages (
                    message_id, conversation_id, role, content, referenced_ids, references_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (mid, conversation_id, role, content, ref_ids_str, refs_str, now),
            )

            # Touch updated_at on conversation
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )

        return {
            "message_id": mid,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "referenced_ids": ref_ids_clean,
            "references": refs_clean,
            "created_at": now,
        }

    def get_messages(self, conversation_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all messages for a conversation ordered chronologically."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT message_id, conversation_id, role, content, referenced_ids, references_json, created_at
                FROM chat_messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()

            messages = []
            for r in rows:
                try:
                    ref_ids = json.loads(r["referenced_ids"]) if r["referenced_ids"] else []
                except Exception:
                    ref_ids = []
                try:
                    refs = json.loads(r["references_json"]) if r["references_json"] else []
                except Exception:
                    refs = []

                messages.append({
                    "message_id": r["message_id"],
                    "conversation_id": r["conversation_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "referenced_ids": ref_ids,
                    "references": refs,
                    "created_at": r["created_at"],
                })
            return messages

    def get_last_assistant_message(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the immediately preceding assistant message for a conversation."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT message_id, conversation_id, role, content, referenced_ids, references_json, created_at
                FROM chat_messages
                WHERE conversation_id = ? AND role = 'assistant'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if not row:
                return None
            try:
                ref_ids = json.loads(row["referenced_ids"]) if row["referenced_ids"] else []
            except Exception:
                ref_ids = []
            try:
                refs = json.loads(row["references_json"]) if row["references_json"] else []
            except Exception:
                refs = []
            return {
                "message_id": row["message_id"],
                "conversation_id": row["conversation_id"],
                "role": row["role"],
                "content": row["content"],
                "referenced_ids": ref_ids,
                "references": refs,
                "created_at": row["created_at"],
            }
