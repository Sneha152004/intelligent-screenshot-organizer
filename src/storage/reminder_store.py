"""
I.R.I.S. Reminder Store
========================
SQLite persistence manager for screenshot reminders.
Supports CRUD operations, status management, filtering, and summary metrics.
"""

from datetime import datetime, timezone
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReminderStore:
    """
    Manages persistent SQLite storage of reminders.
    Ensures table creation, indexes, atomic updates, and query filtering.
    """

    def __init__(self, db_path: str = "data/metadata.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes the screenshot_reminders table and performance indexes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS screenshot_metadata (
                    screenshot_id TEXT PRIMARY KEY,
                    filename TEXT,
                    image_uri TEXT,
                    category TEXT,
                    intent TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS screenshot_reminders (
                    reminder_id TEXT PRIMARY KEY,
                    screenshot_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    reminder_date TEXT NOT NULL,
                    reminder_time TEXT,
                    reminder_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    importance_score INTEGER DEFAULT 3,
                    status TEXT NOT NULL DEFAULT 'pending',
                    source_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (screenshot_id) REFERENCES screenshot_metadata(screenshot_id) ON DELETE SET NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_status ON screenshot_reminders(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_date ON screenshot_reminders(reminder_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_screenshot ON screenshot_reminders(screenshot_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_type ON screenshot_reminders(reminder_type)")
            conn.commit()

    def create_reminder(self, reminder_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inserts a new reminder into the database."""
        now_iso = datetime.now(timezone.utc).isoformat()
        r_id = reminder_data["reminder_id"]
        sid = reminder_data.get("screenshot_id")
        title = reminder_data["title"]
        desc = reminder_data.get("description", "")
        r_date = reminder_data["reminder_date"]
        r_time = reminder_data.get("reminder_time")
        r_type = reminder_data.get("reminder_type", "general")
        priority = reminder_data.get("priority", "medium")
        importance = reminder_data.get("importance_score", 3)
        status = reminder_data.get("status", "pending")
        source_text = reminder_data.get("source_text", "")
        created_at = reminder_data.get("created_at", now_iso)
        updated_at = reminder_data.get("updated_at", now_iso)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO screenshot_reminders (
                    reminder_id, screenshot_id, title, description,
                    reminder_date, reminder_time, reminder_type, priority,
                    importance_score, status, source_text, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r_id, sid, title, desc, r_date, r_time, r_type,
                    priority, importance, status, source_text, created_at, updated_at
                ),
            )
            conn.commit()

        return self.get_reminder(r_id)

    def get_reminder(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a single reminder by ID, joining screenshot details if available."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.*,
                       s.filename, s.image_uri, s.category, s.intent
                FROM screenshot_reminders r
                LEFT JOIN screenshot_metadata s ON r.screenshot_id = s.screenshot_id
                WHERE r.reminder_id = ?
                """,
                (reminder_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def find_duplicate_reminder(
        self,
        screenshot_id: Optional[str],
        reminder_date: str,
        reminder_type: str,
        title: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Checks if an identical reminder already exists to prevent duplicate creation.
        """
        with self._get_connection() as conn:
            if screenshot_id:
                cursor = conn.execute(
                    """
                    SELECT reminder_id FROM screenshot_reminders
                    WHERE screenshot_id = ? AND reminder_date = ? AND reminder_type = ?
                    LIMIT 1
                    """,
                    (screenshot_id, reminder_date, reminder_type),
                )
                row = cursor.fetchone()
                if row:
                    return self.get_reminder(row["reminder_id"])

            if title:
                cursor = conn.execute(
                    """
                    SELECT reminder_id FROM screenshot_reminders
                    WHERE title = ? AND reminder_date = ?
                    LIMIT 1
                    """,
                    (title, reminder_date),
                )
                row = cursor.fetchone()
                if row:
                    return self.get_reminder(row["reminder_id"])

        return None

    def list_reminders(
        self,
        status: Optional[str] = None,
        reminder_type: Optional[str] = None,
        date_filter: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        sort: str = "asc",
    ) -> List[Dict[str, Any]]:
        """
        Lists reminders matching specified filter conditions.
        """
        query = """
            SELECT r.*,
                   s.filename, s.image_uri, s.category, s.intent
            FROM screenshot_reminders r
            LEFT JOIN screenshot_metadata s ON r.screenshot_id = s.screenshot_id
            WHERE 1=1
        """
        params: List[Any] = []

        if status and status.lower() != "all":
            query += " AND r.status = ?"
            params.append(status.lower())

        if reminder_type and reminder_type.lower() != "all":
            query += " AND r.reminder_type = ?"
            params.append(reminder_type.lower())

        today_str = datetime.now().strftime("%Y-%m-%d")

        if date_filter == "overdue":
            query += " AND r.reminder_date < ? AND r.status = 'pending'"
            params.append(today_str)
        elif date_filter == "today":
            query += " AND r.reminder_date = ? AND r.status = 'pending'"
            params.append(today_str)
        elif date_filter == "upcoming":
            query += " AND r.reminder_date > ? AND r.status = 'pending'"
            params.append(today_str)
        elif date_filter == "completed":
            query += " AND r.status = 'completed'"

        if from_date:
            query += " AND r.reminder_date >= ?"
            params.append(from_date)

        if to_date:
            query += " AND r.reminder_date <= ?"
            params.append(to_date)

        order_dir = "ASC" if sort.lower() == "asc" else "DESC"
        query += f" ORDER BY r.reminder_date {order_dir}, r.created_at {order_dir}"

        with self._get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update_reminder(self, reminder_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates specific fields of an existing reminder."""
        allowed_fields = {
            "title", "description", "reminder_date", "reminder_time",
            "reminder_type", "priority", "importance_score", "status"
        }
        set_clauses = []
        params = []
        for k, v in updates.items():
            if k in allowed_fields:
                set_clauses.append(f"{k} = ?")
                params.append(v)

        if not set_clauses:
            return self.get_reminder(reminder_id)

        set_clauses.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(reminder_id)

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE screenshot_reminders SET {', '.join(set_clauses)} WHERE reminder_id = ?",
                tuple(params),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None

        return self.get_reminder(reminder_id)

    def delete_reminder(self, reminder_id: str) -> bool:
        """Deletes a reminder from the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM screenshot_reminders WHERE reminder_id = ?",
                (reminder_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_summary_metrics(self) -> Dict[str, int]:
        """Computes summary counts across reminder statuses and due horizons."""
        today_str = datetime.now().strftime("%Y-%m-%d")

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'pending' AND reminder_date = ? THEN 1 ELSE 0 END) as due_today,
                    SUM(CASE WHEN status = 'pending' AND reminder_date > ? THEN 1 ELSE 0 END) as upcoming,
                    SUM(CASE WHEN status = 'pending' AND reminder_date < ? THEN 1 ELSE 0 END) as overdue,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'dismissed' THEN 1 ELSE 0 END) as dismissed
                FROM screenshot_reminders
                """,
                (today_str, today_str, today_str),
            )
            row = cursor.fetchone()
            if not row:
                return {
                    "total": 0, "pending": 0, "due_today": 0,
                    "upcoming": 0, "overdue": 0, "completed": 0, "dismissed": 0,
                }
            return {
                "total": int(row["total"] or 0),
                "pending": int(row["pending"] or 0),
                "due_today": int(row["due_today"] or 0),
                "upcoming": int(row["upcoming"] or 0),
                "overdue": int(row["overdue"] or 0),
                "completed": int(row["completed"] or 0),
                "dismissed": int(row["dismissed"] or 0),
            }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Transforms a database row into a structured reminder dictionary with screenshot info."""
        d = {
            "reminder_id": row["reminder_id"],
            "screenshot_id": row["screenshot_id"],
            "title": row["title"],
            "description": row["description"] or "",
            "reminder_date": row["reminder_date"],
            "reminder_time": row["reminder_time"],
            "reminder_type": row["reminder_type"],
            "priority": row["priority"],
            "importance_score": row["importance_score"],
            "status": row["status"],
            "source_text": row["source_text"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

        # Attach nested screenshot details if joined
        if row["screenshot_id"]:
            fname = row["filename"] or f"{row['screenshot_id']}.jpg"
            img_uri = row["image_uri"] or f"data/screenshots/{fname}"
            img_url = f"/data/screenshots/{fname}" if not img_uri.startswith("/") else img_uri
            d["screenshot"] = {
                "screenshot_id": row["screenshot_id"],
                "filename": fname,
                "image_url": img_url,
                "thumbnail_url": img_url,
                "category": row["category"] or "General",
                "intent": row["intent"] or "General Information",
            }
        else:
            d["screenshot"] = None

        return d
