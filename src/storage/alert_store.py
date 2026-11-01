"""
I.R.I.S. Alert Store
====================
SQLite persistence manager for intelligent screenshot alerts.
Provides atomic transactions, indexed lookups, status transitions,
and alert summary aggregations.
"""

from datetime import datetime, timezone
import logging
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertStore:
    """
    Manages persistent SQLite storage of system alerts.
    """

    def __init__(self, db_path: str = "data/metadata.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes the screenshot_alerts table and performance indexes."""
        with self._get_connection() as conn:
            # Ensure prerequisite tables exist for foreign references if in isolated test db
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
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS screenshot_alerts (
                    alert_id TEXT PRIMARY KEY,
                    reminder_id TEXT,
                    screenshot_id TEXT,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    trigger_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (reminder_id) REFERENCES screenshot_reminders(reminder_id) ON DELETE SET NULL,
                    FOREIGN KEY (screenshot_id) REFERENCES screenshot_metadata(screenshot_id) ON DELETE SET NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON screenshot_alerts(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON screenshot_alerts(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_reminder ON screenshot_alerts(reminder_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_screenshot ON screenshot_alerts(screenshot_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type ON screenshot_alerts(alert_type)")
            conn.commit()

    def create_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inserts a new alert into SQLite."""
        now_iso = datetime.now(timezone.utc).isoformat()
        a_id = alert_data["alert_id"]
        rem_id = alert_data.get("reminder_id")
        sid = alert_data.get("screenshot_id")
        title = alert_data["title"]
        msg = alert_data["message"]
        a_type = alert_data["alert_type"]
        sev = alert_data.get("severity", "medium")
        status = alert_data.get("status", "active")
        trigger_date = alert_data["trigger_date"]
        created_at = alert_data.get("created_at", now_iso)
        updated_at = alert_data.get("updated_at", now_iso)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO screenshot_alerts (
                    alert_id, reminder_id, screenshot_id, title, message,
                    alert_type, severity, status, trigger_date, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (a_id, rem_id, sid, title, msg, a_type, sev, status, trigger_date, created_at, updated_at),
            )
            conn.commit()

        return self.get_alert(a_id)

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single alert with joined reminder and screenshot details."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT a.*,
                       s.filename, s.image_uri, s.category, s.intent,
                       r.title as reminder_title, r.reminder_date, r.status as reminder_status
                FROM screenshot_alerts a
                LEFT JOIN screenshot_metadata s ON a.screenshot_id = s.screenshot_id
                LEFT JOIN screenshot_reminders r ON a.reminder_id = r.reminder_id
                WHERE a.alert_id = ?
                """,
                (alert_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def find_active_alert(self, reminder_id: Optional[str], alert_type: str) -> Optional[Dict[str, Any]]:
        """Finds an existing active or unread alert for a reminder to prevent duplicate alerts."""
        if not reminder_id:
            return None
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT alert_id FROM screenshot_alerts
                WHERE reminder_id = ? AND alert_type = ? AND status IN ('active', 'read')
                LIMIT 1
                """,
                (reminder_id, alert_type),
            )
            row = cursor.fetchone()
            if row:
                return self.get_alert(row["alert_id"])
        return None

    def list_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        sort: str = "desc",
    ) -> List[Dict[str, Any]]:
        """Lists alerts matching query filters."""
        query = """
            SELECT a.*,
                   s.filename, s.image_uri, s.category, s.intent,
                   r.title as reminder_title, r.reminder_date, r.status as reminder_status
            FROM screenshot_alerts a
            LEFT JOIN screenshot_metadata s ON a.screenshot_id = s.screenshot_id
            LEFT JOIN screenshot_reminders r ON a.reminder_id = r.reminder_id
            WHERE 1=1
        """
        params: List[Any] = []

        if status and status.lower() != "all":
            query += " AND a.status = ?"
            params.append(status.lower())

        if severity and severity.lower() != "all":
            query += " AND a.severity = ?"
            params.append(severity.lower())

        if alert_type and alert_type.lower() != "all":
            query += " AND a.alert_type = ?"
            params.append(alert_type.lower())

        # Order by severity priority (critical > high > medium > low), then trigger_date
        order_dir = "DESC" if sort.lower() == "desc" else "ASC"
        query += f"""
            ORDER BY 
                CASE a.severity
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                    ELSE 5
                END ASC,
                a.trigger_date {order_dir},
                a.created_at {order_dir}
        """

        with self._get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates fields of an existing alert."""
        allowed_fields = {"title", "message", "severity", "status", "trigger_date"}
        set_clauses = []
        params = []
        for k, v in updates.items():
            if k in allowed_fields:
                set_clauses.append(f"{k} = ?")
                params.append(v)

        if not set_clauses:
            return self.get_alert(alert_id)

        set_clauses.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(alert_id)

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE screenshot_alerts SET {', '.join(set_clauses)} WHERE alert_id = ?",
                tuple(params),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None

        return self.get_alert(alert_id)

    def delete_alert(self, alert_id: str) -> bool:
        """Deletes an alert from the database."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM screenshot_alerts WHERE alert_id = ?", (alert_id,))
            conn.commit()
            return cursor.rowcount > 0

    def resolve_alerts_for_reminder(self, reminder_id: str, new_status: str = "resolved") -> int:
        """Resolves or dismisses all active/read alerts linked to a reminder."""
        if not reminder_id:
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE screenshot_alerts
                SET status = ?, updated_at = ?
                WHERE reminder_id = ? AND status IN ('active', 'read')
                """,
                (new_status, now_iso, reminder_id),
            )
            conn.commit()
            return cursor.rowcount

    def get_summary_metrics(self) -> Dict[str, int]:
        """Calculates counts across alert statuses and severity tiers."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status IN ('active', 'read') THEN 1 ELSE 0 END) as unread,
                    SUM(CASE WHEN status = 'active' AND severity = 'critical' THEN 1 ELSE 0 END) as critical,
                    SUM(CASE WHEN status = 'active' AND severity = 'high' THEN 1 ELSE 0 END) as high,
                    SUM(CASE WHEN status = 'active' AND severity = 'medium' THEN 1 ELSE 0 END) as medium,
                    SUM(CASE WHEN status = 'active' AND severity = 'low' THEN 1 ELSE 0 END) as low,
                    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved,
                    SUM(CASE WHEN status = 'dismissed' THEN 1 ELSE 0 END) as dismissed
                FROM screenshot_alerts
                """
            )
            row = cursor.fetchone()
            if not row:
                return {
                    "total": 0, "active": 0, "unread": 0, "critical": 0,
                    "high": 0, "medium": 0, "low": 0, "resolved": 0, "dismissed": 0,
                }
            return {
                "total": int(row["total"] or 0),
                "active": int(row["active"] or 0),
                "unread": int(row["unread"] or 0),
                "critical": int(row["critical"] or 0),
                "high": int(row["high"] or 0),
                "medium": int(row["medium"] or 0),
                "low": int(row["low"] or 0),
                "resolved": int(row["resolved"] or 0),
                "dismissed": int(row["dismissed"] or 0),
            }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Transforms an alert row into a clean dictionary with joined metadata."""
        d = {
            "alert_id": row["alert_id"],
            "reminder_id": row["reminder_id"],
            "screenshot_id": row["screenshot_id"],
            "title": row["title"],
            "message": row["message"],
            "alert_type": row["alert_type"],
            "severity": row["severity"],
            "status": row["status"],
            "trigger_date": row["trigger_date"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

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

        if row["reminder_id"]:
            d["reminder"] = {
                "reminder_id": row["reminder_id"],
                "title": row["reminder_title"],
                "reminder_date": row["reminder_date"],
                "status": row["reminder_status"],
            }
        else:
            d["reminder"] = None

        return d
