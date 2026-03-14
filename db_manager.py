# db_manager.py — SQLite persistence for chat history and feedback

import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = "chat_history.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                metadata    TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                message_id  INTEGER NOT NULL,
                rating      INTEGER NOT NULL,
                timestamp   TEXT    NOT NULL
            );
        """)


def save_message(session_id: str, role: str, content: str, metadata: dict = None) -> int:
    """
    Persist a chat message.

    Returns:
        int: The auto-assigned row id (used for linking feedback).
    """
    ts = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata) if metadata else None
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, ts, meta_json),
        )
        return cur.lastrowid


def load_messages(session_id: str) -> list:
    """
    Load all messages for a session in chronological order.

    Returns:
        list[dict]: Each dict has keys: id, role, content, timestamp, metadata (dict or None).
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, timestamp, metadata FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()

    result = []
    for row in rows:
        meta = json.loads(row["metadata"]) if row["metadata"] else None
        result.append({
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
            "metadata": meta,
        })
    return result


def save_feedback(session_id: str, message_id: int, rating: int):
    """
    Store thumbs up (+1) or thumbs down (-1) for a message.
    Replaces any existing rating for the same message.
    """
    ts = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        # Remove any previous rating for this message in this session
        conn.execute(
            "DELETE FROM feedback WHERE session_id = ? AND message_id = ?",
            (session_id, message_id),
        )
        conn.execute(
            "INSERT INTO feedback (session_id, message_id, rating, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, message_id, rating, ts),
        )


def clear_session(session_id: str):
    """Delete all messages and feedback for a session."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM feedback WHERE session_id = ?", (session_id,))
