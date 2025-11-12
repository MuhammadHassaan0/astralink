import errno
import os
import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

DEFAULT_STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")


def _ensure_storage_dir() -> str:
    base = os.environ.get("ASTRALINK_STORAGE_DIR", DEFAULT_STORAGE_DIR)
    path = os.path.abspath(base)
    try:
        os.makedirs(path, exist_ok=True)
        _probe_path = os.path.join(path, ".write-test")
        with open(_probe_path, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(_probe_path)
        return path
    except OSError as exc:
        if exc.errno not in (errno.EROFS, errno.EPERM, errno.EACCES):
            raise
    fallback = os.path.join(tempfile.gettempdir(), "astralink_storage")
    os.makedirs(fallback, exist_ok=True)
    return fallback


STORAGE_DIR = _ensure_storage_dir()
DB_PATH = os.path.join(STORAGE_DIR, "conversations.db")
_INIT_LOCK = threading.Lock()


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _connect() -> sqlite3.Connection:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables() -> None:
    with _INIT_LOCK:
        conn = _connect()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    convo_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    FOREIGN KEY(convo_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_convo ON messages(convo_id, id)"
            )
        conn.close()


_ensure_tables()


def list_conversations(user_id: str) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.convo_id = c.id
            WHERE c.user_id = ?
            GROUP BY c.id
            ORDER BY datetime(c.updated_at) DESC, datetime(c.created_at) DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_conversation(user_id: str, title: str) -> Dict:
    convo_id = uuid.uuid4().hex
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (convo_id, user_id, (title or "Conversation").strip() or "Conversation", now, now),
        )
    return {
        "id": convo_id,
        "user_id": user_id,
        "title": (title or "Conversation").strip() or "Conversation",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def get_conversation(user_id: str, convo_id: str) -> Optional[Dict]:
    with _connect() as conn:
        convo = conn.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations
            WHERE id = ? AND user_id = ?
            """,
            (convo_id, user_id),
        ).fetchone()
        if not convo:
            return None
        message_rows = conn.execute(
            """
            SELECT role, text, ts
            FROM messages
            WHERE convo_id = ?
            ORDER BY id ASC
            """,
            (convo_id,),
        ).fetchall()
    convo_dict = dict(convo)
    convo_dict["messages"] = [dict(row) for row in message_rows]
    convo_dict["message_count"] = len(message_rows)
    return convo_dict


def append_message(user_id: str, convo_id: str, role: str, text: str, ts: Optional[str] = None) -> bool:
    ts = ts or _now()
    with _connect() as conn:
        convo = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
            (convo_id, user_id),
        ).fetchone()
        if not convo:
            return False
        conn.execute(
            """
            INSERT INTO messages (convo_id, user_id, role, text, ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (convo_id, user_id, role, text, ts),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (ts, convo_id),
        )
    return True


def rename_conversation(user_id: str, convo_id: str, title: str) -> bool:
    clean_title = (title or "").strip()
    if not clean_title:
        return False
    with _connect() as conn:
        result = conn.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (clean_title, _now(), convo_id, user_id),
        )
        return result.rowcount > 0


def conversation_counts(user_id: str) -> Dict:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.convo_id = c.id
            WHERE c.user_id = ?
            GROUP BY c.id
            ORDER BY datetime(c.updated_at) DESC
            """,
            (user_id,),
        ).fetchall()
    conversations = [dict(row) for row in rows]
    total = sum(row.get("message_count", 0) for row in conversations)
    return {"total_messages": total, "conversations": conversations}
