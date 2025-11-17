import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:
    psycopg = None  # type: ignore
    dict_row = None  # type: ignore

DB_URL = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or os.environ.get("SUPABASE_URL")
_USE_PG = bool(DB_URL and psycopg is not None)
_STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
_SQLITE_PATH = os.path.join(_STORAGE_DIR, "conversations.db")
_INIT_LOCK = threading.Lock()


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _connect_pg():
    return psycopg.connect(DB_URL, autocommit=True, row_factory=dict_row)  # type: ignore[arg-type]


def _connect_sqlite():
    os.makedirs(_STORAGE_DIR, exist_ok=True)
    conn = sqlite3.connect(_SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _connect():
    if _USE_PG:
        return _connect_pg()
    return _connect_sqlite()


def _ph(count: int) -> str:
    token = "%s" if _USE_PG else "?"
    return ", ".join([token] * count)


def _ensure_tables() -> None:
    with _INIT_LOCK:
        conn = _connect()
        cur = conn.cursor()
        if _USE_PG:
            cur.execute(
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    convo_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    CONSTRAINT fk_convo FOREIGN KEY(convo_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_convo ON messages(convo_id, id)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_stats (
                    user_id TEXT PRIMARY KEY,
                    messages_sent BIGINT DEFAULT 0,
                    messages_received BIGINT DEFAULT 0,
                    conversations_started BIGINT DEFAULT 0,
                    last_seen TEXT,
                    last_conversation_id TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        else:
            cur.execute(
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
            cur.execute(
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
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_convo ON messages(convo_id, id)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_stats (
                    user_id TEXT PRIMARY KEY,
                    messages_sent INTEGER DEFAULT 0,
                    messages_received INTEGER DEFAULT 0,
                    conversations_started INTEGER DEFAULT 0,
                    last_seen TEXT,
                    last_conversation_id TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        conn.commit()
        conn.close()


_ensure_tables()


def _bump_user_stats(
    cur,
    user_id: str,
    messages_sent: int = 0,
    messages_received: int = 0,
    conversations_started: int = 0,
    last_conversation_id: Optional[str] = None,
    ts: Optional[str] = None,
) -> None:
    ts = ts or _now()
    cur.execute(
        f"""
        INSERT INTO user_stats (user_id, messages_sent, messages_received, conversations_started, last_seen, last_conversation_id, created_at)
        VALUES ({_ph(7)})
        ON CONFLICT (user_id) DO UPDATE SET
            messages_sent = user_stats.messages_sent + {_ph(1)},
            messages_received = user_stats.messages_received + {_ph(1)},
            conversations_started = user_stats.conversations_started + {_ph(1)},
            last_seen = {_ph(1)},
            last_conversation_id = {_ph(1)}
        """,
        (
            user_id,
            messages_sent,
            messages_received,
            conversations_started,
            ts,
            last_conversation_id or "",
            ts,
            messages_sent,
            messages_received,
            conversations_started,
            ts,
            last_conversation_id or "",
        ),
    )


def user_stats(user_id: str) -> Dict[str, Optional[str]]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT user_id, messages_sent, messages_received, conversations_started, last_seen, last_conversation_id, created_at
            FROM user_stats
            WHERE user_id = {_ph(1)}
            """,
            (user_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else {}


def user_stats_all(limit: int = 200) -> List[Dict[str, Optional[str]]]:
    sql = f"""
        SELECT user_id, messages_sent, messages_received, conversations_started, last_seen, last_conversation_id, created_at
        FROM user_stats
        ORDER BY COALESCE(last_seen, created_at) DESC
        LIMIT {_ph(1)}
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def get_messages(convo_id: str, limit: int = 500) -> List[Dict[str, str]]:
    sql = f"""
        SELECT role, text, ts
        FROM messages
        WHERE convo_id = {_ph(1)}
        ORDER BY id ASC
        LIMIT {_ph(1)}
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, (convo_id, limit))
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def list_conversations(user_id: str) -> List[Dict]:
    sql = f"""
        SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count
        FROM conversations c
        LEFT JOIN messages m ON m.convo_id = c.id
        WHERE c.user_id = {_ph(1)}
        GROUP BY c.id
        ORDER BY c.updated_at DESC, c.created_at DESC
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, (user_id,))
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def create_conversation(user_id: str, title: str) -> Dict:
    convo_id = uuid.uuid4().hex
    now = _now()
    clean_title = (title or "Conversation").strip() or "Conversation"
    sql = f"""
        INSERT INTO conversations (id, user_id, title, created_at, updated_at)
        VALUES ({_ph(5)})
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, (convo_id, user_id, clean_title, now, now))
        _bump_user_stats(cur, user_id, conversations_started=1, last_conversation_id=convo_id, ts=now)
        conn.commit()
    return {
        "id": convo_id,
        "user_id": user_id,
        "title": clean_title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def get_conversation(user_id: str, convo_id: str) -> Optional[Dict]:
    convo_sql = f"""
        SELECT id, user_id, title, created_at, updated_at
        FROM conversations
        WHERE id = {_ph(1)} AND user_id = {_ph(1)}
    """
    messages_sql = f"""
        SELECT role, text, ts
        FROM messages
        WHERE convo_id = {_ph(1)}
        ORDER BY id ASC
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(convo_sql, (convo_id, user_id))
        convo = cur.fetchone()
        if not convo:
            return None
        cur.execute(messages_sql, (convo_id,))
        msgs = cur.fetchall()
    convo_dict = dict(convo)
    convo_dict["messages"] = [dict(m) for m in msgs]
    convo_dict["message_count"] = len(msgs)
    return convo_dict


def append_message(user_id: str, convo_id: str, role: str, text: str, ts: Optional[str] = None) -> bool:
    ts = ts or _now()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT 1 FROM conversations WHERE id = {_ph(1)} AND user_id = {_ph(1)}",
            (convo_id, user_id),
        )
        if not cur.fetchone():
            return False
        cur.execute(
            f"""
            INSERT INTO messages (convo_id, user_id, role, text, ts)
            VALUES ({_ph(5)})
            """,
            (convo_id, user_id, role, text, ts),
        )
        cur.execute(
            f"UPDATE conversations SET updated_at = {_ph(1)} WHERE id = {_ph(1)}",
            (ts, convo_id),
        )
        if role == "user":
            _bump_user_stats(cur, user_id, messages_sent=1, last_conversation_id=convo_id, ts=ts)
        elif role == "assistant":
            _bump_user_stats(cur, user_id, messages_received=1, last_conversation_id=convo_id, ts=ts)
        conn.commit()
    return True


def rename_conversation(user_id: str, convo_id: str, title: str) -> bool:
    clean_title = (title or "").strip()
    if not clean_title:
        return False
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            UPDATE conversations
            SET title = {_ph(1)}, updated_at = {_ph(1)}
            WHERE id = {_ph(1)} AND user_id = {_ph(1)}
            """,
            (clean_title, _now(), convo_id, user_id),
        )
        conn.commit()
        return bool(cur.rowcount)


def conversation_counts(user_id: str) -> Dict:
    sql = f"""
        SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count
        FROM conversations c
        LEFT JOIN messages m ON m.convo_id = c.id
        WHERE c.user_id = {_ph(1)}
        GROUP BY c.id
        ORDER BY c.updated_at DESC
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, (user_id,))
        rows = cur.fetchall()
    conversations = [dict(row) for row in rows]
    total = sum(row.get("message_count", 0) for row in conversations)
    return {"total_messages": total, "conversations": conversations}
