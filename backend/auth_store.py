import errno
import hashlib
import json
import os
import secrets
import tempfile
import threading
from typing import Dict, Optional, Tuple


DEFAULT_USERS_PATH = os.path.join(os.path.dirname(__file__), "users.json")


def _resolve_users_path() -> str:
    base = os.environ.get("ASTRALINK_USERS_PATH", DEFAULT_USERS_PATH)
    path = os.path.abspath(base)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8"):
            pass
        return path
    except OSError as exc:
        if exc.errno not in (errno.EROFS, errno.EPERM, errno.EACCES):
            raise
    tmp_dir = os.path.join(tempfile.gettempdir(), "astralink_users")
    os.makedirs(tmp_dir, exist_ok=True)
    fallback = os.path.join(tmp_dir, "users.json")
    if not os.path.exists(fallback):
        with open(fallback, "w", encoding="utf-8") as fh:
            fh.write("{}")
    return fallback


_USERS_PATH = _resolve_users_path()
_token_lock = threading.Lock()
_users_lock = threading.Lock()
_active_tokens: Dict[str, str] = {}


def _load_users() -> Dict[str, Dict]:
    with _users_lock:
        if not os.path.exists(_USERS_PATH):
            return {}
        try:
            with open(_USERS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}


def _save_users(users: Dict[str, Dict]) -> None:
    with _users_lock:
        os.makedirs(os.path.dirname(_USERS_PATH), exist_ok=True)
        with open(_USERS_PATH, "w", encoding="utf-8") as fh:
            json.dump(users, fh, indent=2)


def _hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 390000)
    return digest.hex()


def create_user(email: str, full_name: str, password: str) -> Tuple[bool, str]:
    email = (email or "").strip().lower()
    if not email or not password or not full_name.strip():
        return False, "Missing required fields."
    users = _load_users()
    if email in users:
        return False, "An account with this email already exists."
    salt = secrets.token_hex(16)
    pwd = _hash_password(password, salt)
    users[email] = {
        "name": full_name.strip(),
        "salt": salt,
        "password": pwd,
        "session_id": None,
        "profile": {},
    }
    _save_users(users)
    return True, "Account created."


def verify_credentials(email: str, password: str) -> bool:
    email = (email or "").strip().lower()
    users = _load_users()
    user = users.get(email)
    if not user:
        return False
    expected = user.get("password")
    salt = user.get("salt")
    if not expected or not salt:
        return False
    calc = _hash_password(password, salt)
    return secrets.compare_digest(calc, expected)


def create_token(email: str) -> str:
    token = secrets.token_hex(32)
    with _token_lock:
        _active_tokens[token] = email
    return token


def revoke_token(token: str) -> None:
    with _token_lock:
        _active_tokens.pop(token, None)


def get_email_for_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    with _token_lock:
        return _active_tokens.get(token)


def set_user_session(email: str, session_id: str) -> None:
    email = (email or "").strip().lower()
    users = _load_users()
    if email not in users:
        return
    users[email]["session_id"] = session_id
    _save_users(users)


def get_user_session(email: str) -> Optional[str]:
    email = (email or "").strip().lower()
    users = _load_users()
    user = users.get(email)
    if not user:
        return None
    return user.get("session_id")


def save_user_profile(email: str, profile: Dict) -> None:
    email = (email or "").strip().lower()
    users = _load_users()
    if email not in users:
        return
    users[email]["profile"] = profile or {}
    _save_users(users)


def get_user_profile(email: str) -> Dict:
    email = (email or "").strip().lower()
    users = _load_users()
    user = users.get(email)
    if not user:
        return {}
    return user.get("profile") or {}


def get_user_display_name(email: str) -> Optional[str]:
    email = (email or "").strip().lower()
    users = _load_users()
    user = users.get(email)
    if not user:
        return None
    return user.get("name")


def get_email_for_session(session_id: Optional[str]) -> Optional[str]:
    session_id = (session_id or "").strip()
    if not session_id:
        return None
    users = _load_users()
    for email, info in users.items():
        if (info or {}).get("session_id") == session_id:
            return email
    return None
