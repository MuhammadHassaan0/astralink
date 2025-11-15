# server.py

import csv
import logging
import os
import traceback
from datetime import datetime
from typing import List

from flask import Flask, request, jsonify, send_from_directory, abort

try:
    from flask_cors import CORS
except ImportError:
    # Fallback: no-op CORS so the app still runs if flask-cors isn't installed
    def CORS(app, *args, **kwargs):
        return app

try:  # Allow running as a package or standalone module.
    from .llm_core import AstralinkCore, ChatGenerationError
    from . import auth_store, conversation_store, llm_core as llm_core_module
except ImportError:  # pragma: no cover
    from llm_core import AstralinkCore, ChatGenerationError  # type: ignore
    import auth_store  # type: ignore
    import conversation_store  # type: ignore
    import llm_core as llm_core_module  # type: ignore


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

core = AstralinkCore()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_session_id(container) -> str | None:
    """Extract session id from any dict-like container."""
    if not container:
        return None
    return container.get("session") or container.get("session_id")


def _frontend_dir() -> str:
    """Absolute path to the frontend folder."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))


def _get_auth_token(extra: dict | None = None) -> str | None:
    token = None
    if extra:
        token = extra.get("auth_token") or extra.get("token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
    if not token:
        token = request.args.get("auth_token")
    return token or None


def _extract_session_id(payload: dict | None = None) -> str | None:
    sid = None
    if payload:
        sid = payload.get("session") or payload.get("session_id")
    if not sid:
        sid = request.headers.get("X-Astralink-Session") or request.headers.get("X-Session-Id")
    if not sid:
        sid = request.args.get("session") or request.args.get("session_id")
    if not sid:
        sid = request.form.get("session") or request.form.get("session_id")
    return sid


def _ensure_session_id(sid: str | None) -> str:
    if sid and sid in core.sessions:
        return sid
    new_sid, _ = core.new_session({})
    return new_sid


def _resolve_user_context(require_auth: bool = False) -> dict:
    payload = request.get_json(silent=True)
    token = _get_auth_token(payload if isinstance(payload, dict) else None)
    email = auth_store.get_email_for_token(token) if token else None
    sid = _extract_session_id(payload if isinstance(payload, dict) else None)

    if email:
        resolved_sid = _ensure_session_for_email(email)
        return {
            "user_key": email,
            "email": email,
            "session_id": resolved_sid,
            "is_authenticated": True,
        }

    sid = _ensure_session_id(sid)
    if require_auth:
        resp = jsonify({"ok": False, "error": "not_authenticated"})
        resp.status_code = 401
        abort(resp)
    return {
        "user_key": f"guest:{sid}",
        "email": None,
        "session_id": sid,
        "is_authenticated": False,
    }


def _require_user_email() -> str:
    data = request.get_json(silent=True)
    token = _get_auth_token(data if isinstance(data, dict) else None)
    email = auth_store.get_email_for_token(token)
    if not email:
        session_id = None
        if isinstance(data, dict):
            session_id = data.get("session") or data.get("session_id")
        if not session_id:
            session_id = request.headers.get("X-Astralink-Session") or request.headers.get("X-Session-Id")
        if not session_id:
            session_id = (
                request.args.get("session")
                or request.args.get("session_id")
                or request.form.get("session")
                or request.form.get("session_id")
            )
        email = auth_store.get_email_for_session(session_id)
    if not email:
        resp = jsonify({"ok": False, "error": "not_authenticated"})
        resp.status_code = 401
        abort(resp)
    return email


def _ensure_session_for_email(email: str) -> str:
    profile = auth_store.get_user_profile(email) or {}
    sid = auth_store.get_user_session(email)
    if not sid or sid not in core.sessions:
        sid, _ = core.new_session(profile)
    else:
        if profile:
            core.save_profile(sid, profile)
    auth_store.set_user_session(email, sid)
    return sid


def _default_convo_title(email: str) -> str:
    profile = auth_store.get_user_profile(email) or {}
    loved_name = profile.get("name") or "them"
    return f"Chat with {loved_name}".strip()


def _resolve_session(container=None, data=None) -> tuple[str | None, str | None]:
    token = _get_auth_token(data)
    email = auth_store.get_email_for_token(token) if token else None
    if email:
        sid = auth_store.get_user_session(email)
        if not sid or sid not in core.sessions:
            sid, _ = core.new_session({})
            auth_store.set_user_session(email, sid)
        return sid, email
    sid = _get_session_id(container) or _get_session_id(data or {}) or _get_session_id(request.args)
    return sid, None


def _rows_from_csv(text: str) -> List[str]:
    rows = []
    reader = csv.reader(text.splitlines())
    for row in reader:
        joined = " ".join(col.strip() for col in row if col and col.strip())
        if not joined:
            continue
        if joined.lower() == "memory":
            continue
        rows.append(joined)
    return rows


def _memories_from_upload(file_storage) -> List[str]:
    if not file_storage:
        return []
    try:
        raw = file_storage.read()
    except Exception:
        return []
    if not raw:
        return []
    text = raw.decode("utf-8", errors="ignore")
    name = (file_storage.filename or "").lower()
    if name.endswith(".csv"):
        rows = _rows_from_csv(text)
        if rows:
            return rows
    return [line.strip() for line in text.splitlines() if line.strip()]


# -----------------------------------------------------------------------------
# Frontend routes
# -----------------------------------------------------------------------------

def _serve_frontend(page: str):
    return send_from_directory(_frontend_dir(), page)


@app.route("/")
def serve_index():
    return _serve_frontend("index.html")


@app.route("/beta")
def serve_beta_index():
    return _serve_frontend("index.html")


@app.route("/how")
def serve_how():
    return _serve_frontend("how.html")


@app.route("/beta/how")
def serve_beta_how():
    return _serve_frontend("how.html")


@app.route("/interview")
def serve_interview():
    return _serve_frontend("interview.html")


@app.route("/beta/interview")
def serve_beta_interview():
    return _serve_frontend("interview.html")


@app.route("/chat")
def serve_chat():
    return _serve_frontend("chat.html")


@app.route("/beta/chat")
def serve_beta_chat():
    return _serve_frontend("chat.html")


@app.route("/pay")
def serve_pay():
    return _serve_frontend("pay.html")


@app.route("/beta/pay")
def serve_beta_pay():
    return _serve_frontend("pay.html")


@app.route("/profile")
def serve_profile():
    return _serve_frontend("profile.html")


@app.route("/beta/profile")
def serve_beta_profile():
    return _serve_frontend("profile.html")


@app.route("/memories")
def serve_memories():
    return _serve_frontend("memories.html")


@app.route("/beta/memories")
def serve_beta_memories():
    return _serve_frontend("memories.html")


@app.route("/auth")
@app.route("/auth.html")
def serve_auth():
    return _serve_frontend("auth.html")


@app.route("/beta/auth")
def serve_beta_auth():
    return _serve_frontend("auth.html")


@app.route("/<path:path>")
def serve_static(path: str):
    """
    Serve other frontend assets.
    Do NOT eat /api/* routes.
    """
    if path.startswith("api/"):
        abort(404)
    return send_from_directory(_frontend_dir(), path)


@app.route("/beta/<path:path>")
def serve_beta_static(path: str):
    if path.startswith("api/"):
        abort(404)
    return send_from_directory(_frontend_dir(), path)


# -----------------------------------------------------------------------------
# API: auth
# -----------------------------------------------------------------------------


@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.get_json() or {}
    full_name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    ok, msg = auth_store.create_user(email, full_name, password)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400

    sid, _ = core.new_session({})
    auth_store.set_user_session(email, sid)
    token = auth_store.create_token(email)
    return jsonify({
        "ok": True,
        "auth_token": token,
        "session_id": sid,
        "profile": {},
        "name": full_name,
    })


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not auth_store.verify_credentials(email, password):
        return jsonify({"ok": False, "error": "Invalid email or password."}), 400

    profile = auth_store.get_user_profile(email)
    sid = auth_store.get_user_session(email)
    if not sid or sid not in core.sessions:
        sid, _ = core.new_session(profile or {})
    else:
        if profile:
            core.save_profile(sid, profile)
    auth_store.set_user_session(email, sid)
    token = auth_store.create_token(email)
    return jsonify({
        "ok": True,
        "auth_token": token,
        "session_id": sid,
        "profile": profile,
        "name": auth_store.get_user_display_name(email),
    })


@app.route("/api/logout", methods=["POST"])
def api_logout():
    token = _get_auth_token(request.get_json(silent=True) or {})
    if token:
        auth_store.revoke_token(token)
    return jsonify({"ok": True})


# -----------------------------------------------------------------------------
# API: profile & memories
# -----------------------------------------------------------------------------

@app.route("/api/save_profile", methods=["POST"])
def api_save_profile():
    data = request.get_json() or {}
    sid, email = _resolve_session(data=data)
    profile = data.get("profile", {}) or {}

    # Create new session if none/invalid
    if not sid or sid not in core.sessions:
        sid, credits = core.new_session(profile)
        created = True
    else:
        core.save_profile(sid, profile)
        credits = core.sessions[sid]["credits"]
        created = False

        credits = core.sessions[sid]["credits"]
        created = False

    if email:
        auth_store.set_user_session(email, sid)
        auth_store.save_user_profile(email, profile)

    return jsonify({
        "ok": True,
        "session": sid,
        "session_id": sid,
        "credits": credits,
        "created": created,
        "profile": profile,
    })


@app.route("/api/upload_memories", methods=["POST"])
def api_upload_memories():
    """
    Accepts:
      JSON: { session, text, source }
      or form-data: text/note, [source], optional file(s)
    """
    if request.is_json:
        data = request.get_json() or {}
        sid, _ = _resolve_session(data=data)
        text = (data.get("text") or data.get("note") or "").strip()
        source = data.get("source")
        files = []
    else:
        form = request.form
        sid, _ = _resolve_session(container=form)
        text = (form.get("text") or form.get("note") or "").strip()
        source = form.get("source")
        files = request.files.getlist("files") or []
        if not files and "file" in request.files:
            files = [request.files["file"]]

    # Ensure we have a session
    if not sid or sid not in core.sessions:
        sid, _ = core.new_session({})

    saved_labels: List[str] = []
    saved_any = 0

    if text:
        core.add_text_memory(sid, text, source or "note")
        core.add_memory_chunk(sid, text)
        saved_labels.append("text")
        saved_any += 1

    for f in files:
        items = _memories_from_upload(f)
        if not items:
            continue
        filename = f.filename or "file"
        for chunk in items:
            core.add_text_memory(sid, chunk, source or filename)
            core.add_memory_chunk(sid, chunk)
            saved_any += 1
        saved_labels.append(filename)

    if saved_any == 0:
        return jsonify({"ok": False, "error": "No text or files provided"}), 400

    return jsonify({
        "ok": True,
        "session": sid,
        "session_id": sid,
        "saved": saved_labels,
        "count": saved_any,
    })


# -----------------------------------------------------------------------------
# API: chat
# -----------------------------------------------------------------------------

@app.post("/chat")
@app.post("/api/chat")
def api_chat():
    data = request.get_json() or {}
    sid, _ = _resolve_session(data=data)
    message = (data.get("message") or "").strip()

    if not message:
        resp = jsonify({"ok": False, "error": "Empty message"})
        resp.status_code = 400
        resp.headers["X-Astralink-Fallback"] = "false"
        return resp

    # Ensure session exists (allows cold-start chat)
    if not sid or sid not in core.sessions:
        sid, _ = core.new_session({})

    session_obj = core.get_session(sid)
    history = session_obj["messages"] if session_obj else []
    try:
        reply_text, used_fallback, error_summary, model_used = core.generate_reply(
            sid=sid,
            text=message,
            history_override=history,
        )
    except ChatGenerationError as exc:
        print(f"CHAT: EXCEPTION -> {exc}")
        resp = jsonify({"error": str(exc), "fallback": False})
        resp.status_code = 502
        resp.headers["X-Astralink-Fallback"] = "false"
        resp.headers["X-Astralink-Model-Used"] = ""
        return resp

    if session_obj is not None:
        ts = datetime.utcnow().isoformat() + "Z"
        session_obj["messages"].append({"role": "user", "content": message, "ts": ts})
        session_obj["messages"].append({"role": "assistant", "content": reply_text, "ts": datetime.utcnow().isoformat() + "Z"})

    body = {
        "ok": True,
        "session": sid,
        "session_id": sid,
        "reply": reply_text,
        "model_used": model_used,
        "fallback": used_fallback,
    }
    if used_fallback and error_summary:
        body["error"] = error_summary
    resp = jsonify(body)
    resp.headers["X-Astralink-Fallback"] = "true" if used_fallback else "false"
    resp.headers["X-Astralink-Model-Used"] = model_used
    return resp


# -----------------------------------------------------------------------------
# API: conversation threads
# -----------------------------------------------------------------------------


@app.route("/api/conversations", methods=["GET"])
def api_list_conversations():
    ctx = _resolve_user_context()
    convos = conversation_store.list_conversations(ctx["user_key"])
    return jsonify({"ok": True, "session_id": ctx["session_id"], "conversations": convos})


@app.route("/api/conversations", methods=["POST"])
def api_create_conversation():
    ctx = _resolve_user_context()
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        title = _default_convo_title(ctx["email"] or "")
    convo = conversation_store.create_conversation(ctx["user_key"], title)
    return jsonify({"ok": True, "session_id": ctx["session_id"], "conversation": convo})


@app.route("/api/conversations/<convo_id>", methods=["GET"])
def api_get_conversation(convo_id: str):
    ctx = _resolve_user_context()
    convo = conversation_store.get_conversation(ctx["user_key"], convo_id)
    if not convo:
        resp = jsonify({"ok": False, "error": "conversation_not_found"})
        resp.status_code = 404
        resp.headers["X-Astralink-Fallback"] = "false"
        return resp
    return jsonify({"ok": True, "session_id": ctx["session_id"], "conversation": convo})


@app.route("/api/conversations/<convo_id>/message", methods=["POST"])
def api_conversation_message(convo_id: str):
    ctx = _resolve_user_context()
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        resp = jsonify({"ok": False, "error": "empty_message"})
        resp.status_code = 400
        resp.headers["X-Astralink-Fallback"] = "false"
        return resp

    convo = conversation_store.get_conversation(ctx["user_key"], convo_id)
    if not convo:
        resp = jsonify({"ok": False, "error": "conversation_not_found"})
        resp.status_code = 404
        resp.headers["X-Astralink-Fallback"] = "false"
        return resp

    history = (convo.get("messages") or [])[:]
    if not conversation_store.append_message(ctx["user_key"], convo_id, "user", text):
        resp = jsonify({"ok": False, "error": "conversation_not_found"})
        resp.status_code = 404
        resp.headers["X-Astralink-Fallback"] = "false"
        return resp

    sid = ctx["session_id"]
    if ctx["email"]:
        sid = _ensure_session_for_email(ctx["email"])
    try:
        reply_text, used_fallback, error_summary, model_used = core.generate_reply(
            sid=sid,
            text=text,
            history_override=history,
        )
    except ChatGenerationError as exc:
        print(f"CHAT: EXCEPTION -> {exc}")
        resp = jsonify({"error": str(exc), "fallback": False})
        resp.status_code = 502
        resp.headers["X-Astralink-Fallback"] = "false"
        resp.headers["X-Astralink-Model-Used"] = ""
        return resp

    if not conversation_store.append_message(ctx["user_key"], convo_id, "assistant", reply_text):
        resp = jsonify({"ok": False, "error": "conversation_not_found"})
        resp.status_code = 404
        resp.headers["X-Astralink-Fallback"] = "true" if used_fallback else "false"
        resp.headers["X-Astralink-Model-Used"] = model_used
        return resp

    updated = conversation_store.get_conversation(ctx["user_key"], convo_id) or convo
    body = {
        "ok": True,
        "reply": reply_text,
        "session_id": sid,
        "messages": updated.get("messages", []),
        "conversation": {
            "id": updated.get("id", convo_id),
            "title": updated.get("title", convo.get("title")),
            "updated_at": updated.get("updated_at"),
            "created_at": updated.get("created_at"),
            "message_count": updated.get("message_count", len(updated.get("messages", []))),
        },
        "model_used": model_used,
        "fallback": used_fallback,
    }
    if used_fallback and error_summary:
        body["error"] = error_summary
    resp = jsonify(body)
    resp.headers["X-Astralink-Fallback"] = "true" if used_fallback else "false"
    resp.headers["X-Astralink-Model-Used"] = model_used
    return resp


@app.post("/conversations/<conv_id>/message")
@app.post("/api/conversations/<conv_id>/message")
def conversations_message(conv_id: str):
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        resp = jsonify({"error": "missing text"})
        resp.status_code = 400
        resp.headers["X-Astralink-Fallback"] = "false"
        return resp

    ctx = _resolve_user_context()
    sid = core.ensure_session(ctx.get("session_id"))
    ctx["session_id"] = sid
    print(f"CONV {conv_id}: received user text len={len(text)}")

    core.get_or_create_conversation(sid, conv_id)
    core.append_message(sid, conv_id, role="user", text=text)

    try:
        reply_text, used_fallback, error_summary, model_used = core.generate_reply(
            sid=sid,
            text=text,
            conversation_id=conv_id,
        )
    except ChatGenerationError as exc:
        print(f"CHAT: EXCEPTION -> {exc}")
        resp = jsonify({"error": str(exc), "fallback": False})
        resp.status_code = 502
        resp.headers["X-Astralink-Fallback"] = "false"
        resp.headers["X-Astralink-Model-Used"] = ""
        return resp

    core.append_message(sid, conv_id, role="assistant", text=reply_text)

    messages = core.get_messages(sid, conv_id, limit=50)
    body = {
        "reply": reply_text,
        "conversation_id": conv_id,
        "messages": messages,
        "model_used": model_used,
        "fallback": used_fallback,
    }
    if used_fallback and error_summary:
        body["error"] = error_summary
    resp = jsonify(body)
    resp.headers["X-Astralink-Fallback"] = "true" if used_fallback else "false"
    resp.headers["X-Astralink-Model-Used"] = model_used
    return resp


@app.route("/api/conversations/<convo_id>/title", methods=["POST"])
def api_conversation_rename(convo_id: str):
    ctx = _resolve_user_context()
    data = request.get_json() or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"ok": False, "error": "Title required"}), 400
    ok = conversation_store.rename_conversation(ctx["user_key"], convo_id, title)
    if not ok:
        return jsonify({"ok": False, "error": "conversation_not_found"}), 404
    convo = conversation_store.get_conversation(ctx["user_key"], convo_id)
    return jsonify({"ok": True, "session_id": ctx["session_id"], "conversation": convo})


@app.route("/api/conversations/stats", methods=["GET"])
def api_conversation_stats():
    ctx = _resolve_user_context()
    stats = conversation_store.conversation_counts(ctx["user_key"])
    stats["session_id"] = ctx["session_id"]
    stats["ok"] = True
    return jsonify(stats)


# -----------------------------------------------------------------------------
# API: interview flow
# -----------------------------------------------------------------------------

@app.route("/api/interview/start", methods=["POST"])
def api_interview_start():
    data = request.get_json() or {}
    sid, _ = _resolve_session(data=data)

    # Ensure session
    if not sid or sid not in core.sessions:
        sid, _ = core.new_session({})

    ok, first_q = core.start_interview(sid)

    if not ok:
        return jsonify({
            "ok": False,
            "session": sid,
            "session_id": sid,
            "error": "Interview start failed",
        }), 400

    return jsonify({
        "ok": True,
        "session": sid,
        "session_id": sid,
        "question": first_q,
    })


@app.route("/api/interview/answer", methods=["POST"])
def api_interview_answer():
    data = request.get_json() or {}
    sid, _ = _resolve_session(data=data)
    answer = (data.get("answer") or "").strip()

    if not sid or sid not in core.interviews:
        return jsonify({"ok": False, "error": "Invalid session"}), 400
    if not answer:
        return jsonify({"ok": False, "error": "Empty answer"}), 400

    ok, next_q_or_summary = core.answer_interview(sid, answer)
    if not ok:
        return jsonify({"ok": False, "error": next_q_or_summary}), 400

    # When interview is done, core.answer_interview returns the final summary.
    payload = {"ok": True, "session": sid, "session_id": sid}
    finished = sid not in core.interviews
    if finished:
        payload["done"] = True
        payload["summary"] = next_q_or_summary
    else:
        payload["done"] = False
        payload["next_question"] = next_q_or_summary

    return jsonify(payload)


@app.get("/diag/openai")
@app.get("/api/diag/openai")
def api_diag_openai():
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    offline = os.environ.get("ASTRALINK_OFFLINE", "")
    out = {
        "has_key": has_key,
        "model": llm_core_module.configured_model(),
        "fallback": llm_core_module.fallback_model(),
        "offline": offline,
    }
    try:
        client = llm_core_module._openai_client()
    except Exception as exc:
        logging.error("diag_openai init failed:\n%s", traceback.format_exc())
        out.update({
            "can_call": False,
            "error_code": "init_failed",
            "error": f"{exc.__class__.__name__}: {exc}",
        })
        return jsonify(out)

    try:
        resp = client.chat.completions.create(
            model=out["model"],
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=1,
        )
        _ = resp.choices[0].message.content
        out.update({"can_call": True, "model_used": out["model"]})
    except Exception as exc:
        logging.error("diag_openai call failed:\n%s", traceback.format_exc())
        msg = f"{exc.__class__.__name__}: {exc}"
        lower = msg.lower()
        if "does not exist" in lower or "not found" in lower or "unknown model" in lower:
            code = "bad_model"
        elif "invalid api key" in lower or "unauthorized" in lower or "401" in lower:
            code = "bad_key"
        else:
            code = "call_failed"
        out.update({"can_call": False, "error_code": code, "error": msg})
    return jsonify(out)


@app.route("/api/interview/transcribe", methods=["POST"])
def api_interview_transcribe():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"ok": False, "error": "audio_file_required"}), 400
    try:
        blob = audio.read()
    except Exception:
        blob = None
    if not blob:
        return jsonify({"ok": False, "error": "empty_audio"}), 400
    try:
        transcript = core.transcribe_audio(blob, audio.filename or "voice.webm")
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    sid, _ = _resolve_session(container=request.form)
    response = {"ok": True, "transcript": transcript}
    if sid:
        response["session_id"] = sid
    return jsonify(response)


# -----------------------------------------------------------------------------
# API: list memories (debug / helper)
# -----------------------------------------------------------------------------

@app.route("/api/list_memories", methods=["GET"])
def api_list_memories():
    sid, _ = _resolve_session(container=request.args)

    if not sid or sid not in core.sessions:
        return jsonify({"ok": True, "session": None, "memories": []})

    mems = core.list_memories(sid)

    return jsonify({
        "ok": True,
        "session": sid,
        "session_id": sid,
        "memories": mems,
    })


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Use the same port your frontend expects (17680)
    app.run(host="127.0.0.1", port=17680, debug=True)
