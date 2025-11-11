# astralink/backend/server.py
import os
from uuid import uuid4

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from llm_core import AstralinkCore

# === Serve the Stitch-style static files and expose JSON APIs ===

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONT = os.path.join(ROOT, "frontend")

app = Flask(__name__, static_folder=FRONT, static_url_path="/")
CORS(app, resources={r"/api/*": {"origins": "*"}})

core = AstralinkCore()

# -------- Static pages (if you have multiple .html files) --------
@app.get("/")
def index():
    return send_from_directory(FRONT, "index.html")

@app.get("/how")
def how():
    return send_from_directory(FRONT, "how.html") if os.path.isfile(os.path.join(FRONT, "how.html")) else send_from_directory(FRONT, "index.html")

@app.get("/interview")
def interview():
    return send_from_directory(FRONT, "interview.html") if os.path.isfile(os.path.join(FRONT, "interview.html")) else send_from_directory(FRONT, "index.html")

@app.get("/chat")
def chat_page():
    return send_from_directory(FRONT, "chat.html") if os.path.isfile(os.path.join(FRONT, "chat.html")) else send_from_directory(FRONT, "index.html")

@app.get("/pay")
def pay_page():
    return send_from_directory(FRONT, "pay.html") if os.path.isfile(os.path.join(FRONT, "pay.html")) else send_from_directory(FRONT, "index.html")

# Let css/js/images be fetched directly if present
@app.get("/<path:path>")
def static_proxy(path):
    full = os.path.join(FRONT, path)
    if os.path.isfile(full):
        return send_from_directory(FRONT, path)
    return send_from_directory(FRONT, "index.html")

# ------------------------- JSON APIs ----------------------------

@app.post("/api/build")
def api_build():
    """
    Build = create a session from the profile sent by the front-end.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        profile = {
            "name": data.get("name", ""),
            "relationship": data.get("relationship", ""),
            "call_you": data.get("call_you", ""),
            "traits": data.get("traits", []) or [],
            "catchphrases": data.get("catchphrases", []) or [],
            "mode": _map_mode(data.get("mode")),
        }

        # helpful debug
        print("\n[api_build] Incoming profile:", profile)

        session_id, credits = core.new_session(profile)
        print("[api_build] Created session:", session_id, "credits:", credits)
        return jsonify({"session_id": session_id, "credits": credits})

    except Exception as e:
        import traceback
        print("\n[api_build] ERROR:\n", traceback.format_exc())
        return jsonify({"error": f"save failed: {e}"}), 500

@app.post("/api/save_profile")   # legacy alias
def api_save_profile():
    return api_build()

@app.post("/api/chat")
def api_chat():
    try:
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id", "")
        msg = (data.get("message") or "").strip()
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400
        if not msg:
            return jsonify({"error": "Empty message"}), 400

        reply, credits = core.chat(session_id, msg)
        return jsonify({"reply": reply, "credits": credits})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/api/end")
def api_end():
    try:
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id", "")
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400
        reflection = core.end_session(session_id)
        return jsonify({"reflection": reflection})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ------------------------ helpers -------------------------------

def _map_mode(val: str) -> str:
    s = (val or "").strip().lower()
    # Accept the exact labels used in your dropdown and convert to canonical values
    if "passed" in s or "memory" in s:
        return "memory"
    if "alive" in s or "real-time" in s or "real time" in s:
        return "alive"
    return "memory"


# ------------------------ run -----------------------------------

if __name__ == "__main__":
    # Flask dev server — good enough for local testing
    app.run(host="0.0.0.0", port=7860, debug=True)
