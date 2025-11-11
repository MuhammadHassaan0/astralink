# astralink/backend/llm_core.py
import os
from uuid import uuid4
from typing import Dict, List, Tuple

try:
    # OpenAI SDK v1+
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False
    OpenAI = None  # type: ignore


def _trim(s: str) -> str:
    return (s or "").strip()


def _csv_to_list(s: str) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


class AstralinkCore:
    """
    Minimal in-memory core for Astralink sessions.
    - create_session/new_session(profile) -> (session_id, credits)
    - chat(session_id, message) -> (reply, credits_left)
    - end_session(session_id) -> reflection string
    """
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.default_credits = 5

        # Prepare LLM client if API key exists (otherwise we’ll fall back)
        self.client = None
        if _OPENAI_OK and os.environ.get("OPENAI_API_KEY"):
            try:
                self.client = OpenAI()
            except Exception:
                self.client = None

    # ---- Sessions ----
    def create_session(self, profile: Dict) -> Tuple[str, int]:
        """
        profile keys we accept:
            - name
            - relationship (user->them): e.g., son/daughter/friend
            - call_you  (what they call the user) e.g., beta/betay/son
            - traits (list[str])
            - catchphrases (list[str])
            - mode: "memory" (they've passed away) OR "alive" (they are alive)
        """
        p = {
            "name": _trim(profile.get("name", "")),
            "relationship": _trim(profile.get("relationship", "")),
            "call_you": _trim(profile.get("call_you", "")),
            "traits": profile.get("traits") or [],
            "catchphrases": profile.get("catchphrases") or [],
            "mode": (profile.get("mode") or "memory").strip().lower(),
        }

        # Normalize traits/catchphrases if they accidentally arrive as CSV strings
        if isinstance(p["traits"], str):
            p["traits"] = _csv_to_list(p["traits"])
        if isinstance(p["catchphrases"], str):
            p["catchphrases"] = _csv_to_list(p["catchphrases"])

        if p["mode"] not in ("memory", "alive"):
            p["mode"] = "memory"

        sid = uuid4().hex
        self.sessions[sid] = {
            "profile": p,
            "history": [],   # list of {"role":"user/assistant", "content": "..."}
            "credits": self.default_credits,
        }
        return sid, self.sessions[sid]["credits"]

    # Backwards-compat for older server code that calls new_session()
    def new_session(self, profile: Dict) -> Tuple[str, int]:
        return self.create_session(profile)

    # ---- Chat ----
    def chat(self, session_id: str, message: str) -> Tuple[str, int]:
        s = self.sessions.get(session_id)
        if not s:
            raise ValueError("Unknown session_id")

        if s["credits"] <= 0:
            raise ValueError("No credits left")

        profile = s["profile"]
        sys_prompt = self._system_prompt(profile)
        msgs = [{"role": "system", "content": sys_prompt}]
        # include last few exchanges for continuity
        for m in s["history"][-6:]:
            msgs.append(m)
        msgs.append({"role": "user", "content": message})

        reply_text = self._llm_reply(msgs)
        # book-keeping
        s["history"].append({"role": "user", "content": message})
        s["history"].append({"role": "assistant", "content": reply_text})
        s["credits"] -= 1
        return reply_text, s["credits"]

    def end_session(self, session_id: str) -> str:
        s = self.sessions.get(session_id)
        if not s:
            raise ValueError("Unknown session_id")

        profile = s["profile"]
        name = profile.get("name") or "Your loved one"
        # very lightweight “reflection”
        # (if you want LLM-based reflection, you can call self._llm_reply here too)
        lines = [m["content"] for m in s["history"] if m["role"] == "assistant"]
        takeaway = lines[-1] if lines else "They care about you and are by your side."
        return f"{name} would want you to remember: {takeaway}"

    # ---- Prompting ----
    def _system_prompt(self, p: Dict) -> str:
        """
        Style guardrails to avoid ‘video call’ talk in memory mode, etc.
        """
        name = p.get("name") or "Your loved one"
        rel = p.get("relationship") or "family"
        call_you = p.get("call_you") or "beta"
        traits = ", ".join(p.get("traits") or [])
        catch = ", ".join(p.get("catchphrases") or [])

        base_tone = (
            "Warm, grounded, concise. Speak like an ordinary human, not a chatbot. "
            "Use everyday language. Avoid therapy clichés."
        )

        if p.get("mode") == "memory":
            # They've passed away — never suggest calling or meeting
            constraints = (
                "Important: You represent a preserved memory of them.\n"
                "- Do NOT suggest phone calls, video calls, meet-ups, or any real-time contact.\n"
                "- Speak in the present-tense warmth of memory (e.g., ‘I’m proud of you’), "
                "but never imply physical presence or that you can call or meet.\n"
                "- It’s okay to say ‘I’m with you in spirit’ or ‘I remember when…’\n"
            )
        else:
            # Alive
            constraints = (
                "Important: You represent this person while alive.\n"
                "- It’s okay to suggest practical, real-world connection (e.g., ‘call me later’), "
                "but stay natural and subtle.\n"
            )

        persona = (
            f"You are {name}, the user's {rel}. You call the user '{call_you}'. "
            f"Traits: {traits if traits else 'caring'}. "
            f"Catchphrases (use rarely, naturally): {catch if catch else '—'}."
        )

        return f"{base_tone}\n{constraints}\n{persona}"

    # ---- LLM or fallback ----
    def _llm_reply(self, messages: List[Dict]) -> str:
        """
        Try OpenAI if available; otherwise a simple fallback that keeps the tone rules.
        """
        if self.client:
            try:
                # You can swap the model if you prefer
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.6,
                    max_tokens=220,
                )
                txt = resp.choices[0].message.content.strip()
                if txt:
                    return txt
            except Exception:
                pass  # fall through to fallback

        # Fallback: very simple echo with warmth
        user_last = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_last = m["content"].strip()
                break
        return (
            "I hear you. I’m with you in spirit. "
            f"{user_last[:140]}"[:220].rstrip()
        )
