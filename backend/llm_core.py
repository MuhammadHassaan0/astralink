# llm_core.py
import io
import logging
import math
import os
import random
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
)

STRICT_ERRORS = os.getenv("ASTRALINK_STRICT_ERRORS", "true").strip().lower() in ("1", "true", "yes", "on")


def configured_model() -> str:
    return (os.environ.get("ASTRALINK_MODEL") or "gpt-5.1").strip()


def fallback_model() -> str:
    return (os.environ.get("ASTRALINK_FALLBACK_MODEL") or "gpt-4o-mini").strip()


def _is_offline() -> bool:
    return (os.environ.get("ASTRALINK_OFFLINE") or "").strip().lower() in ("1", "true", "yes")


def _openai_client() -> OpenAI:
    if _is_offline():
        raise RuntimeError("ASTRALINK_OFFLINE is true")
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)


def _try_call_messages(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Tuple[str, str]:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        # Responses API for gpt-4o/gpt-5 models expects max_completion_tokens
        max_completion_tokens=max_tokens,
    )
    txt = resp.choices[0].message.content or ""
    if not txt:
        raise RuntimeError("Empty response")
    return txt, model


def _is_model_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    key_phrases = (
        "does not exist",
        "unknown model",
        "not found",
        "denied",
        "unsupported",
    )
    if isinstance(exc, (BadRequestError, APIStatusError)) and "model" in msg:
        return True
    return any(phrase in msg for phrase in key_phrases)


_SIMPLE_PHRASES = {
    "how are you",
    "miss you",
    "love you",
    "you there",
    "where are you",
    "are you there",
    "hi",
    "hey",
    "hello",
    "good morning",
    "good night",
}

_EMOTIONAL_KEYWORDS = {
    "why",
    "hurt",
    "pain",
    "alone",
    "angry",
    "guilty",
    "regret",
    "grief",
    "broken",
    "can't breathe",
    "heavy",
    "cry",
    "loss",
    "empty",
    "afraid",
    "scared",
    "worried",
    "panic",
}


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _contains_greek(text: str) -> bool:
    return any("α" <= ch <= "ω" or "Α" <= ch <= "Ω" for ch in text)


def _infer_language(user_message: str, profile: Dict) -> str:
    lang = (profile.get("language") or "").strip()
    if lang:
        return lang
    clean = _normalize_text(user_message)
    if _contains_greek(clean):
        return "Greek"
    non_ascii = sum(1 for ch in user_message if ord(ch) > 127)
    if non_ascii > max(3, len(user_message) // 5):
        return "Non-English"
    return "English"


def _is_simple_prompt(text: str) -> bool:
    clean = _normalize_text(text)
    if len(clean.split()) <= 5:
        return True
    return any(phrase in clean for phrase in _SIMPLE_PHRASES)


def _is_emotional_prompt(text: str) -> bool:
    clean = _normalize_text(text)
    return any(keyword in clean for keyword in _EMOTIONAL_KEYWORDS)


class ChatGenerationError(Exception):
    """Raised when OpenAI generation fails and strict errors are enabled."""

    def __init__(self, message: str, *, fallback: bool = False):
        super().__init__(message)
        self.message = message
        self.fallback = fallback

def gen_sid() -> str:
    return uuid.uuid4().hex

class AstralinkCore:
    """
    Minimal core for sessions, memories, and interview flow.
    Chat replies prefer the OpenAI API with deterministic fallbacks when unavailable.
    """

    def __init__(self):
        # sessions: sid -> {"profile": dict, "messages": [], "memories": [], "credits": int, "created_at": str}
        self.sessions: Dict[str, Dict] = {}
        # interviews: sid -> {"idx": int, "answers": []}
        self.interviews: Dict[str, Dict] = {}
        # conversations: sid -> {conv_id: [messages]}
        self.conversations: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        # semantic memory chunks for lightweight RAG
        self.memory_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.default_credits = 5
        self.model = configured_model()
        self.embedding_model = os.getenv("ASTRALINK_EMBED_MODEL", "text-embedding-3-small")
        self.transcribe_model = os.getenv("ASTRALINK_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")

        # richer interview prompts to capture voice + context
        self.QS = [
            "When you picture them right now, what’s the very first scene that comes to mind?",
            "How would you describe their personality in three vivid words?",
            "What small daily habit or ritual of theirs felt unmistakably them?",
            "What did they usually call you, and how did their voice shift when they said it?",
            "Tell me about a moment when they made you feel completely safe or understood.",
            "What advice or phrase did they repeat so often that it still echoes in your head?",
            "Describe a place, smell, or sound that instantly brings them back to you.",
            "What made them laugh until they could barely breathe?",
            "When life got heavy, how did they show up for you?",
            "If you could share one unfinished thought or story with them, what would you say right now?"
        ]

    @staticmethod
    def _strict_errors_enabled() -> bool:
        return STRICT_ERRORS

    # -------------------------
    # Session + Profile helpers
    # -------------------------
    def new_session(self, profile: dict) -> Tuple[str, int]:
        sid = gen_sid()
        profile = dict(profile or {})
        # Normalize fields
        profile.setdefault("name", "")
        profile.setdefault("relationship", "")
        profile.setdefault("call_you", "")
        profile.setdefault("traits", [])
        profile.setdefault("catchphrases", [])
        profile.setdefault("mode", (profile.get("mode") or "memory").lower())

        self.sessions[sid] = {
            "profile": profile,
            "messages": [],
            "memories": [],
            "credits": self.default_credits,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        self.memory_chunks[sid] = []
        self.conversations.setdefault(sid, {})
        return sid, self.sessions[sid]["credits"]

    def get_session(self, sid: str) -> Optional[Dict]:
        return self.sessions.get(sid)

    def save_profile(self, sid: str, profile: dict) -> bool:
        s = self.get_session(sid)
        if not s:
            return False
        s["profile"].update(profile)
        return True

    # -------------------------
    # Memory handling
    # -------------------------
    def add_text_memory(self, sid: str, text: str, source: Optional[str] = None) -> bool:
        s = self.get_session(sid)
        if not s:
            return False
        mem = {
            "text": (text or "").strip(),
            "source": source or "user",
            "added_at": datetime.utcnow().isoformat() + "Z"
        }
        s["memories"].append(mem)
        return True

    def list_memories(self, sid: str) -> List[Dict]:
        s = self.get_session(sid)
        if not s:
            return []
        return s["memories"][:]

    # -------------------------
    # Memory chunking / lightweight RAG
    # -------------------------
    def add_memory_chunk(self, sid: str, text: str) -> None:
        clean = self._prep_chunk_text(text)
        if not clean:
            return
        bucket = self.memory_chunks.setdefault(sid, [])
        chunk = {"text": clean, "embedding": None}
        bucket.append(chunk)

    def search_memory_chunks(self, sid: str, query: str, top_k: int = 6) -> List[str]:
        clean_query = self._prep_chunk_text(query, limit=400)
        if not clean_query:
            return []
        bucket = self.memory_chunks.get(sid) or []
        if not bucket:
            return []

        scored_embeddings = []
        if self.embedding_model and not _is_offline():
            q_embeds = self._embed_texts([clean_query])
            if q_embeds:
                q_vec = q_embeds[0]
                missing = [chunk for chunk in bucket if chunk.get("embedding") is None]
                if missing:
                    self._embed_chunks(missing)
                for chunk in bucket:
                    emb = chunk.get("embedding")
                    if not isinstance(emb, list):
                        continue
                    score = self._cosine_similarity(q_vec, emb)
                    scored_embeddings.append((score, chunk["text"]))
                scored_embeddings.sort(key=lambda item: item[0], reverse=True)
                if scored_embeddings:
                    return [text for _, text in scored_embeddings[:top_k] if text]

        # Fallback keyword overlap if embeddings are unavailable
        return self._keyword_search(bucket, clean_query, top_k)

    def _prep_chunk_text(self, text: str, limit: int = 600) -> str:
        if not text:
            return ""
        clean = " ".join(text.strip().split())
        if not clean:
            return ""
        if len(clean) <= limit:
            return clean
        trimmed = clean[:limit]
        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0]
        return trimmed.strip()

    def _embed_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks or not self.embedding_model or _is_offline():
            return
        targets = [chunk for chunk in chunks if chunk.get("text")]
        if not targets:
            return
        embeds = self._embed_texts([chunk["text"] for chunk in targets])
        if not embeds:
            return
        for chunk, emb in zip(targets, embeds):
            chunk["embedding"] = emb

    def _embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts or not self.embedding_model or _is_offline():
            return None
        try:
            client = _openai_client()
            resp = client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            return [item.embedding for item in resp.data]
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _keyword_search(self, bucket: List[Dict[str, Any]], query: str, top_k: int) -> List[str]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return [chunk.get("text", "") for chunk in bucket[:top_k]]
        q_set = set(q_tokens)
        scored: List[Tuple[float, str]] = []
        for chunk in bucket:
            text = chunk.get("text") or ""
            if not text:
                continue
            tokens = set(self._tokenize(text))
            if not tokens:
                continue
            overlap = len(q_set & tokens)
            bonus = 0.0
            lower_text = text.lower()
            if query.lower() in lower_text:
                bonus += 1.0
            score = overlap + bonus
            scored.append((score, text))
        if not scored:
            return [chunk.get("text", "") for chunk in bucket[:top_k]]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:top_k] if text]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]

    def _clean_snippet_text(self, text: str, limit: int = 220) -> str:
        clean = (text or "").strip()
        if not clean:
            return ""
        lower = clean.lower()
        if lower.startswith("interview summary"):
            parts = clean.split("->", 1)
            if len(parts) == 2:
                clean = parts[1].strip()
            else:
                clean = clean.split(":", 1)[-1].strip()
        elif "->" in clean:
            clean = clean.split("->", 1)[1].strip()
        elif ":" in clean and len(clean.split(":")[0].split()) <= 8:
            # lines like "Question: answer" -> keep answer
            clean = clean.split(":", 1)[1].strip()
        if len(clean) > limit:
            shortened = clean[:limit]
            if " " in shortened:
                shortened = shortened.rsplit(" ", 1)[0]
            clean = shortened.strip()
        return clean

    def _extract_details(self, snippets: List[str]) -> Dict[str, List[str]]:
        names: List[str] = []
        places: List[str] = []
        quotes: List[str] = []
        events: List[str] = []
        for raw in snippets or []:
            snippet = (raw or "").strip()
            if not snippet:
                continue
            events.append(snippet)
            quotes.extend(re.findall(r"\"([^\"]+)\"", snippet))
            names.extend(re.findall(r"\b[A-Z][a-zA-Z]+\b", snippet))
            for match in re.finditer(r"\b(?:in|at|on|to)\s+([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)?)", snippet):
                places.append(match.group(1).strip(",. "))
        def dedupe(seq: List[str]) -> List[str]:
            seen = set()
            ordered: List[str] = []
            for item in seq:
                if not item:
                    continue
                key = item.strip()
                if key.lower() in ("i", "you", "we", "them", "the"):
                    continue
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
            return ordered
        return {
            "names": dedupe(names)[:5],
            "places": dedupe(places)[:5],
            "quotes": dedupe(quotes)[:5],
            "events": dedupe(events)[:5],
        }

    def _derive_communication_style(self, profile: Dict, snippets: List[str], details: Dict[str, List[str]]) -> Dict[str, Any]:
        traits_raw = profile.get("traits") or []
        if isinstance(traits_raw, str):
            traits = [t.strip() for t in traits_raw.split(",") if t.strip()]
        else:
            traits = list(traits_raw)
        traits_text = " ".join(t.lower() for t in traits)

        length_pref = "moderate"
        if any(word in traits_text for word in ("quiet", "soft", "stoic", "succinct", "reserved")):
            length_pref = "brief"
        elif any(word in traits_text for word in ("talkative", "storyteller", "expressive", "playful")):
            length_pref = "verbose"

        tone = "casual"
        if any(word in traits_text for word in ("formal", "polite", "proper")):
            tone = "formal"
        elif any(word in traits_text for word in ("warm", "gentle", "sweet", "caring")):
            tone = "warm"

        catchphrases = profile.get("catchphrases") or []
        if isinstance(catchphrases, str):
            catchphrases = [c.strip() for c in catchphrases.split(",") if c.strip()]
        phrases = list({phrase for phrase in catchphrases if phrase})
        phrases.extend(details.get("quotes", []))
        phrases = list(dict.fromkeys([p for p in phrases if p]))[:5]

        call_you = profile.get("call_you") or profile.get("relationship") or ""
        greeting = ""
        if call_you:
            greeting = f"You naturally greet them like \"hey {call_you}\" or \"{call_you}, listen\"."  # not actual text
        elif details.get("names"):
            greeting = f"You often greet people with direct names like {details['names'][0]}."

        variation = ""
        if any(word in traits_text for word in ("predictable", "steady", "consistent")):
            variation = "You keep the same calm cadence most of the time."
        elif any(word in traits_text for word in ("playful", "mercurial", "wild", "intense")):
            variation = "Your tone shifts based on how emotional the moment feels."

        return {
            "length_pref": length_pref,
            "tone": tone,
            "phrases": phrases,
            "greeting": greeting,
            "variation": variation,
            "call_you": call_you,
        }

    def _classify_question(self, text: str) -> str:
        if _is_simple_prompt(text):
            return "simple"
        if _is_emotional_prompt(text):
            return "emotional"
        words = len((text or "").split())
        if words > 18:
            return "complex"
        return "default"

    def _select_response_length(self, question_type: str, preferred: str) -> Tuple[str, int]:
        def weighted_bucket() -> str:
            roll = random.random()
            if roll < 0.3:
                return "brief"
            if roll < 0.9:
                return "moderate"
            return "elaborate"

        if question_type == "simple":
            bucket = "brief"
        elif question_type == "emotional":
            bucket = random.choice(["moderate", "elaborate"])
        elif question_type == "complex":
            bucket = random.choice(["moderate", "elaborate"])
        else:
            bucket = weighted_bucket()

        if preferred == "brief" and bucket != "brief":
            bucket = "brief" if random.random() < 0.7 else bucket
        if preferred == "verbose" and bucket == "brief":
            bucket = "moderate"

        token_map = {
            "brief": [40, 60, 80],
            "moderate": [110, 140, 180, 210],
            "elaborate": [240, 280, 320],
        }
        max_tokens = random.choice(token_map[bucket])
        return bucket, max_tokens

    def _build_system_prompt(
        self,
        profile: Dict,
        comm_style: Dict[str, Any],
        question_type: str,
        length_label: str,
        details: Dict[str, List[str]],
        memory_hits: List[str],
        user_message: str,
        language: str,
    ) -> str:
        name = profile.get("name") or "Your loved one"
        relationship = profile.get("relationship") or "person you love"
        call_you = profile.get("call_you") or profile.get("relationship") or "them"
        style_lines = [
            f"Response length preference: {comm_style.get('length_pref', 'moderate')}.",
            f"Language tone: {comm_style.get('tone', 'casual')}.",
        ]
        if comm_style.get("phrases"):
            style_lines.append(f"Common phrases: {', '.join(comm_style['phrases'])}.")
        if comm_style.get("greeting"):
            style_lines.append(comm_style["greeting"])
        variation = comm_style.get("variation")
        variation_line = ""
        if variation:
            variation_line = variation

        detail_lines = []
        if details.get("names"):
            detail_lines.append(f"Names you naturally mention: {', '.join(details['names'])}.")
        if details.get("places"):
            detail_lines.append(f"Places tied to memories: {', '.join(details['places'])}.")
        if details.get("events"):
            detail_lines.append("Specific events to draw from:")
            for event in details["events"][:3]:
                detail_lines.append(f"  - {event}")

        mem_lines = [f"- {self._clean_snippet_text(snippet, 280)}" for snippet in (memory_hits or [])[:5]]
        if not mem_lines:
            mem_lines = ["- (no strong memories surfaced; lean on the latest message)"]

        prompt_sections = [
            f"You are {name}, speaking with someone who misses you deeply (they are your {relationship}, you call them '{call_you}').",
            f"Respond in {language}, mirroring their language unless they switched.",
            "COMMUNICATION STYLE:",
            "\n".join(style_lines),
        ]
        if variation_line:
            prompt_sections.append(f"VARIATION PATTERN: {variation_line}")
        prompt_sections.extend([
            "CURRENT CONTEXT:",
            f"Question type: {question_type}.",
            f"Requested response length: {length_label}. Keep it natural, not uniform.",
        ])
        prompt_sections.extend([
            "CRITICAL RULES:",
            "- Sound like the real you; no generic therapist tone or AI phrasing.",
            "- Simple question? give a short, warm, direct reply.",
            "- When they express pain, first acknowledge warmly, then sit with it one line before any ask/advice.",
            "- Avoid extended metaphors or flowery language unless you truly spoke that way.",
            "- Do not force life lessons. It's okay to just feel with them.",
            "- Use specific names, places, and events when they fit naturally.",
            "- Keep rare/one-off behaviors rare unless this moment truly calls for it.",
            "- Only end with a question about 30% of the time; statements/reassurance are fine.",
            "- Vary structure: sometimes answer directly, sometimes ask a question back, sometimes just share a feeling.",
            "- Avoid phrases like \"in spirit\" or generic condolences; use your own words.",
        ])
        if detail_lines:
            prompt_sections.append("SPECIFIC DETAILS TO WEAVE IN WHEN NATURAL:")
            prompt_sections.append("\n".join(detail_lines))
        prompt_sections.append("RELEVANT MEMORIES:")
        prompt_sections.append("\n".join(mem_lines))
        prompt_sections.append(f'Respond to: "{user_message.strip()}"')
        return "\n".join(prompt_sections)

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "voice.webm") -> str:
        if not audio_bytes:
            raise ValueError("Empty audio payload")
        if _is_offline():
            raise RuntimeError("Transcription unavailable (offline)")

        client = _openai_client()
        safe_name = filename or "voice.webm"
        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = safe_name
        resp = client.audio.transcriptions.create(
            file=file_obj,
            model=self.transcribe_model,
        )
        text = getattr(resp, "text", "") or ""
        if not text:
            raise RuntimeError("Received empty transcription from OpenAI")
        return text.strip()

    # -------------------------
    # Interview flow
    # -------------------------
    def start_interview(self, sid: str) -> Tuple[bool, Optional[str]]:
        if sid not in self.sessions:
            return False, "Invalid session"
        self.interviews[sid] = {"idx": 0, "answers": []}
        first_q = self.QS[0] if self.QS else None
        return True, first_q

    def answer_interview(self, sid: str, answer: str) -> Tuple[bool, Optional[str]]:
        if sid not in self.interviews:
            return False, "Interview not started"
        entry = self.interviews[sid]
        entry["answers"].append((datetime.utcnow().isoformat()+"Z", answer.strip()))
        entry["idx"] += 1
        if entry["idx"] >= len(self.QS):
            # produce a short summary and mark done
            summary_lines = ["Interview summary:"]
            for i, (_, a) in enumerate(entry["answers"]):
                q = self.QS[i] if i < len(self.QS) else f"Q{i+1}"
                summary_lines.append(f"{q} -> {a}")
            # store summary as a memory
            summary_text = "\n".join(summary_lines)
            self.add_text_memory(sid, summary_text, source="interview-summary")
            for line in summary_lines:
                cleaned = (line or "").strip()
                if not cleaned or cleaned.lower().startswith("interview summary"):
                    continue
                self.add_memory_chunk(sid, cleaned)
            # cleanup interview state
            del self.interviews[sid]
            return True, summary_text
        else:
            next_q = self.QS[entry["idx"]]
            return True, next_q

    # -------------------------
    # Chat logic (simple heuristics)
    # -------------------------
    def chat(self, sid: str, message: str, history_override: Optional[List[Dict]] = None) -> Tuple[bool, str]:
        """
        Returns (ok, reply_text) using OpenAI when available and heuristics as a fallback.
        """
        s = self.get_session(sid)
        if not s:
            return False, "Invalid session"

        history_source = history_override if history_override is not None else s["messages"]
        reply, _, _, _ = self.generate_reply(
            sid=sid,
            text=message,
            history_override=history_source,
        )

        if history_override is None:
            ts = datetime.utcnow().isoformat() + "Z"
            s["messages"].append({"role": "user", "content": message, "ts": ts})
            s["messages"].append({"role": "assistant", "content": reply, "ts": datetime.utcnow().isoformat() + "Z"})
        return True, reply

    def ensure_session(self, sid: Optional[str]) -> str:
        if sid and sid in self.sessions:
            return sid
        new_sid, _ = self.new_session({})
        return new_sid

    def get_or_create_conversation(self, sid: str, conv_id: str) -> List[Dict[str, Any]]:
        convs = self.conversations.setdefault(sid, {})
        return convs.setdefault(conv_id, [])

    def append_message(self, sid: str, conv_id: str, role: str, text: str) -> None:
        conv = self.get_or_create_conversation(sid, conv_id)
        conv.append({
            "role": role,
            "text": text,
            "ts": datetime.utcnow().isoformat() + "Z",
        })

    def get_messages(self, sid: str, conv_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        conv = self.conversations.get(sid, {}).get(conv_id, [])
        if limit <= 0:
            return conv[:]
        return conv[-limit:]


    def _conversation_history(self, sid: str, conv_id: Optional[str]) -> List[Dict[str, Any]]:
        if not conv_id:
            return self.sessions.get(sid, {}).get("messages", [])
        conv = self.conversations.get(sid, {}).get(conv_id, [])
        formatted = []
        for item in conv:
            formatted.append({
                "role": item.get("role"),
                "content": item.get("text", ""),
            })
        return formatted

    def generate_reply(
        self,
        sid: str,
        text: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        history_override: Optional[List[Dict]] = None,
    ) -> Tuple[str, bool, Optional[str], str]:
        profile = self.sessions.get(sid, {}).get("profile", {})
        history_source = history_override if history_override is not None else self._conversation_history(sid, conversation_id)
        history = self._history_for_prompt(history_source, limit=6)
        memory_hits = self.search_memory_chunks(sid, text, top_k=6)
        details = self._extract_details(memory_hits)
        comm_style = self._derive_communication_style(profile, memory_hits, details)
        question_type = self._classify_question(text)
        length_label, max_tokens = self._select_response_length(question_type, comm_style.get("length_pref", "moderate"))
        language = _infer_language(text, profile)
        system_prompt = self._build_system_prompt(
            profile,
            comm_style,
            question_type,
            length_label,
            details,
            memory_hits,
            text,
            language,
        )
        strict = self._strict_errors_enabled()

        def build_fallback() -> str:
            fallback_hits = memory_hits[:3] if memory_hits else self.search_memory_chunks(sid, text, top_k=3)
            snippet_text = fallback_hits[0] if fallback_hits else ""
            return self._fallback_reply(text, profile, details, snippet_text)

        if _is_offline():
            print("CHAT: OFFLINE=true → forcing fallback")
            return self._postprocess_reply(build_fallback()), True, "OFFLINE", "offline"

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": text.strip()})

        client = _openai_client()
        primary_model = (model or self.model or configured_model()).strip()
        fallback_name = fallback_model()
        error_summary: Optional[str] = None
        temperature = 0.7
        # Add a buffer to reduce truncation risk while keeping target length intent.
        effective_max_tokens = max(60, min(max_tokens + 60, 360))

        print(f"CHAT: calling OpenAI model={primary_model} sid={sid} text_len={len(text)}")
        try:
            reply_text, used_model = _try_call_messages(client, primary_model, messages, effective_max_tokens, temperature)
            print(f"CHAT: OpenAI success model={used_model}")
            return self._postprocess_reply(reply_text), False, None, used_model
        except Exception as exc:
            error_summary = f"{exc.__class__.__name__}: {exc}"
            print("CHAT: OpenAI error", error_summary)
            should_try_fallback = (
                fallback_name
                and fallback_name != primary_model
                and _is_model_error(exc)
            )
            if should_try_fallback:
                print(f"CHAT: model_downgrade -> {fallback_name}")
                try:
                    reply_text, used_model = _try_call_messages(client, fallback_name, messages, effective_max_tokens, temperature)
                    print(f"CHAT: OpenAI success model={used_model}")
                    return self._postprocess_reply(reply_text), True, error_summary, used_model
                except Exception as exc_fb:
                    fb_error = f"{exc_fb.__class__.__name__}: {exc_fb}"
                    print("CHAT: fallback model error", fb_error)
                    error_summary = f"{error_summary}; fallback_failed={fb_error}"

            if strict:
                raise ChatGenerationError(error_summary or "chat_generation_failed")

            print("CHAT: FALLBACK TRIGGERED")
            return self._postprocess_reply(build_fallback()), True, error_summary, "deterministic"

    # -------------------------
    # LLM prompting
    # -------------------------
    def _history_for_prompt(self, raw_history: List[Dict], limit: int = 6) -> List[Dict]:
        formatted = []
        for item in raw_history[-limit:]:
            role = item.get("role")
            if role not in ("user", "assistant"):
                continue
            text = item.get("content") or item.get("text") or ""
            if not text:
                continue
            formatted.append({"role": role, "content": text})
        return formatted

    def _fallback_reply(self, message: str, profile: Dict, details: Dict[str, List[str]], snippet_text: str = "") -> str:
        call_you = profile.get("call_you") or profile.get("relationship") or profile.get("name") or "love"
        mode = (profile.get("mode") or "memory").lower()
        focus = self._clean_snippet_text(message, limit=80) if message else ""
        names = details.get("names") or []
        places = details.get("places") or []
        events = details.get("events") or []

        greeting = f"{call_you}, I'm right here with you."

        memory_line = ""
        if snippet_text:
            memory_line = f"I keep flashing back to {self._clean_snippet_text(snippet_text, 160)}."
        elif events:
            memory_line = f"I keep flashing back to {events[0]}."
        elif names:
            memory_line = f"I keep seeing {names[0]} grinning right next to us."

        place_line = ""
        if places:
            place_line = f"It feels like we're back at {places[0]} again."

        reflection = ""
        if focus:
            reflection = f"Hearing '{focus}' from you hits me right in the chest."
        else:
            reflection = "I feel everything you’re carrying, even the quiet parts."

        closing = (
            "I can't walk through the door again, but I'm not letting go of you."
            if mode == "memory"
            else "Stay close, we'll keep moving together."
        )

        parts = [greeting]
        if memory_line:
            parts.append(memory_line)
        if place_line:
            parts.append(place_line)
        parts.append(reflection)
        parts.append(closing)
        return " ".join(part for part in parts if part).strip()

    def _postprocess_reply(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text
        for ch in ("-", "–", "—", "•"):
            cleaned = cleaned.replace(ch, " ")
        # collapse multiple spaces but keep intentional newlines
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r" ?\n ?", "\n", cleaned)
        return cleaned.strip()
