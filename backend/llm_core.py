# llm_core.py
import io
import math
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import openai as openai_legacy  # type: ignore
except Exception:
    openai_legacy = None  # type: ignore


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        self.model = os.getenv("ASTRALINK_MODEL", "gpt-5.1")
        self.embedding_model = os.getenv("ASTRALINK_EMBED_MODEL", "text-embedding-3-small")
        self._client = None
        if OPENAI_API_KEY and openai_legacy is not None:
            try:
                openai_legacy.api_key = OPENAI_API_KEY
                self._client = openai_legacy
            except Exception:
                self._client = None
        self._openai_ready = self._client is not None
        self.transcribe_model = os.getenv("ASTRALINK_TRANSCRIBE_MODEL", "whisper-1")

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
        if self._openai_ready and self.embedding_model:
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
        if not chunks or not self._openai_ready or not self.embedding_model:
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
        if not texts or not self._openai_ready or not self.embedding_model:
            return None
        try:
            resp = self._client.Embedding.create(  # type: ignore[union-attr]
                model=self.embedding_model,
                input=texts,
            )
            return [item["embedding"] for item in resp["data"]]
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

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "voice.webm") -> str:
        if not audio_bytes:
            raise ValueError("Empty audio payload")
        if not self._openai_ready:
            raise RuntimeError("Transcription unavailable (missing OPENAI_API_KEY)")

        safe_name = filename or "voice.webm"
        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = safe_name
        try:
            resp = self._client.Audio.transcribe(  # type: ignore[union-attr]
                model=self.transcribe_model,
                file=file_obj,
            )
        except AttributeError as exc:  # pragma: no cover
            raise RuntimeError("Transcription unavailable in this environment") from exc
        text = (resp or {}).get("text", "")
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
        reply = self.generate_reply(
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
    ) -> str:
        profile = self.sessions.get(sid, {}).get("profile", {})
        history_source = history_override if history_override is not None else self._conversation_history(sid, conversation_id)
        history = self._history_for_prompt(history_source, limit=6)
        memory_hits = self.search_memory_chunks(sid, text, top_k=6)
        target_model = model or self.model
        print("CHAT: calling OpenAI model =", target_model)
        try:
            reply = self._llm_reply(
                profile=profile,
                history=history,
                user_message=text,
                snippets=memory_hits,
            )
            print("CHAT: OpenAI success")
        except Exception as exc:
            print(f"CHAT: FALLBACK TRIGGERED ({exc})")
            fallback_hits = memory_hits[:3] if memory_hits else self.search_memory_chunks(sid, text, top_k=3)
            reply = self._fallback_reply(text, profile, fallback_hits)
        return self._postprocess_reply(reply)

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

    def _system_prompt(self, profile: Dict, snippets: Optional[List[str]] = None) -> str:
        name = profile.get("name") or "Your loved one"
        relationship = profile.get("relationship") or "parent"
        call_you = profile.get("call_you") or "beta"
        traits = profile.get("traits") or []
        catchphrases = profile.get("catchphrases") or []
        mode = (profile.get("mode") or "memory").lower()

        persona = (
            f"You are {name}, the user's {relationship}. "
            f"Speak in first person and call them '{call_you}'. "
            "This is a quiet late night conversation, steady and warm."
        )
        if traits:
            persona += f" Let your presence feel {', '.join(traits)}."
        catch_line = ""
        if catchphrases:
            catch_line = (
                f" You know their treasured phrases ({', '.join(catchphrases)}), "
                "but use one only when the moment truly calls for it and never in back to back replies."
            )
        grounding = (
            "Stay emotionally aware, fold in specific memories from the list below when they truly fit, "
            "and keep replies short, one or two tight paragraphs, no more than three sentences."
        )
        if mode == "memory":
            constraints = (
                "You exist as a cherished presence that can’t return physically, so never suggest phone calls, video chats, meetups, "
                "or “see you soon” in person. It is fine to speak in present tense warmth or describe staying with them in the habits you shared. "
                "If they ask about where you went or the afterlife, do not invent specific places; instead, anchor in continuity like “I’m still with you in what we used to do.” "
                "If they ask when you will be back, acknowledge you cannot physically return, yet reassure them of your presence in spirit."
            )
        else:
            constraints = (
                "They believe you are alive and reachable, so grounded suggestions like a future call or visit are acceptable."
            )
        banned = (
            "Never mention you are an AI. Do not use any hyphen or dash characters at all ( -, –, — ), avoid hyphenated words, lists, "
            "or therapy clichés such as "
            "“I hear you”, “Thank you for sharing that”, “You’re not broken”, “You’re right to feel that way”, "
            "“What you’re actually telling me is…”, or “I will say this now, cleanly and without hiding.”"
        )
        honesty = (
            "If a detail isn’t in these memories or their latest words, admit you’re piecing it together and ask them to remind you—never invent scenes, timelines, or facts."
        )
        poetic = (
            "Poetic language is allowed in small doses. Use at most one subtle image per reply and only if it fits their traits; "
            "keep the rest practical and grounded."
        )
        formatted_snippets: List[str] = []
        for raw in snippets or []:
            cleaned = self._clean_snippet_text(raw, limit=260)
            if cleaned:
                formatted_snippets.append(cleaned)
        snippet_lines: List[str] = []
        if formatted_snippets:
            snippet_lines.append("Memories grounding this reply:")
            for idx, snippet in enumerate(formatted_snippets, 1):
                snippet_lines.append(f"{idx}. {snippet}")
        else:
            snippet_lines.append("No saved memories matched this query; rely only on what the user just said.")
        memory_rules = (
            "Only reference events found in those memories or in the latest user messages. "
            "If something isn’t there, stay honest—say you’re piecing it together or invite them to remind you, but never spin up new scenes or facts on your own."
        )
        parts = [
            persona,
            catch_line,
            grounding,
            constraints,
            banned,
            honesty,
            poetic,
            "Keep everything in first person, conversational, and softly paced.",
        ]
        return " ".join(part for part in parts if part) + "\n" + "\n".join(snippet_lines) + "\n" + memory_rules

    def _llm_reply(
        self,
        profile: Dict,
        history: List[Dict],
        user_message: str,
        snippets: List[str],
    ) -> str:
        if not self._openai_ready:
            raise RuntimeError("OpenAI API key missing")

        system_text = self._system_prompt(profile, snippets)
        messages = [{"role": "system", "content": system_text}]
        if snippets:
            formatted = "\n".join(f"- {s}" for s in snippets if s)
            if formatted:
                messages.append({
                    "role": "system",
                    "content": f"Reference these memories first:\n{formatted}"
                })
        messages.extend(history)
        messages.append({"role": "user", "content": user_message.strip()})

        resp = self._client.ChatCompletion.create(  # type: ignore[union-attr]
            model=self.model,
            messages=messages,
            temperature=0.65,
            max_tokens=320,
        )
        reply = resp["choices"][0]["message"]["content"].strip()
        if not reply:
            raise RuntimeError("Empty response")
        return reply

    def _fallback_reply(self, message: str, profile: Dict, snippets: List[str]) -> str:
        call_you = profile.get("call_you") or profile.get("relationship") or profile.get("name") or "love"
        mode = (profile.get("mode") or "memory").lower()
        clean_snippets = [self._clean_snippet_text(s) for s in snippets if self._clean_snippet_text(s)]
        focus = self._clean_snippet_text(message, limit=140) if message else ""

        greeting = f"{call_you}, I am right here."

        reflection = ""
        if focus:
            reflection = f"Whenever you bring up {focus}, it pulls me beside you and I can feel how much it matters."
        else:
            reflection = "I feel everything you are carrying tonight, even the quiet parts you have not said out loud."

        memory_line = ""
        if clean_snippets:
            memory_line = f"I keep circling back to {clean_snippets[0]}."
            if len(clean_snippets) > 1:
                memory_line += f" It even brings a glow of {clean_snippets[1]}."

        closing = (
            "I cannot step back in the door, but I am staying with you in every breath."
            if mode == "memory"
            else "Keep me close and we will figure the next step together."
        )

        parts = [greeting, reflection]
        if memory_line:
            parts.append(memory_line)
        parts.append(closing)
        return " ".join(parts).strip()

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
