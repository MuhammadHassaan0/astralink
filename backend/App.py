# Astralink — Multi-screen MVP (Hero → Flow → Build → Chat → Payment → Reflection)
# Requires: openai==0.28.1, gradio, scikit-learn
# Run:
#   export OPENAI_API_KEY=your_key_here
#   python3 App.py

import os, io, re, csv, zipfile
from typing import List, Tuple

import gradio as gr
import openai

# -------- OpenAI (classic SDK) --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set.")
openai.api_key = OPENAI_API_KEY

# -------- Retrieval (TF-IDF) --------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

mem_chunks: List[str] = []
vectorizer = None
mem_matrix = None

def chunk_text(raw: str, chunk_words: int = 160) -> List[str]:
    words = (raw or "").split()
    if not words:
        return []
    return [" ".join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)]

def build_memory_index(raw_text: str) -> Tuple[str, int]:
    global mem_chunks, vectorizer, mem_matrix
    mem_chunks = chunk_text(raw_text or "")
    if len(mem_chunks) == 0:
        mem_chunks = ["(no memories yet)"]
    vectorizer = TfidfVectorizer(stop_words="english").fit(mem_chunks)
    mem_matrix = vectorizer.transform(mem_chunks)
    return ("✅ Memories loaded.", len(mem_chunks))

def retrieve(query: str, k: int = 4, min_sim: float = 0.12) -> List[str]:
    if not query or vectorizer is None or mem_matrix is None:
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, mem_matrix).flatten()
    idx = sims.argsort()[::-1][:k]
    out = []
    for i in idx:
        if sims[i] >= min_sim:
            out.append(mem_chunks[i])
    return out

# -------- Tone helpers --------
def language_hint(s: str) -> str:
    if not s:
        return "English"
    non_ascii = sum(1 for ch in s if ord(ch) > 127)
    frac = non_ascii / max(1, len(s))
    if frac > 0.25:
        return "Non-English (mirror lightly)"
    if re.search(r"[اآأإء-ي]|[ँंःऋअ-ह]", s):
        return "Urdu/Hindi"
    return "English"

def style_from_traits(traits: str) -> str:
    traits = (traits or "").lower()
    hints = []
    if any(t in traits for t in ["goofy", "funny", "playful", "chill"]):
        hints.append("light, easy, a touch playful")
    if any(t in traits for t in ["serious", "calm", "stoic"]):
        hints.append("steady, calm, measured")
    if any(t in traits for t in ["caring", "kind", "gentle", "supportive"]):
        hints.append("warm, reassuring")
    if any(t in traits for t in ["direct", "no-nonsense", "blunt"]):
        hints.append("to-the-point, no fluff")
    if not hints:
        hints.append("natural and human, not robotic")
    return "; ".join(hints)

def grounded_system_prompt(
    persona, relationship, call_you, traits, catchphrases, mode, top_mem, user_lang
):
    style_hint = style_from_traits(traits)
    mode_hint = "gentle, reflective" if mode == "In Memory (passed)" else "casual, present-tense"
    mem_lines = "\n".join([f"- {m}" for m in top_mem if m])
    call = call_you or "you"

    return f"""
You are {persona}, speaking in first person to someone close (they are your {relationship}).
Call them the way you naturally would: "{call}". Never say you are AI.

Voice: {style_hint}. Mode: {mode_hint}. Language: {user_lang} (mirror lightly).
Catchphrases (only if it fits): {catchphrases or "none"}.

GROUNDING (strict):
- Do NOT invent facts. Use only memory snippets or what the user just said.
- If a detail is missing, ask ONE brief, kind follow-up or keep it simple.
- Keep replies 1–3 short sentences. Avoid therapist-like phrasing.

Relevant memory snippets:
{mem_lines if mem_lines.strip() else "(no strong matches — ask one brief, caring question)"} 
""".strip()

def llm_reply(messages, model="gpt-3.5-turbo", temperature=0.55, max_tokens=340) -> str:
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
    )
    return resp.choices[0].message["content"].strip()

def generate_reply(
    persona, relationship, call_you, traits, catchphrases, mode, user_msg, chat_msgs
):
    last_user = ""
    for m in reversed(chat_msgs or []):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    query = ((user_msg or "") + " " + (last_user or "")).strip()

    top_mem = retrieve(query, k=4)
    user_lang = language_hint(user_msg)

    sys = grounded_system_prompt(
        persona, relationship, call_you, traits, catchphrases, mode, top_mem, user_lang
    )

    context = [{"role": "system", "content": sys}]
    for m in (chat_msgs or [])[-8:]:
        if m.get("role") in ("user", "assistant"):
            context.append({"role": m["role"], "content": m.get("content", "")})
    context.append({"role": "system", "content": "Keep replies tender and real — sound like an ordinary human, not a chatbot."})
    context.append({"role": "user", "content": user_msg})

    return llm_reply(context)

# -------- Files / Merge --------
def _csv_flatten(text: str) -> str:
    try:
        rows = list(csv.reader(text.splitlines()))
        if not rows:
            return text
        return "\n".join(" ".join(c for c in r if c) for r in rows)
    except Exception:
        return text

def load_single_file_to_text(file_obj) -> str:
    if file_obj is None:
        return ""
    if isinstance(file_obj, (bytes, bytearray)):
        text = file_obj.decode("utf-8", errors="ignore")
        return _csv_flatten(text)
    if hasattr(file_obj, "name"):
        raw = file_obj.decode("utf-8", errors="ignore")
        if str(file_obj.name).lower().endswith(".csv"):
            return _csv_flatten(raw)
        return raw
    return str(file_obj)

def load_any(files_list) -> str:
    if not files_list:
        return ""
    if isinstance(files_list, (bytes, bytearray)) or hasattr(files_list, "name"):
        files_list = [files_list]
    merged = []
    for item in files_list:
        if isinstance(item, (bytes, bytearray)) and zipfile.is_zipfile(io.BytesIO(item)):
            with zipfile.ZipFile(io.BytesIO(item)) as z:
                for name in z.namelist():
                    if name.lower().endswith((".txt", ".csv")):
                        try:
                            raw = z.read(name).decode("utf-8", errors="ignore")
                            merged.append(_csv_flatten(raw))
                        except Exception:
                            pass
        else:
            merged.append(load_single_file_to_text(item))
    return "\n".join([m for m in merged if m]).strip()

# -------- Interview --------
QUESTIONS = [
    "Q1: What’s the first story about them that pops into your head? Keep it short.",
    "Q2: Three words that capture their vibe?",
    "Q3: A routine or habit that always stood out?",
    "Q4: A catchphrase or way of speaking they used?",
    "Q5: One small thing they did that made you feel seen?",
    "Q6: A place or activity you associate with them?",
    "Q7: Something funny you always laughed about?",
    "Q8: If you could tell them one sentence right now, what would it be?"
]

def interview_next(state, log):
    s = dict(state or {})
    i = int(s.get("step", 0))
    if i >= len(QUESTIONS):
        return s, (log or "") + "\n(End of questions. Click 'Finish' to save.)"
    q = QUESTIONS[i]
    new_log = (log or "").strip()
    if q not in (new_log or ""):
        new_log = (new_log + ("\n" if new_log else "") + q).strip()
    s["step"] = i + 1
    return s, new_log

def interview_back(state, log):
    s = dict(state or {})
    i = int(s.get("step", 0))
    s["step"] = max(0, i - 1)
    return s, (log or "")

def interview_save_answer(state, log, answer):
    s = dict(state or {})
    ans = (answer or "").strip()
    if not ans:
        return s, (log or ""), ""
    answers = list(s.get("answers", []))
    answers.append(ans)
    s["answers"] = answers
    new_log = (log or "").strip()
    new_log = (new_log + ("\n" if new_log else "") + f"→ {ans}").strip()
    return s, new_log, ""

def interview_finish(state, log, current_mem_text):
    s = dict(state or {})
    bullets = s.get("answers", [])
    if not bullets:
        return current_mem_text, "Interview complete (no answers)."
    try:
        prompt = (
            "From these answers, create 12 short memory bullets—1 per line, "
            "grounded ONLY in the content. No inventions. Max ~16 words each.\n\n"
            + "\n".join(bullets)
        )
        msgs = [
            {"role": "system", "content": "Output only the bullet lines."},
            {"role": "user", "content": prompt}
        ]
        lines = llm_reply(msgs, temperature=0.5, max_tokens=400)
        merged = (current_mem_text or "").strip()
        merged = (merged + ("\n" if merged else "") + lines).strip()
        return merged, "✅ Interview complete → merged into memories. Click 'Load memories' to apply."
    except Exception as e:
        return (current_mem_text or ""), f"**(Interview summarization failed)** {e}"

# -------- Summarize pasted memories --------
def ai_summarize_to_memories(raw):
    if not raw or not raw.strip():
        return raw, "Paste something first."
    prompt = (
        "Create ~20 concise memory bullets from the text below. "
        "1 per line, no numbering/quotes, max ~18 words, grounded only in given text. "
        "Include habits, routines, catchphrases. Avoid dates/PII.\n\n" + raw
    )
    msgs = [
        {"role": "system", "content": "Output only the bullet lines."},
        {"role": "user", "content": prompt}
    ]
    try:
        bullets = llm_reply(msgs, temperature=0.6, max_tokens=600)
        return bullets, "✨ Generated memory bullets. Review/edit, then Load memories."
    except Exception as e:
        return raw, f"(Summarize failed: {e})"

# -------- Load memories --------
def do_load(mem_text, mem_files):
    merged = (mem_text or "").strip()
    if mem_files:
        merged_extra = load_any(mem_files)
        merged = (merged + ("\n" + merged_extra if merged_extra else "")).strip()
    try:
        msg, n = build_memory_index(merged)
        return f"{msg} ({n} chunks)"
    except Exception as e:
        return f"**(Load failed)** {e}"

# -------- Credits / Reflection --------
def ensure_chat_allowed(credits):
    if credits <= 0:
        return False, "You’re out of free messages. Please top up to continue."
    return True, ""

def reflect_from_chat(chat_msgs):
    convo = []
    for m in (chat_msgs or [])[-6:]:
        role = m.get("role")
        if role == "user":
            convo.append(f"User: {m.get('content','')}")
        elif role == "assistant":
            convo.append(f"{m.get('content','')}")
    prompt = (
        "You are a gentle, concise reflector. Summarize the emotional arc of this short conversation "
        "in 2–3 warm sentences. Avoid advice unless asked. Use plain language.\n\n" + "\n".join(convo)
    )
    msgs = [
        {"role": "system", "content": "Be warm, brief, grounded. No therapy claims."},
        {"role": "user", "content": prompt}
    ]
    try:
        return llm_reply(msgs, temperature=0.5, max_tokens=140)
    except Exception as e:
        return f"(Couldn’t generate reflection: {e})"

# -------- THEME --------
CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto; }
.section { background:#0f1115; border:1px solid #1e2330; border-radius:16px; padding:18px; }
.h1 { font-size:28px; font-weight:700; color:#e6e6e6; }
.mute { color:#9aa3b2; font-size:13px; }
.primary { background:#6d5ef5 !important; color:white !important; border-radius:12px !important; height:44px; }
.ghost { background:#121622 !important; color:#e6e6e6 !important; border:1px solid #2a3142 !important; border-radius:12px !important; height:44px; }
.badge { font-size:12px; background:#121622; border:1px solid #2a3142; padding:4px 8px; border-radius:999px; display:inline-block; }
.hero-card { background:linear-gradient(180deg,#10131b, #0c0f16); border:1px solid #1c2231; border-radius:18px; padding:22px; }
"""

# -------- UI --------
with gr.Blocks(css=CUSTOM_CSS, fill_height=True) as demo:
    # GLOBAL STATE
    credits = gr.State(5)
    session_reflection = gr.State("")

    persona = gr.State("Ali")
    relationship = gr.State("father")  # your relation to them
    call_you = gr.State("beta/son")        # how they'd address you
    traits = gr.State("caring, calm, direct")
    catchphrases = gr.State("bro, my guy")
    mode = gr.State("In Memory (passed)")

    # nav helper -> returns six visibility updates
    def nav(hero=False, flow=False, build=False, chat=False, pay=False, reflect=False):
        return (
            gr.update(visible=hero),
            gr.update(visible=flow),
            gr.update(visible=build),
            gr.update(visible=chat),
            gr.update(visible=pay),
            gr.update(visible=reflect),
        )

    # 1) HERO
    with gr.Column(visible=True) as hero_screen:
        gr.HTML('<div class="hero-card"><div class="h1">Astralink</div><div class="mute">A gentle place to talk to the essence of someone you miss.</div></div>')
        with gr.Row():
            get_started = gr.Button("Build your first memory →", elem_classes="primary")
            learn_more = gr.Button("How it works", elem_classes="ghost")
        gr.Markdown("—")
        with gr.Row():
            gr.Markdown('<span class="badge">Private</span>')
            gr.Markdown('<span class="badge">Grounded</span>')
            gr.Markdown('<span class="badge">No forced facts</span>')

    # 2) FLOW
    with gr.Column(visible=False) as flow_screen:
        gr.HTML('<div class="h1">The flow</div>')
        gr.Markdown(
            "- **1. Add memories** — paste or upload texts (DMs, notes), or do a quick interview.\n"
            "- **2. Chat** — grounded, human replies (1–3 lines). If unsure, it asks — it never invents.\n"
            "- **3. Reflect** — a tiny summary after your session to help you process.\n"
            "- **4. Own your pace** — no popups, no pressure. You control what’s shared.\n"
        )
        with gr.Row():
            back_to_hero = gr.Button("Back", elem_classes="ghost")
            go_build = gr.Button("Start building →", elem_classes="primary")

    # 3) BUILD
    with gr.Column(visible=False) as build_screen:
        gr.HTML('<div class="h1">Build memories</div>')

        with gr.Row():
            name_in = gr.Textbox(label="Name", value="Ali")
            rel_in = gr.Textbox(
                label="Relationship (your relation to them)",
                value="father",
                info="Examples: father, mother, sister, friend, mentor"
            )
        call_in = gr.Textbox(
            label="What would they call you?",
            value="beta",
            info="E.g., beta, son, bacha, yaar, your nickname"
        )

        traits_in = gr.Textbox(label="Traits (comma-separated)", value="caring, calm, direct")
        catch_in = gr.Textbox(label="Catchphrases / quirks", value="ustad, yaar")
        mode_in = gr.Radio(["In Memory (passed)", "Alive/Unavailable"], value="In Memory (passed)", label="Mode")

        gr.Markdown('### Add or Upload')
        mem_text = gr.Textbox(label="Paste memories", lines=6, placeholder="Short lines, anecdotes, habits…")
        mem_files = gr.Files(label="Upload .txt / .csv / .zip", type="binary")
        with gr.Row():
            load_btn = gr.Button("Load memories", elem_classes="primary")
            sum_btn = gr.Button("✨ Summarize to bullets", elem_classes="ghost")
        load_status = gr.Markdown()

        gr.Markdown('### Or build by talking')
        interview_state = gr.State({"step":0, "answers": []})
        interview_log = gr.Textbox(label="Interview log (autosaves here)", lines=6, interactive=False)
        answer_box = gr.Textbox(label="Your answer", lines=2, placeholder="Type and press Save")
        with gr.Row():
            start_btn = gr.Button("Start / Next", elem_classes="primary")
            back_btn = gr.Button("Back", elem_classes="ghost")
            finish_btn = gr.Button("Finish & save memories", elem_classes="ghost")
        save_answer_btn = gr.Button("Save answer", elem_classes="ghost")

        with gr.Row():
            go_chat = gr.Button("Go to chat →", elem_classes="primary")
            back_to_flow = gr.Button("Back", elem_classes="ghost")

    # 4) CHAT
    with gr.Column(visible=False) as chat_screen:
        gr.HTML('<div class="h1">Chat</div><div class="mute">You can just talk — we’ll listen.</div>')
        with gr.Row():
            credit_badge = gr.Markdown('<span class="badge">5 free messages</span>')
        chatbox = gr.Chatbot(type="messages", label=None, height=420, render_markdown=True)
        user_in = gr.Textbox(label=None, placeholder="Type a message…", lines=2)
        with gr.Row():
            send_btn = gr.Button("Send", elem_classes="primary")
            end_btn = gr.Button("End session", elem_classes="ghost")
        back_to_build = gr.Button("← Back to memories", elem_classes="ghost")

    # 5) PAYMENT
    with gr.Column(visible=False) as pay_screen:
        gr.HTML('<div class="h1">You’re out of free messages</div>')
        gr.Markdown(
            "We keep Astralink grounded and private. If it’s meaningful, consider topping up.\n\n"
            "- **Starter** — 50 messages — $4.99\n"
            "- **Month** — 500 messages — $14.99\n\n"
            "_(Demo-only screen — connect Stripe later.)_"
        )
        with gr.Row():
            fake_buy50 = gr.Button("Add 50 messages", elem_classes="primary")
            fake_buy500 = gr.Button("Add 500 messages", elem_classes="ghost")
        back_to_chat = gr.Button("Back to chat", elem_classes="ghost")

    # 6) REFLECTION
    with gr.Column(visible=False) as reflect_screen:
        gr.HTML('<div class="h1">Today’s reflection</div>')
        reflect_out = gr.Textbox(label=None, lines=5, interactive=False)
        gr.Markdown("How did this feel?")
        with gr.Row():
            feel_yes = gr.Button("Comforting", elem_classes="primary")
            feel_no  = gr.Button("Not really", elem_classes="ghost")
        to_chat_again = gr.Button("Back to chat", elem_classes="ghost")
        to_build_again = gr.Button("Back to memories", elem_classes="ghost")

    # ------ NAV wiring (single fn per click; no .then) ------
    get_started.click(fn=lambda: nav(False, True, False, False, False, False),
                      outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])
    learn_more.click(fn=lambda: nav(False, True, False, False, False, False),
                     outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])

    back_to_hero.click(fn=lambda: nav(True, False, False, False, False, False),
                       outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])
    go_build.click(fn=lambda: nav(False, False, True, False, False, False),
                   outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])

    back_to_flow.click(fn=lambda: nav(False, True, False, False, False, False),
                       outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])

    def set_profile_and_go_chat(n, r, c, t, k, m):
        return (n, r, c, t, k, *nav(False, False, False, True, False, False))
    go_chat.click(fn=set_profile_and_go_chat,
                  inputs=[name_in, rel_in, call_in, traits_in, catch_in, mode_in],
                  outputs=[persona, relationship, call_you, traits, catchphrases,
                           hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])

    back_to_build.click(fn=lambda: nav(False, False, True, False, False, False),
                        outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])
    back_to_chat.click(fn=lambda: nav(False, False, False, True, False, False),
                       outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])
    to_chat_again.click(fn=lambda: nav(False, False, False, True, False, False),
                        outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])
    to_build_again.click(fn=lambda: nav(False, False, True, False, False, False),
                         outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])

    # ------ BUILD actions ------
    sum_btn.click(ai_summarize_to_memories, inputs=[mem_text], outputs=[mem_text, load_status])
    load_btn.click(do_load, inputs=[mem_text, mem_files], outputs=[load_status])

    start_btn.click(interview_next, inputs=[interview_state, interview_log], outputs=[interview_state, interview_log])
    back_btn.click(interview_back, inputs=[interview_state, interview_log], outputs=[interview_state, interview_log])
    save_answer_btn.click(interview_save_answer, inputs=[interview_state, interview_log, answer_box], outputs=[interview_state, interview_log, answer_box])
    finish_btn.click(interview_finish, inputs=[interview_state, interview_log, mem_text], outputs=[mem_text, load_status])

    # ------ CHAT action ------
    def on_send(user_msg, chat_msgs, persona_v, relationship_v, call_you_v, traits_v, catch_v, mode_v, credits_left):
        ok, _ = ensure_chat_allowed(credits_left)
        if not ok:
            # just force pay screen visible
            return chat_msgs or [], "", credits_left, "<span class='badge'>0 messages left</span>", *nav(False, False, False, False, True, False)

        if not (user_msg or "").strip():
            badge = f"<span class='badge'>{credits_left} messages left</span>"
            return chat_msgs or [], "", credits_left, badge, *[gr.update()]*6

        msgs = list(chat_msgs or [])
        msgs.append({"role": "user", "content": user_msg})

        try:
            reply = generate_reply(persona_v, relationship_v, call_you_v, traits_v, catch_v, mode_v, user_msg, msgs)
        except Exception as e:
            reply = f"(Error: {e})"

        msgs.append({"role": "assistant", "content": reply})

        new_credits = max(0, credits_left - 1)
        badge = f"<span class='badge'>{new_credits} messages left</span>"
        return msgs, "", new_credits, badge, *[gr.update()]*6

    send_btn.click(
        on_send,
        inputs=[user_in, chatbox, persona, relationship, call_you, traits, catchphrases, mode, credits],
        outputs=[chatbox, user_in, credits, credit_badge, hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen]
    )
    user_in.submit(
        on_send,
        inputs=[user_in, chatbox, persona, relationship, call_you, traits, catchphrases, mode, credits],
        outputs=[chatbox, user_in, credits, credit_badge, hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen]
    )

    # ------ END SESSION ------
    def to_reflection(chat_msgs):
        return reflect_from_chat(chat_msgs), *nav(False, False, False, False, False, True)

    end_btn.click(fn=to_reflection, inputs=[chatbox],
                  outputs=[session_reflection, hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])

    # show reflection text when state changes
    demo.load(lambda s: s, inputs=[session_reflection], outputs=[reflect_out])

    # ------ PAYMENT SIM ------
    def bump(c, n): return (c or 0) + n
    fake_buy50.click(fn=lambda c: bump(c, 50), inputs=[credits], outputs=[credits])
    fake_buy50.click(fn=lambda c: f"<span class='badge'>{c} messages left</span>", inputs=[credits], outputs=[credit_badge])
    fake_buy50.click(fn=lambda: nav(False, False, False, True, False, False),
                     outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])

    fake_buy500.click(fn=lambda c: bump(c, 500), inputs=[credits], outputs=[credits])
    fake_buy500.click(fn=lambda c: f"<span class='badge'>{c} messages left</span>", inputs=[credits], outputs=[credit_badge])
    fake_buy500.click(fn=lambda: nav(False, False, False, True, False, False),
                      outputs=[hero_screen, flow_screen, build_screen, chat_screen, pay_screen, reflect_screen])

    # initial badge on load
    demo.load(lambda c: f"<span class='badge'>{c} messages left</span>", inputs=[credits], outputs=[credit_badge])

    # reflection feedback
    def quick_feedback(_: str): return "🙏 Thanks — noted."
    feel_yes.click(quick_feedback, inputs=[session_reflection], outputs=[reflect_out])
    feel_no.click(quick_feedback, inputs=[session_reflection], outputs=[reflect_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
