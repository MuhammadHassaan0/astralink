import OpenAI from "openai";
import { PersonaProfile } from "./personaBuilder";
import { Memory } from "./memoryStore";
import { BASE_SYSTEM_PROMPT } from "./prompts";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const BANNED = [
  "i'm here for you",
  "i am here for you",
  "how are you doing today",
  "what’s on your mind",
  "what's on your mind",
  "how have you been holding up",
  "how’s everything going",
  "how's everything going",
  "i’m really sorry to hear that",
  "i am really sorry to hear that",
  "want to talk about what’s bothering",
  "want to talk about what's bothering",
  "is there something specific you’re worried",
  "is there something specific you're worried",
  "take a deep breath",
  "one step at a time",
  "proud of you",
  "always here for you",
  "you’re never alone",
  "you're never alone",
];

const POETIC_WORDS = [
  "cherished",
  "precious",
  "echoes",
  "warmth",
  "gentle",
  "breeze",
  "ethereal",
  "everlasting",
  "whisper",
  "glow",
  "radiant",
  "embrace",
];

const LENGTH_LIMIT: Record<PersonaProfile["tone"]["typicalLength"], number> = {
  very_short: 40,
  short: 60,
  medium: 90,
  long: 120,
};

function applyPersonaDefaults(persona: PersonaProfile): PersonaProfile {
  return {
    ...persona,
    language: persona.language || "ur",
    tone: {
      energy: persona.tone?.energy || "low",
      warmth: persona.tone?.warmth || "medium",
      humor: persona.tone?.humor || "none",
      typicalLength: persona.tone?.typicalLength || "very_short",
    },
    responseRules: persona.responseRules || [],
  } as PersonaProfile;
}

function tokenize(text: string): string[] {
  return text.split(/\s+/).filter(Boolean);
}

function trimToLength(reply: string, maxTokens: number): string {
  const tokens = tokenize(reply);
  if (tokens.length <= maxTokens) return reply;
  return tokens.slice(0, maxTokens).join(" ");
}

function bannedPhraseFound(reply: string): boolean {
  const low = reply.toLowerCase();
  return BANNED.some((p) => low.includes(p));
}

function containsTooManyQuestions(reply: string, allow: boolean): boolean {
  const count = (reply.match(/\?/g) || []).length;
  if (!allow && count > 0) return true;
  if (allow && count > 1) return true;
  return false;
}

function validateLanguage(reply: string, persona: PersonaProfile): boolean {
  if (persona.language === "ur") {
    const words = tokenize(reply);
    if (!words.length) return true;
    const englishWords = words.filter((w) => /^[a-zA-Z]+$/.test(w)).length;
    return englishWords / words.length <= 0.15;
  }
  return true;
}

function validateTone(reply: string, persona: PersonaProfile): boolean {
  const low = reply.toLowerCase();
  if (persona.tone.energy === "low") {
    if (reply.includes("!")) return false;
    const wellness = ["wellness", "coach", "breathe", "relax", "calm down"];
    if (wellness.some((w) => low.includes(w))) return false;
  }
  return true;
}

function tooPoetic(reply: string): boolean {
  const words = tokenize(reply);
  if (!words.length) return false;
  const poetic = words.filter((w) =>
    POETIC_WORDS.includes(w.replace(/[^a-z]/gi, "").toLowerCase())
  ).length;
  return poetic / words.length > 0.12;
}

function isTherapyLike(reply: string): boolean {
  const low = reply.toLowerCase();
  const therapist = [
    "therapy",
    "therapist",
    "counseling",
    "life coach",
    "deep breath",
    "one step at a time",
    "cope",
    "healing",
    "mindfulness",
  ];
  return therapist.some((t) => low.includes(t));
}

async function regenerate(
  persona: PersonaProfile,
  memories: Memory[],
  userMessage: string,
  maxTokens: number,
  temperature: number,
  override: string
) {
  const systemPrompt = BASE_SYSTEM_PROMPT.replace(
    "{{RELATIONSHIP}}",
    persona.relationshipToUser || "loved one"
  )
    .replace("{{LANGUAGE_NAME}}", persona.language || "en")
    .concat(
      "\nRespond only in the persona’s primary language.\nKeep replies extremely short unless user explicitly requests a long response.\nDo not ask follow-up questions unless the user asked a question first.\nDo not provide therapy, motivational speeches, or emotional coaching.\nDo not respond with generic GPT phrases.\nAvoid emotional flourish. Keep tone aligned with persona’s energy and warmth.\nRespond like a real human with imperfections — quieter, understated, minimal.\nNever say you are an AI or hint at being artificial.\nNever switch languages.\n",
      override
    );

  const memorySnippets = memories
    .map((m) => `- ${m.text}`)
    .slice(0, 5)
    .join("\n");
  const messages = [
    { role: "system", content: systemPrompt },
    { role: "system", content: `PERSONA:\n${JSON.stringify(persona)}` },
    {
      role: "system",
      content: `MEMORIES (use lightly, one line max):\n${memorySnippets || "(none)"}`,
    },
    { role: "user", content: userMessage },
  ];

  const completion = await openai.chat.completions.create({
    model: process.env.OPENAI_MODEL_CHAT || "gpt-5.1",
    temperature,
    max_tokens: maxTokens,
    messages,
  });
  return (completion.choices[0]?.message?.content || "").trim();
}

export async function generatePersonaReply({
  persona,
  memories,
  userMessage,
}: {
  persona: PersonaProfile;
  memories: Memory[];
  userMessage: string;
}): Promise<{ reply: string; modelUsed: string; usedFallback: boolean }> {
  const adjustedPersona = applyPersonaDefaults(persona);
  const baseLimit = LENGTH_LIMIT[adjustedPersona.tone.typicalLength] || 90;
  const model = process.env.OPENAI_MODEL_CHAT || "gpt-5.1";

  const allowQuestions =
    userMessage.includes("?") || /advice|help|what should|how do/i.test(userMessage);

  const attempt = async (limit: number, temp: number, extra = "") => {
    let reply = await regenerate(adjustedPersona, memories, userMessage, limit, temp, extra);
    reply = trimToLength(reply, limit);
    if (bannedPhraseFound(reply)) return { ok: false, reply, reason: "banned" };
    if (!validateLanguage(reply, adjustedPersona)) return { ok: false, reply, reason: "language" };
    if (!validateTone(reply, adjustedPersona)) return { ok: false, reply, reason: "tone" };
    if (tooPoetic(reply)) return { ok: false, reply, reason: "poetic" };
    if (isTherapyLike(reply)) return { ok: false, reply, reason: "therapy" };
    if (containsTooManyQuestions(reply, allowQuestions))
      return { ok: false, reply, reason: "questions" };
    return { ok: true, reply, reason: "" };
  };

  const first = await attempt(baseLimit, 0.4);
  if (first.ok) return { reply: first.reply, modelUsed: model, usedFallback: false };
  const second = await attempt(Math.max(20, baseLimit - 20), 0.25, "Keep it extremely short and low-energy. No questions.");
  if (second.ok) return { reply: second.reply, modelUsed: model, usedFallback: true };
  const thirdReply = await regenerate(
    adjustedPersona,
    memories,
    userMessage,
    Math.max(15, baseLimit - 30),
    0.15,
    "Keep it extremely short and low-energy. No questions."
  );
  return {
    reply: trimToLength(thirdReply, Math.max(15, baseLimit - 30)),
    modelUsed: model,
    usedFallback: true,
  };
}
