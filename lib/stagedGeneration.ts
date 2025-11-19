import OpenAI from "openai";
import { PersonaProfile } from "./personaBuilder";
import { EventMemory } from "./eventStore";
import { PersonaFingerprint } from "./personaFingerprint";
import { SpeakingRules } from "./personaRules";
import { LanguageMode } from "./languageRouter";

export interface ContentPlan {
  draft: string;
}

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function getMaxTokensForPersona(persona: PersonaProfile): number {
  const base = (() => {
    switch (persona.tone.typicalLength) {
      case "very_short":
        return 40;
      case "short":
        return 60;
      case "medium":
        return 120;
      case "long":
      default:
        return 200;
    }
  })();
  const multiplier = (persona as any).overrides?.maxTokensMultiplier ?? 1;
  return Math.max(16, Math.floor(base * multiplier));
}

export async function generateContentPlan(params: {
  persona: PersonaProfile;
  userMessage: string;
  events: EventMemory[];
  memories: { text: string }[];
}): Promise<ContentPlan> {
  const { persona, userMessage, events, memories } = params;
  const eventLines = events
    .slice(0, 3)
    .map(
      (e) =>
        `Situation: ${e.situationDescription}. Reaction: ${e.reactionDescription}. Phrases: ${e.phrasesUsed.join(
          "; "
        )}`
    )
    .join("\n");
  const memLines = memories
    .slice(0, 3)
    .map((m) => `- ${m.text}`)
    .join("\n");

  const prompt = `
You are creating a neutral, semantic content plan for a persona's reply.
- Summarize what the persona would want to say in response to the user message.
- Optionally reference relevant events/memories briefly.
- Do NOT imitate style, tone, or language. Use plain English.
- Avoid therapy, motivational speeches, or emotional flourish.
- Keep it 2–4 sentences max.

Persona: ${persona.name || "Unknown"}, relationship: ${persona.relationshipToUser || "unknown"}
User message: ${userMessage}
Events:\n${eventLines || "(none)"}
Memories:\n${memLines || "(none)"}

Return JSON: { "draft": "<neutral draft>" }
`.trim();

  const completion = await openai.chat.completions.create({
    model: process.env.OPENAI_MODEL_PLAN || "gpt-4o-mini",
    temperature: 0.2,
    max_tokens: 150,
    response_format: { type: "json_object" },
    messages: [{ role: "system", content: prompt }],
  });
  const raw = completion.choices[0]?.message?.content || "{}";
  try {
    const parsed = JSON.parse(raw);
    return { draft: parsed.draft || "" };
  } catch {
    return { draft: raw };
  }
}

function enforceSpeakingRules(reply: string, rules: SpeakingRules): string {
  let trimmed = reply.trim();
  if (rules.energy === "low") {
    trimmed = trimmed.replace(/!/g, ".");
  }

  const sentences = trimmed.split(/(?<=[.!?])/).filter((s) => s.trim());
  if (sentences.length > rules.maxSentences) {
    trimmed = sentences.slice(0, rules.maxSentences).join(" ").trim();
  }

  const tokens = trimmed.split(/\s+/).filter(Boolean);
  if (tokens.length > rules.maxTokens) {
    trimmed = tokens.slice(0, rules.maxTokens).join(" ").trim();
  }

  if (rules.requiredMarkers?.length && !rules.requiredMarkers.some((m) => trimmed.toLowerCase().includes(m.toLowerCase()))) {
    trimmed = `${rules.requiredMarkers[0]} ${trimmed}`.trim();
  }
  return trimmed;
}

export async function rewriteInPersonaStyle(params: {
  persona: PersonaProfile;
  draft: string;
  userMessage: string;
  fingerprint: PersonaFingerprint;
  rules: SpeakingRules;
  targetLanguage: LanguageMode;
  temperature?: number;
  attempt?: number;
}): Promise<string> {
  const { persona, draft, userMessage, fingerprint, rules, targetLanguage, temperature = 0.3, attempt = 0 } = params;
  const maxTokens = getMaxTokensForPersona(persona);
  const templateLines = fingerprint.sentenceTemplates.map((t) => `- ${t}`).join("\n");
  const filler = fingerprint.fillerWords.join(", ") || "(none)";
  const markerLine = rules.requiredMarkers?.length ? rules.requiredMarkers.join(", ") : "(none)";
  const bannedLine = rules.bannedPhrases.length ? rules.bannedPhrases.join(", ") : "(global banned)";

  const tighten = attempt > 0 ? "Previous reply was too generic or long; now respond even shorter and calmer." : "";

  const prompt = `
Rewrite the draft strictly in the persona's voice.
- Target language: ${targetLanguage}. Never switch languages.
- Hard limits: max ${rules.maxSentences} sentences, ${rules.maxTokens} tokens.
- ${rules.forbidQuestions ? "Do not ask any question." : "Ask a question only if the user explicitly asked one."}
- Maintain energy=${rules.energy}; if low, stay flat and calm.
- Required markers: ${markerLine} (use at least one naturally).
- Avoid these phrases entirely: ${bannedLine}.
- Use these sentence templates and pacing (but keep it natural):
${templateLines}
- Use filler words sparingly: ${filler}.
- Use catchphrases/common phrases only when organic.
- Remove therapy tone, poetic flourishes, or motivational speech.
- Obey the fingerprint typical length: ${fingerprint.typicalLength}.
- Speak like a real imperfect human from their background.
- Do NOT add content beyond the draft ideas.
${tighten}

Draft:
${draft}

User message:
${userMessage}
`.trim();

  // TODO: If a per-persona LoRA fine-tuned model is available, plug it in here.
  const completion = await openai.chat.completions.create({
    model: process.env.OPENAI_MODEL_CHAT || "gpt-5.1",
    temperature,
    top_p: 0.85,
    max_tokens: maxTokens,
    messages: [{ role: "system", content: prompt }],
  });
  const raw = (completion.choices[0]?.message?.content || "").trim();
  return enforceSpeakingRules(raw, rules);
}

export interface StagedGenerateParams {
  persona: PersonaProfile;
  fingerprint: PersonaFingerprint;
  rules: SpeakingRules;
  targetLanguage: LanguageMode;
  userMessage: string;
  events: EventMemory[];
  memories: { text: string }[];
  temperatures?: number[];
  maxAttempts?: number;
}

export interface StagedGenerateResult {
  draft: string;
  candidates: string[];
}

export async function stagedGenerate(params: StagedGenerateParams): Promise<StagedGenerateResult> {
  const {
    persona,
    fingerprint,
    rules,
    targetLanguage,
    userMessage,
    events,
    memories,
    temperatures = [0.3, 0.35, 0.4],
    maxAttempts = 2,
  } = params;

  const plan = await generateContentPlan({ persona, userMessage, events, memories });
  const candidates: string[] = [];

  for (const baseTemp of temperatures) {
    let attempt = 0;
    let reply = "";
    while (attempt <= maxAttempts) {
      reply = await rewriteInPersonaStyle({
        persona,
        draft: plan.draft,
        userMessage,
        fingerprint,
        rules,
        targetLanguage,
        temperature: baseTemp - attempt * 0.05,
        attempt,
      });
      if (reply) break;
      attempt += 1;
    }
    candidates.push(reply);
  }

  return { draft: plan.draft, candidates };
}
