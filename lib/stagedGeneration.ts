import OpenAI from "openai";
import { PersonaProfile } from "./personaBuilder";
import { EventMemory } from "./eventStore";

export interface ContentPlan {
  draft: string;
}

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function getMaxTokensForPersona(persona: PersonaProfile): number {
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

export async function rewriteInPersonaStyle(params: {
  persona: PersonaProfile;
  draft: string;
  userMessage: string;
  temperature?: number;
}): Promise<string> {
  const { persona, draft, userMessage, temperature = 0.3 } = params;
  const maxTokens = getMaxTokensForPersona(persona);
  const prompt = `
Rewrite the draft in the persona's voice, language, and tone.
- Reply ONLY in ${persona.language}.
- Match energy=${persona.tone.energy} and keep length very short for short/very_short personas.
- Use catchphrases only when natural.
- Avoid follow-up questions unless the user asked a direct question.
- Avoid therapy, motivational speeches, or GPT clichés.
- Do not add new ideas beyond the draft.
- Never mention being an AI. No generic chatbot lines.

Draft: ${draft}
User message: ${userMessage}
`.trim();

  const completion = await openai.chat.completions.create({
    model: process.env.OPENAI_MODEL_CHAT || "gpt-5.1",
    temperature,
    top_p: 0.9,
    max_tokens: maxTokens,
    messages: [{ role: "system", content: prompt }],
  });
  return (completion.choices[0]?.message?.content || "").trim();
}
