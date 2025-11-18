import OpenAI from "openai";
import { PersonaProfile } from "./personaBuilder";
import { Memory } from "./memoryStore";
import { BASE_SYSTEM_PROMPT } from "./prompts";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const LENGTH_MAP: Record<PersonaProfile["tone"]["typicalLength"], [number, number]> =
  {
    very_short: [30, 70],
    short: [40, 90],
    medium: [60, 130],
    long: [90, 180],
  };

export async function generatePersonaReply({
  persona,
  memories,
  userMessage,
}: {
  persona: PersonaProfile;
  memories: Memory[];
  userMessage: string;
}): Promise<{ reply: string; modelUsed: string; usedFallback: boolean }> {
  const toneRange = LENGTH_MAP[persona.tone.typicalLength] || [60, 130];
  const maxTokens = Math.round((toneRange[0] + toneRange[1]) / 2);
  const systemPrompt = BASE_SYSTEM_PROMPT.replace(
    "{{RELATIONSHIP}}",
    persona.relationshipToUser || "loved one"
  ).replace("{{LANGUAGE_NAME}}", persona.language || "en");

  const personaJson = JSON.stringify(persona);
  const memorySnippets = memories
    .map((m) => `- ${m.text}`)
    .join("\n")
    .slice(0, 2000);

  const messages = [
    { role: "system", content: systemPrompt },
    { role: "system", content: `PERSONA:\n${personaJson}` },
    { role: "system", content: `MEMORIES:\n${memorySnippets || "(none)"}` },
    { role: "user", content: userMessage },
  ];

  const model = process.env.OPENAI_MODEL_CHAT || persona.language.startsWith("en")
    ? "gpt-5.1"
    : "gpt-4o-mini";

  const completion = await openai.chat.completions.create({
    model,
    temperature: 0.4,
    max_tokens: maxTokens,
    messages,
  });

  const reply = (completion.choices[0]?.message?.content || "").trim();
  return { reply, modelUsed: model, usedFallback: false };
}
