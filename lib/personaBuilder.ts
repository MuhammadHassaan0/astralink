import OpenAI from "openai";
import { PERSONA_BUILDER_PROMPT } from "./prompts";

export interface PersonaProfile {
  id?: string;
  name: string | null;
  relationshipToUser: string | null;
  language: string;
  defaultFormality: "formal" | "informal";
  tone: {
    energy: "low" | "medium" | "high";
    warmth: "low" | "medium" | "high";
    humor: "none" | "dry" | "light" | "playful";
    typicalLength: "very_short" | "short" | "medium" | "long";
  };
  catchphrases: string[];
  topics: {
    loves: string[];
    avoids: string[];
  };
  responseRules: string[];
  examples?: string[];
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * Build a structured persona profile from raw texts (interview answers, uploads, etc).
 * Uses a persona-compiler LLM prompt and enforces strict JSON output.
 */
export async function buildPersonaProfile(
  texts: string[],
  defaultLanguage: string
): Promise<PersonaProfile> {
  const corpus = texts.filter(Boolean).join("\n\n").slice(0, 12000);

  const prompt = PERSONA_BUILDER_PROMPT.replace(
    "{{DEFAULT_LANGUAGE}}",
    defaultLanguage
  ).replace("{{TEXT_BLOCK}}", corpus || "No data provided.");

  const completion = await openai.chat.completions.create({
    model: process.env.OPENAI_MODEL_PERSONA || "gpt-4o-mini",
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: prompt },
      {
        role: "user",
        content: `defaultLanguage: ${defaultLanguage}\n\nNOTES:\n${corpus}`,
      },
    ],
  });

  const raw = completion.choices[0]?.message?.content || "{}";
  try {
    const parsed = JSON.parse(raw) as PersonaProfile;
    return parsed;
  } catch (err) {
    throw new Error(`Persona JSON parse failed: ${err}; raw=${raw}`);
  }
}
