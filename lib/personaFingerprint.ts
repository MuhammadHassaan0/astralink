import OpenAI from "openai";
import { PersonaProfile } from "./personaBuilder";
import { PERSONA_FINGERPRINT_PROMPT } from "./prompts";

export interface PersonaFingerprint {
  sentenceTemplates: string[];
  fillerWords: string[];
  commonPhrases: string[];
  languagePreference: "urdu" | "english" | "mix";
  typicalLength: "short" | "medium" | "long";
  energy: "low" | "medium" | "high";
}

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const FALLBACK_FINGERPRINT: PersonaFingerprint = {
  sentenceTemplates: ["<greeting>, keep it brief."],
  fillerWords: [],
  commonPhrases: [],
  languagePreference: "english",
  typicalLength: "short",
  energy: "low",
};

export async function buildPersonaFingerprint(
  persona: PersonaProfile,
  texts: string[]
): Promise<PersonaFingerprint> {
  const corpus = texts.filter(Boolean).join("\n\n").slice(0, 12000);
  if (!corpus) {
    return deriveFallbackFingerprint(persona);
  }

  const prompt = PERSONA_FINGERPRINT_PROMPT.replace(
    "{{PERSONA_NAME}}",
    persona.name || "Unknown"
  ).replace("{{TEXT_BLOCK}}", corpus);

  const completion = await openai.chat.completions.create({
    model: process.env.OPENAI_MODEL_FINGERPRINT || "gpt-4o-mini",
    response_format: { type: "json_object" },
    temperature: 0,
    max_tokens: 250,
    messages: [{ role: "system", content: prompt }],
  });

  const raw = completion.choices[0]?.message?.content || "{}";
  try {
    const parsed = JSON.parse(raw) as PersonaFingerprint;
    return {
      sentenceTemplates: parsed.sentenceTemplates || [],
      fillerWords: parsed.fillerWords || [],
      commonPhrases: parsed.commonPhrases || [],
      languagePreference: parsed.languagePreference || mapLanguage(persona.language),
      typicalLength: parsed.typicalLength || inferLength(persona),
      energy: parsed.energy || persona.tone.energy || "medium",
    };
  } catch (err) {
    console.error("Fingerprint parse failed", err, raw);
    return deriveFallbackFingerprint(persona);
  }
}

export function deriveFallbackFingerprint(persona: PersonaProfile): PersonaFingerprint {
  return {
    sentenceTemplates:
      (persona.examples && persona.examples.length && persona.examples) || persona.catchphrases || FALLBACK_FINGERPRINT.sentenceTemplates,
    fillerWords: [],
    commonPhrases: persona.catchphrases || [],
    languagePreference: mapLanguage(persona.language),
    typicalLength: inferLength(persona),
    energy: persona.tone.energy || "medium",
  };
}

function mapLanguage(code: string | undefined): "urdu" | "english" | "mix" {
  if (!code) return "english";
  if (code.startsWith("ur")) return "urdu";
  if (code === "mix" || code === "multi") return "mix";
  return "english";
}

function inferLength(persona: PersonaProfile): "short" | "medium" | "long" {
  switch (persona.tone.typicalLength) {
    case "very_short":
    case "short":
      return "short";
    case "medium":
      return "medium";
    case "long":
    default:
      return "long";
  }
}
