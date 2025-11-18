import OpenAI from "openai";
import { SituationType } from "./eventStore";

export interface SituationClassification {
  situation: SituationType;
  confidence: number;
}

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const ROUTER_PROMPT = `
You are classifying a user message into high-level situation types. Pick one label: good_news, bad_news, decision, missing_them, small_talk, stress, failure, success, conflict, other. Also give a confidence 0–1.

Reply as JSON: { "situation": "<label>", "confidence": <0-1> }
`.trim();

function toSituation(val: string): SituationType {
  switch ((val || "").toLowerCase()) {
    case "good_news":
    case "bad_news":
    case "decision":
    case "missing_them":
    case "small_talk":
    case "stress":
    case "failure":
    case "success":
    case "conflict":
      return val as SituationType;
    default:
      return "other";
  }
}

export async function classifySituationType(
  userMessage: string
): Promise<SituationClassification> {
  const completion = await openai.chat.completions.create({
    model: process.env.OPENAI_MODEL_CLASSIFIER || "gpt-4o-mini",
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: ROUTER_PROMPT },
      { role: "user", content: userMessage },
    ],
  });
  const raw = completion.choices[0]?.message?.content || "{}";
  try {
    const parsed = JSON.parse(raw);
    return {
      situation: toSituation(parsed.situation || "other"),
      confidence: typeof parsed.confidence === "number" ? parsed.confidence : 0.5,
    };
  } catch {
    return { situation: "other", confidence: 0.5 };
  }
}
