import OpenAI from "openai";
import { v4 as uuidv4 } from "uuid";

export type SituationType =
  | "good_news"
  | "bad_news"
  | "decision"
  | "missing_them"
  | "small_talk"
  | "stress"
  | "failure"
  | "success"
  | "conflict"
  | "other";

export interface EventMemory {
  id: string;
  userId: string;
  personaId: string;
  situation: SituationType;
  situationDescription: string;
  reactionDescription: string;
  phrasesUsed: string[];
  emotion: "neutral" | "happy" | "sad" | "angry" | "proud" | "worried" | "other";
  entities: string[];
  embedding: number[];
  createdAt: Date;
}

// TODO: replace with real persistence
const eventStore: EventMemory[] = [];

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const EVENT_EXTRACTOR_PROMPT = `
You are extracting events from text about a person. For the given text, identify discrete events where something happened (e.g. exam failure, fixing a fan, playing a game) and how the person reacted.
For each event, return JSON with: situationType, situationDescription, reactionDescription, phrasesUsed, emotion, entities.
If there is no clear event, return an empty list.
`.trim();

function toSituationType(raw: string): SituationType {
  const val = raw?.toLowerCase().trim();
  switch (val) {
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

async function embed(text: string): Promise<number[]> {
  const resp = await openai.embeddings.create({
    model: process.env.OPENAI_MODEL_EMBED || "text-embedding-3-small",
    input: text,
  });
  return resp.data[0]?.embedding || [];
}

function splitSegments(text: string): string[] {
  return text
    .split(/\n{2,}/)
    .flatMap((block) => block.split(/(?<=[.!?])\s+(?=[A-Z])/))
    .map((s) => s.trim())
    .filter(Boolean);
}

export async function extractEventMemoriesFromText(
  text: string,
  userId: string,
  personaId: string
): Promise<EventMemory[]> {
  const segments = splitSegments(text);
  const out: EventMemory[] = [];
  for (const seg of segments) {
    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_MODEL_EVENTS || "gpt-4o-mini",
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: EVENT_EXTRACTOR_PROMPT },
        { role: "user", content: seg },
      ],
    });
    const content = completion.choices[0]?.message?.content || "{}";
    let parsed: any = {};
    try {
      parsed = JSON.parse(content);
    } catch {
      parsed = {};
    }
    const events: any[] = Array.isArray(parsed) ? parsed : parsed.events || [];
    for (const ev of events) {
      if (!ev?.situationDescription || !ev?.reactionDescription) continue;
      const embedding = await embed(
        `${ev.situationDescription}\n${ev.reactionDescription}`
      );
      out.push({
        id: uuidv4(),
        userId,
        personaId,
        situation: toSituationType(ev.situationType),
        situationDescription: ev.situationDescription,
        reactionDescription: ev.reactionDescription,
        phrasesUsed: ev.phrasesUsed || [],
        emotion: ev.emotion || "neutral",
        entities: ev.entities || [],
        embedding,
        createdAt: new Date(),
      });
    }
  }
  // persist
  eventStore.push(...out);
  return out;
}

export async function retrieveRelevantEvents(params: {
  personaId: string;
  userId: string;
  situation: SituationType;
  userMessage: string;
  maxEvents?: number;
}): Promise<EventMemory[]> {
  const { personaId, userId, situation, userMessage, maxEvents = 5 } = params;
  const candidates = eventStore.filter(
    (e) =>
      e.personaId === personaId &&
      e.userId === userId &&
      (e.situation === situation || situation === "other")
  );
  if (!candidates.length) return [];
  const embedResp = await openai.embeddings.create({
    model: process.env.OPENAI_MODEL_EMBED || "text-embedding-3-small",
    input: userMessage,
  });
  const q = embedResp.data[0]?.embedding || [];
  const dot = (a: number[], b: number[]) =>
    a.reduce((acc, v, i) => acc + v * (b[i] || 0), 0);
  const norm = (v: number[]) =>
    Math.sqrt(Math.max(1e-9, v.reduce((acc, x) => acc + x * x, 0)));
  const scored = candidates
    .map((e) => ({
      e,
      score: dot(q, e.embedding) / Math.max(1e-9, norm(q) * norm(e.embedding)),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, maxEvents)
    .map((s) => s.e);
  return scored;
}
