import OpenAI from "openai";
import { v4 as uuidv4 } from "uuid";
import { MEMORY_TAGGER_PROMPT } from "./prompts";

export interface Memory {
  id: string;
  userId: string;
  personaId: string;
  text: string;
  topics: string[];
  emotion: string;
  embedding: number[];
}

// Placeholder persistence; replace with your DB layer
const memoryDB: Memory[] = [];

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

function chunkText(text: string): string[] {
  const parts = text
    .split(/(?<=[.!?])\s+/)
    .map((p) => p.trim())
    .filter(Boolean);
  const chunks: string[] = [];
  let bucket: string[] = [];
  for (const part of parts) {
    bucket.push(part);
    const joined = bucket.join(" ");
    if (joined.length >= 180 || bucket.length >= 3) {
      chunks.push(joined);
      bucket = [];
    }
  }
  if (bucket.length) chunks.push(bucket.join(" "));
  return chunks;
}

async function classifyTopicsAndEmotion(text: string): Promise<{
  topics: string[];
  emotion: string;
}> {
  const memoryTaggerPrompt = `
${MEMORY_TAGGER_PROMPT.replace("{{MEMORY_TEXT}}", text)}
`.trim();
  const completion = await openai.chat.completions.create({
    model: process.env.OPENAI_MODEL_CLASSIFIER || "gpt-4o-mini",
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [{ role: "system", content: memoryTaggerPrompt }],
  });
  const content = completion.choices[0]?.message?.content || "{}";
  try {
    const parsed = JSON.parse(content);
    return { topics: parsed.topics || [], emotion: parsed.emotion || "neutral" };
  } catch {
    return { topics: [], emotion: "neutral" };
  }
}

async function embed(text: string): Promise<number[]> {
  const resp = await openai.embeddings.create({
    model: process.env.OPENAI_MODEL_EMBED || "text-embedding-3-small",
    input: text,
  });
  return resp.data[0]?.embedding || [];
}

/**
 * Ingest raw memory texts: chunk, tag topics/emotion, embed, and persist.
 */
export async function ingestMemories(
  texts: string[],
  userId: string,
  personaId: string
): Promise<Memory[]> {
  const output: Memory[] = [];
  for (const raw of texts.filter(Boolean)) {
    const chunks = chunkText(raw);
    for (const chunk of chunks) {
      const { topics, emotion } = await classifyTopicsAndEmotion(chunk);
      const embedding = await embed(chunk);
      const memory: Memory = {
        id: uuidv4(),
        userId,
        personaId,
        text: chunk,
        topics,
        emotion,
        embedding,
      };
      memoryDB.push(memory);
      output.push(memory);
    }
  }
  return output;
}

// Simple vector search placeholder
export function vectorSearch(
  personaId: string,
  userId: string,
  queryEmbedding: number[],
  max = 5
): Memory[] {
  const dot = (a: number[], b: number[]) =>
    a.reduce((acc, v, i) => acc + v * (b[i] || 0), 0);
  const norms: Map<string, number> = new Map();
  const norm = (vec: number[]) => {
    const key = String(vec.length) + vec[0];
    if (norms.has(key)) return norms.get(key)!;
    const n = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0));
    norms.set(key, n);
    return n;
  };
  const candidates = memoryDB.filter(
    (m) => m.personaId === personaId && m.userId === userId
  );
  const scored = candidates
    .map((m) => {
      const score =
        dot(queryEmbedding, m.embedding) /
        Math.max(1e-9, norm(queryEmbedding) * norm(m.embedding));
      return { m, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, max)
    .map((s) => s.m);
  return scored;
}
