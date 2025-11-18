import OpenAI from "openai";
import { PersonaProfile } from "./personaBuilder";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

type CacheEntry = { centroid: number[]; examples: string[] };
const styleCache: Map<string, CacheEntry> = new Map();

function cosine(a: number[], b: number[]): number {
  const dot = a.reduce((acc, v, i) => acc + v * (b[i] || 0), 0);
  const na = Math.sqrt(a.reduce((acc, v) => acc + v * v, 0));
  const nb = Math.sqrt(b.reduce((acc, v) => acc + v * v, 0));
  if (!na || !nb) return 0;
  return dot / (na * nb);
}

async function embed(texts: string[]): Promise<number[][]> {
  const resp = await openai.embeddings.create({
    model: process.env.OPENAI_MODEL_STYLE_EMBED || "text-embedding-3-small",
    input: texts,
  });
  return resp.data.map((d) => d.embedding || []);
}

function mean(vectors: number[][]): number[] {
  if (!vectors.length) return [];
  const len = vectors[0].length;
  const sum = new Array(len).fill(0);
  vectors.forEach((vec) => {
    vec.forEach((v, i) => (sum[i] += v));
  });
  return sum.map((v) => v / vectors.length);
}

export async function computeStyleSimilarity(
  candidate: string,
  persona: PersonaProfile
): Promise<number> {
  const personaKey = (persona as any).id || persona.name || "default";
  let entry = styleCache.get(personaKey);

  if (!entry) {
    const examples =
      (persona as any).examples ||
      persona.catchphrases ||
      [`${persona.name || "They"} says short, practical things.`];
    const exampleEmbeds = await embed(examples);
    entry = { centroid: mean(exampleEmbeds), examples };
    styleCache.set(personaKey, entry);
  }

  const [candEmbed] = await embed([candidate]);
  if (!entry.centroid.length || !candEmbed) return 0;
  return cosine(entry.centroid, candEmbed);
}
