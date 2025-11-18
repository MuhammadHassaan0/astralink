import type { NextApiRequest, NextApiResponse } from "next";
import { PersonaProfile } from "../../lib/personaBuilder";
import { retrieveRelevantMemories } from "../../lib/retrieveMemories";
import { classifySituationType } from "../../lib/situationRouter";
import { retrieveRelevantEvents } from "../../lib/eventStore";
import { generateContentPlan, rewriteInPersonaStyle } from "../../lib/stagedGeneration";
import { rerankCandidates } from "../../lib/reranker";

// Placeholder persona loader; replace with DB lookup
async function loadPersona(userId: string): Promise<PersonaProfile> {
  return {
    name: "",
    relationshipToUser: "parent",
    language: "en",
    defaultFormality: "informal",
    tone: { energy: "medium", warmth: "medium", humor: "none", typicalLength: "short" },
    catchphrases: [],
    topics: { loves: [], avoids: [] },
    responseRules: [],
  };
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).end();
  }

  try {
    const userId = String(req.body.userId || "anon");
    const personaId = String(req.body.personaId || "default");
    const userMessage = String(req.body.message || "").trim();
    if (!userMessage) return res.status(400).json({ error: "message required" });

    const persona = await loadPersona(userId);

    // classify situation and pull events/memories
    const situation = await classifySituationType(userMessage);
    const events = await retrieveRelevantEvents({
      personaId,
      userId,
      situation: situation.situation,
      userMessage,
      maxEvents: 3,
    });
    const memories = await retrieveRelevantMemories({ personaId, userId, userMessage, maxMemories: 5 });

    // Stage 1: content plan
    const contentPlan = await generateContentPlan({
      persona,
      userMessage,
      events,
      memories,
    });

    // Stage 2: multiple style rewrites
    const candidatePromises = [
      rewriteInPersonaStyle({ persona, draft: contentPlan.draft, userMessage, temperature: 0.3 }),
      rewriteInPersonaStyle({ persona, draft: contentPlan.draft, userMessage, temperature: 0.35 }),
      rewriteInPersonaStyle({ persona, draft: contentPlan.draft, userMessage, temperature: 0.4 }),
    ];
    const candidateReplies = await Promise.all(candidatePromises);

    // Re-rank
    const ranked = await rerankCandidates({
      candidates: candidateReplies,
      persona,
      userMessage,
    });

    const best = ranked[0];
    return res.status(200).json({
      reply: best?.text || candidateReplies[0],
      model_used: process.env.OPENAI_MODEL_CHAT || "gpt-5.1",
      fallback: !best?.criticPass,
      debug: {
        draft: contentPlan.draft,
        candidates: ranked,
        situation,
      },
    });
  } catch (err: any) {
    return res.status(500).json({ error: err?.message || "server_error" });
  }
}
