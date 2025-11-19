import type { NextApiRequest, NextApiResponse } from "next";
import { randomUUID } from "crypto";
import { PersonaProfile } from "../../lib/personaBuilder";
import { retrieveRelevantMemories } from "../../lib/retrieveMemories";
import { classifySituationType } from "../../lib/situationRouter";
import { retrieveRelevantEvents } from "../../lib/eventStore";
import { generateContentPlan, rewriteInPersonaStyle } from "../../lib/stagedGeneration";
import { rerankCandidates } from "../../lib/reranker";
import { adaptPersonaWithFeedback } from "../../lib/adaptation";
import { buildPersonaFingerprint, deriveFallbackFingerprint, PersonaFingerprint } from "../../lib/personaFingerprint";
import { derivePersonaSpeakingRules } from "../../lib/personaRules";
import { chooseLanguage, LanguageMode } from "../../lib/languageRouter";
import { checkReplyQuality } from "../../lib/replyCritic";

// Placeholder persona loader; replace with DB lookup
async function loadPersona(userId: string, personaId: string): Promise<PersonaProfile> {
  return {
    id: personaId,
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

async function loadFingerprint(persona: PersonaProfile): Promise<PersonaFingerprint> {
  try {
    return await buildPersonaFingerprint(persona, persona.examples || []);
  } catch (err) {
    console.warn("fingerprint fallback", err);
    return deriveFallbackFingerprint(persona);
  }
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

    const basePersona = await loadPersona(userId, personaId);
    const persona = await adaptPersonaWithFeedback(basePersona);
    const fingerprint = await loadFingerprint(persona);
    const speakingRules = derivePersonaSpeakingRules(persona, fingerprint);
    const targetLanguage: LanguageMode = chooseLanguage(speakingRules, userMessage);

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
    async function produceCandidate(baseTemp: number): Promise<string> {
      let attempt = 0;
      let latest = "";
      while (attempt < 3) {
        latest = await rewriteInPersonaStyle({
          persona,
          draft: contentPlan.draft,
          userMessage,
          fingerprint,
          rules: speakingRules,
          targetLanguage,
          temperature: baseTemp - attempt * 0.05,
          attempt,
        });
        const verdict = await checkReplyQuality({
          persona,
          userMessage,
          candidateReply: latest,
          rules: speakingRules,
          fingerprint,
          languageMode: targetLanguage,
          strict: true,
        });
        if (verdict === "PASS") break;
        attempt += 1;
      }
      return latest;
    }

    const candidateReplies = await Promise.all([
      produceCandidate(0.3),
      produceCandidate(0.35),
      produceCandidate(0.4),
    ]);

    // Re-rank
    const ranked = await rerankCandidates({
      candidates: candidateReplies,
      persona,
      userMessage,
      rules: speakingRules,
      fingerprint,
      languageMode: targetLanguage,
    });

    const best = ranked[0];
    const replyId = randomUUID();
    return res.status(200).json({
      replyId,
      reply: best?.text || candidateReplies[0],
      model_used: process.env.OPENAI_MODEL_CHAT || "gpt-5.1",
      fallback: !best?.criticPass,
      debug: {
        draft: contentPlan.draft,
        candidates: ranked,
        situation,
        speakingRules,
        fingerprint,
      },
    });
  } catch (err: any) {
    return res.status(500).json({ error: err?.message || "server_error" });
  }
}
