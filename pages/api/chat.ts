import type { NextApiRequest, NextApiResponse } from "next";
import { generatePersonaReply } from "../../lib/responseEngine";
import { retrieveRelevantMemories } from "../../lib/retrieveMemories";
import { checkReplyQuality } from "../../lib/replyCritic";
import { PersonaProfile } from "../../lib/personaBuilder";

// Placeholder data loaders — wire to your real DB
async function loadPersona(userId: string): Promise<PersonaProfile> {
  // TODO: replace with real DB lookup
  return {
    name: "",
    relationshipToUser: "parent",
    language: "en",
    defaultFormality: "informal",
    tone: { energy: "medium", warmth: "high", humor: "light", typicalLength: "medium" },
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
    const memories = await retrieveRelevantMemories({ personaId, userId, userMessage, maxMemories: 5 });

    let replyObj = await generatePersonaReply({ persona, memories, userMessage });
    let verdict = await checkReplyQuality({ persona, userMessage, candidateReply: replyObj.reply });

    // Simple anti-GPT heuristic
    const forbidden = [
      "as an ai",
      "i am here to help",
      "how are you feeling today",
      "hope you're doing well",
    ];
    const hasBadPhrase = forbidden.some((p) => replyObj.reply.toLowerCase().includes(p));

    if (verdict === "FAIL" || hasBadPhrase) {
      replyObj = await generatePersonaReply({ persona, memories, userMessage });
      verdict = await checkReplyQuality({ persona, userMessage, candidateReply: replyObj.reply });
    }

    return res.status(200).json({ reply: replyObj.reply, model_used: replyObj.modelUsed, fallback: replyObj.usedFallback, verdict });
  } catch (err: any) {
    return res.status(500).json({ error: err?.message || "server_error" });
  }
}
