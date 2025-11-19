import type { NextApiRequest, NextApiResponse } from "next";
import { randomUUID } from "crypto";
import { loadPersona } from "../../lib/personaBuilder";
import type { PersonaProfile } from "../../lib/personaBuilder";
import { retrieveMemoriesForChat } from "../../lib/retrieveMemories";
import { getSituationForMessage } from "../../lib/situationRouter";
import { retrieveRelevantEvents, appendEvent } from "../../lib/eventStore";
import { stagedGenerate } from "../../lib/stagedGeneration";
import { rerankCandidates } from "../../lib/reranker";
import { adaptPersonaWithFeedback } from "../../lib/adaptation";
import {
  buildPersonaFingerprint,
  deriveFallbackFingerprint,
  PersonaFingerprint,
} from "../../lib/personaFingerprint";
import { derivePersonaSpeakingRules } from "../../lib/personaRules";
import { chooseLanguage, LanguageMode } from "../../lib/languageRouter";
import { runReplyCritic } from "../../lib/replyCritic";

async function loadFingerprint(persona: PersonaProfile): Promise<PersonaFingerprint> {
  try {
    return await buildPersonaFingerprint(persona, persona.examples || []);
  } catch (err) {
    console.warn("fingerprint fallback", err);
    return deriveFallbackFingerprint(persona);
  }
}

type ChatMessage = { role: "user" | "assistant" | string; content: string };

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).end();
  }

  try {
    const body = req.body || {};
    const userId = String(body.userId || "anon");
    const personaId = String(body.personaId || "default");
    const messages: ChatMessage[] = Array.isArray(body.messages) ? body.messages : [];
    const fallbackMessage = typeof body.message === "string" ? body.message : "";
    const latestContent = messages.length ? messages[messages.length - 1]?.content : fallbackMessage;
    const userMessage = String(latestContent || "").trim();
    if (!userMessage) {
      return res.status(400).json({ error: "message required" });
    }

    const basePersona = await loadPersona(userId, personaId);
    const persona = await adaptPersonaWithFeedback(basePersona);
    const fingerprint = await loadFingerprint(persona);
    const speakingRules = derivePersonaSpeakingRules(persona, fingerprint);
    const targetLanguage: LanguageMode = chooseLanguage(speakingRules, userMessage);

    const situation = await getSituationForMessage(userMessage);
    const [events, memories] = await Promise.all([
      retrieveRelevantEvents({
        personaId,
        userId,
        situation: situation.situation,
        userMessage,
        maxEvents: 3,
      }),
      retrieveMemoriesForChat({ personaId, userId, userMessage, maxMemories: 5 }),
    ]);

    const generation = await stagedGenerate({
      persona,
      fingerprint,
      rules: speakingRules,
      targetLanguage,
      userMessage,
      events,
      memories,
    });

    const survivingCandidates: string[] = [];
    for (const candidate of generation.candidates) {
      const verdict = await runReplyCritic({
        persona,
        userMessage,
        candidateReply: candidate,
        rules: speakingRules,
        fingerprint,
        languageMode: targetLanguage,
        strict: true,
      });
      if (verdict === "PASS") {
        survivingCandidates.push(candidate);
      }
    }

    let finalReply = "";
    let fallbackUsed = false;
    if (survivingCandidates.length) {
      const ranked = await rerankCandidates({
        candidates: survivingCandidates,
        persona,
        userMessage,
        rules: speakingRules,
        fingerprint,
        languageMode: targetLanguage,
      });
      finalReply = ranked[0]?.text || survivingCandidates[0];
    } else {
      fallbackUsed = true;
      const marker = speakingRules.requiredMarkers?.[0] || "";
      finalReply = `${marker ? marker + " " : ""}I'm listening.`.trim();
    }

    const replyId = randomUUID();
    await appendEvent({
      userId,
      personaId,
      situation: situation.situation,
      userMessage,
      personaReply: finalReply,
    });

    return res.status(200).json({
      replyId,
      content: finalReply,
      fallback: fallbackUsed,
      personaDebug: {
        energy: persona.tone.energy,
        speakingRules,
      },
    });
  } catch (err: any) {
    return res.status(500).json({ error: err?.message || "server_error" });
  }
}
