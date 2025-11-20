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

type ChatMessage = { role: string; content: string };

function normalizeMessages(body: any): {
  userId: string | undefined;
  personaId: string | undefined;
  messages: ChatMessage[];
} {
  const b = body || {};

  const userId = b.userId ?? b.user_id ?? b.uid ?? undefined;
  const personaId = b.personaId ?? b.persona_id ?? b.pid ?? undefined;

  const messages: ChatMessage[] = [];

  if (Array.isArray(b.messages)) {
    for (const m of b.messages) {
      if (!m) continue;
      const role = typeof m.role === "string" && m.role.trim().length > 0 ? m.role : "user";
      const contentCandidate =
        typeof m.content === "string"
          ? m.content
          : typeof m.text === "string"
          ? m.text
          : typeof m.message === "string"
          ? m.message
          : "";
      const content = typeof contentCandidate === "string" ? contentCandidate.trim() : "";
      if (content.length > 0) {
        messages.push({ role, content });
      }
    }
  }

  const singleCandidates = [b.message, b.text, b.content, b.input];
  for (const c of singleCandidates) {
    if (typeof c === "string" && c.trim().length > 0) {
      messages.push({ role: "user", content: c.trim() });
      break;
    }
  }

  return { userId, personaId, messages };
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST"]);
    return res.status(405).json({ ok: false, error: "Method not allowed" });
  }

  try {
    const { userId, personaId, messages } = normalizeMessages(req.body);

    console.log(
      "CHAT_DEBUG_BODY",
      JSON.stringify({ userId, personaId, messages }, null, 2)
    );

    const hasMessages = Array.isArray(messages) && messages.length > 0;
    const lastUser = hasMessages ? [...messages].reverse().find((m) => m.role === "user") : undefined;
    const lastContent = typeof lastUser?.content === "string" ? lastUser.content.trim() : "";

  if (!hasMessages || lastContent.length === 0) {
    return res.status(400).json({
      ok: false,
      error: "Empty message",
      debug: {
        userId,
        personaId,
        rawBody: req.body,
        messages,
        hasMessages,
        lastUser,
        lastContent,
      },
    });
  }

    const resolvedUserId = String(userId || "anon");
    const resolvedPersonaId = String(personaId || "default");
    const userMessage = lastContent;

    const basePersona = await loadPersona(resolvedUserId, resolvedPersonaId);
    const persona = await adaptPersonaWithFeedback(basePersona);
    const fingerprint = await loadFingerprint(persona);
    const speakingRules = derivePersonaSpeakingRules(persona, fingerprint);
    const targetLanguage: LanguageMode = chooseLanguage(speakingRules, userMessage);

    const situation = await getSituationForMessage(userMessage);
    const [events, memories] = await Promise.all([
      retrieveRelevantEvents({
        personaId: resolvedPersonaId,
        userId: resolvedUserId,
        situation: situation.situation,
        userMessage,
        maxEvents: 3,
      }),
      retrieveMemoriesForChat({
        personaId: resolvedPersonaId,
        userId: resolvedUserId,
        userMessage,
        maxMemories: 5,
      }),
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

    const passing: string[] = [];
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
        passing.push(candidate);
      }
    }

    let finalReply = "";
    let fallbackUsed = false;

    if (passing.length) {
      const ranked = await rerankCandidates({
        candidates: passing,
        persona,
        userMessage,
        rules: speakingRules,
        fingerprint,
        languageMode: targetLanguage,
      });
      finalReply = ranked[0]?.text || passing[0];
    } else {
      fallbackUsed = true;
      const marker = speakingRules.requiredMarkers?.[0] || "";
      finalReply = `${marker ? marker + " " : ""}Main yahin hoon.`.trim();
    }

    const replyId = randomUUID();
    try {
      await appendEvent({
        userId: resolvedUserId,
        personaId: resolvedPersonaId,
        situation: situation.situation,
        userMessage,
        personaReply: finalReply,
      });
    } catch (logErr) {
      console.error("appendEvent failed", logErr);
    }

    return res.status(200).json({
      ok: true,
      replyId,
      content: finalReply,
      fallback: fallbackUsed,
      personaDebug: {
        energy: persona.tone.energy,
        speakingRules,
      },
    });
  } catch (err: any) {
    console.error("chat handler error", err);
    return res.status(500).json({ ok: false, error: "Internal server error" });
  }
}
