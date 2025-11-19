import { PersonaProfile } from "./personaBuilder";
import { computeStyleSimilarity } from "./styleSimilarity";
import { containsBannedPhrase, countQuestions, hasExcessiveLength } from "./textGuards";
import { checkReplyQuality } from "./replyCritic";
import { SpeakingRules } from "./personaRules";
import { PersonaFingerprint } from "./personaFingerprint";
import { LanguageMode } from "./languageRouter";

export interface RankedCandidate {
  text: string;
  styleScore: number;
  lengthPenalty: number;
  finalScore: number;
  criticPass: boolean;
  banned: boolean;
}

function getMaxTokensForPersona(persona: PersonaProfile): number {
  switch (persona.tone.typicalLength) {
    case "very_short":
      return 40;
    case "short":
      return 60;
    case "medium":
      return 120;
    case "long":
    default:
      return 200;
  }
}

export async function rerankCandidates(params: {
  candidates: string[];
  persona: PersonaProfile;
  userMessage?: string;
  rules: SpeakingRules;
  fingerprint: PersonaFingerprint;
  languageMode: LanguageMode;
}): Promise<RankedCandidate[]> {
  const { candidates, persona, userMessage = "", rules, fingerprint, languageMode } = params;
  const baseMax = getMaxTokensForPersona(persona);
  const multiplier = (persona as any).overrides?.maxTokensMultiplier ?? 1;
  const maxTokens = Math.max(16, Math.floor(baseMax * multiplier));
  const strict = Boolean((persona as any).overrides?.stricterCritic);
  const allowQuestions = !rules.forbidQuestions && (userMessage.includes("?") || /advice|help|what should|how do/i.test(userMessage));

  const results: RankedCandidate[] = [];

  for (const text of candidates) {
    const banned = containsBannedPhrase(text);
    const questions = countQuestions(text);
    const questionFail = (rules.forbidQuestions && questions > 0) || (questions > 1 && !allowQuestions);
    const criticPass =
      !questionFail &&
      (await checkReplyQuality({
        persona,
        userMessage,
        candidateReply: text,
        rules,
        fingerprint,
        languageMode,
        strict,
      })) === "PASS";
    const styleScore = await computeStyleSimilarity(text, persona);
    const lengthPenalty = hasExcessiveLength(text, strict ? Math.floor(maxTokens * 0.9) : maxTokens) ? 1 : 0;
    let finalScore = styleScore - lengthPenalty;
    if (banned || questionFail || !criticPass) finalScore = -1e6;
    const cand: RankedCandidate = {
      text,
      styleScore,
      lengthPenalty,
      finalScore,
      criticPass,
      banned,
    };
    results.push(cand);
  }

  return results.sort((a, b) => b.finalScore - a.finalScore);
}
