import { PersonaProfile } from "./personaBuilder";
import { computeStyleSimilarity } from "./styleSimilarity";
import { containsBannedPhrase, countQuestions, hasExcessiveLength } from "./textGuards";
import { checkReplyQuality } from "./replyCritic";

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
}): Promise<RankedCandidate[]> {
  const { candidates, persona, userMessage = "" } = params;
  const maxTokens = getMaxTokensForPersona(persona);
  const allowQuestions =
    userMessage.includes("?") || /advice|help|what should|how do/i.test(userMessage);

  const results: RankedCandidate[] = [];

  for (const text of candidates) {
    const banned = containsBannedPhrase(text);
    const questions = countQuestions(text);
    const criticPass =
      (await checkReplyQuality({
        persona,
        userMessage,
        candidateReply: text,
      })) === "PASS";
    const styleScore = await computeStyleSimilarity(text, persona);
    const lengthPenalty =
      questions > 1 && !allowQuestions
        ? 2
        : hasExcessiveLength(text, maxTokens)
        ? 1
        : 0;
    let finalScore = styleScore - lengthPenalty;
    if (banned || !criticPass) finalScore = -1e6;
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
