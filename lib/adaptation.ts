import { PersonaProfile } from "./personaBuilder";
import { FeedbackTag, getFeedbackForPersona } from "./feedbackStore";

export interface AdaptedPersona extends PersonaProfile {
  overrides?: {
    maxTokensMultiplier?: number;
    stricterCritic?: boolean;
    lowerEnergy?: boolean;
    feedbackSummary?: string;
    loraModelName?: string; // TODO: per-persona LoRA hook
  };
}

/**
 * Adjust persona behavior based on historical feedback tags.
 * This is intentionally simple and rule-based; swap with a DB/analytics-driven
 * adaptation later.
 */
export async function adaptPersonaWithFeedback(
  persona: PersonaProfile
): Promise<AdaptedPersona> {
  const personaId = (persona as any).id || "default";
  const feedback = await getFeedbackForPersona(personaId);

  if (!feedback.length) {
    return { ...persona, overrides: {} };
  }

  const total = feedback.length;
  const tagCounts: Record<FeedbackTag, number> = {
    too_generic: 0,
    too_long: 0,
    too_emotional: 0,
    too_cold: 0,
    wrong_language: 0,
    not_like_them: 0,
    good_match: 0,
    other: 0,
  };

  for (const fb of feedback) {
    for (const tag of fb.tags) {
      tagCounts[tag] = (tagCounts[tag] || 0) + 1;
    }
  }

  const adapted: AdaptedPersona = { ...persona, overrides: {} };
  const summaryParts: string[] = [];

  // If users complain about length, shorten replies.
  if (tagCounts.too_long / total > 0.2) {
    adapted.overrides!.maxTokensMultiplier = 0.7;
    summaryParts.push("Users say replies are too long.");
  }

  // If replies feel generic or unlike the persona, make critic stricter.
  if (
    tagCounts.too_generic / total > 0.2 ||
    tagCounts.not_like_them / total > 0.2
  ) {
    adapted.overrides!.stricterCritic = true;
    summaryParts.push("Users say replies feel generic / not like the persona.");
  }

  // If replies are too emotional, push tone toward low energy.
  if (tagCounts.too_emotional / total > 0.2) {
    adapted.overrides!.lowerEnergy = true;
    summaryParts.push("Users say replies are too emotional.");
  }

  if (summaryParts.length) {
    adapted.overrides!.feedbackSummary = summaryParts.join(" ");
  }

  // Apply energy override so downstream guards see the adjusted tone immediately.
  if (adapted.overrides?.lowerEnergy) {
    adapted.tone = { ...adapted.tone, energy: "low" };
  }

  return adapted;
}
