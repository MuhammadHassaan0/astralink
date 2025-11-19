import { PersonaProfile } from "./personaBuilder";
import { PersonaFingerprint } from "./personaFingerprint";

export interface SpeakingRules {
  maxSentences: number;
  maxTokens: number;
  forbidQuestions: boolean;
  defaultLanguage: "urdu" | "english" | "mix";
  energy: "low" | "medium" | "high";
  bannedPhrases: string[];
  requiredMarkers: string[];
}

const BASE_THERAPY_BANNED = [
  "i understand",
  "it's okay to feel that way",
  "you’re stronger than you think",
  "i'm always here for you",
  "take it one step at a time",
  "what’s on your mind",
  "how are you feeling",
  "how have you been holding up",
  "how are you doing today",
  "what’s been tiring you out",
  "this warms my heart",
  "stay strong",
  "i miss you a lot",
  "want to talk about what’s bothering",
  "do you remember how that felt",
];

export function derivePersonaSpeakingRules(
  persona: PersonaProfile,
  fingerprint: PersonaFingerprint
): SpeakingRules {
  const energy = fingerprint.energy || persona.tone.energy || "medium";
  const typical = fingerprint.typicalLength;
  let maxSentences = typical === "long" ? 4 : typical === "medium" ? 3 : 2;
  let maxTokens = typical === "long" ? 160 : typical === "medium" ? 90 : 45;

  if (energy === "low") {
    maxSentences = Math.min(maxSentences, 2);
    maxTokens = Math.min(maxTokens, 45);
  }

  const defaultLanguage =
    persona.speakingRules?.defaultLanguage || fingerprint.languagePreference;

  const forbidQuestions = Boolean(
    persona.responseRules?.some((r) => /no questions|avoid questions|rarely ask/i.test(r)) ||
      energy === "low"
  );

  const neverSay = persona.responseRules
    ?.filter((r) => /never say/i.test(r))
    .map((r) => r.replace(/never say/gi, "")) || [];

  const bannedPhrases = Array.from(
    new Set(
      [
        ...BASE_THERAPY_BANNED,
        ...(persona.speakingRules?.bannedPhrases || []),
        ...neverSay,
      ]
        .map((p) => p.trim().toLowerCase())
        .filter(Boolean)
    )
  );

  const requiredMarkers =
    persona.speakingRules?.requiredMarkers?.length
      ? persona.speakingRules.requiredMarkers
      : fingerprint.commonPhrases.slice(0, 1);

  return {
    maxSentences,
    maxTokens,
    forbidQuestions,
    defaultLanguage,
    energy,
    bannedPhrases,
    requiredMarkers,
  };
}
