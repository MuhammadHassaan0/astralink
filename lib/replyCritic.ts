import OpenAI from "openai";
import { PersonaProfile } from "./personaBuilder";
import { CRITIC_PROMPT } from "./prompts";
import { SpeakingRules } from "./personaRules";
import { PersonaFingerprint } from "./personaFingerprint";
import { violatesLanguage } from "./languageRouter";
import type { LanguageMode } from "./languageRouter";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const AI_MASKS = [
  "as an ai",
  "i am here to help",
  "how are you feeling today",
  "hope you're doing well",
  "hope you are doing well",
  "i am here for you",
  "i'm here for you",
  "in spirit",
  "virtual assistant",
  "ai model",
  "chatgpt",
];

function hasAiTell(reply: string): boolean {
  const lower = reply.toLowerCase();
  if (/[😀-🙏]/u.test(reply)) return true;
  return AI_MASKS.some((mask) => lower.includes(mask));
}

/**
 * Check reply quality against persona constraints.
 */
const EXTRA_BANNED = [
  "i understand",
  "it’s okay to feel that way",
  "you're stronger than you think",
  "i'm always here for you",
  "take it one step at a time",
  "stay strong",
  "what’s on your mind",
  "how are you feeling",
  "do you remember how that felt",
  "this warms my heart",
  "i miss you a lot",
];

function countSentences(reply: string): number {
  return reply.split(/[.!?]+/).filter((s) => s.trim().length > 0).length || 0;
}

function countTokens(reply: string): number {
  return reply.split(/\s+/).filter(Boolean).length;
}

function containsRequiredMarker(reply: string, markers: string[]): boolean {
  if (!markers.length) return true;
  const lower = reply.toLowerCase();
  return markers.some((marker) => lower.includes(marker.toLowerCase()));
}

function containsBanned(reply: string, banned: string[]): boolean {
  const lower = reply.toLowerCase();
  return banned.some((phrase) => lower.includes(phrase.toLowerCase()));
}

const POETIC_REGEX = /(gentle breeze|soft glow|eternal|echoes of|lingering warmth|fading light)/i;

export async function checkReplyQuality({
  persona,
  userMessage,
  candidateReply,
  rules,
  fingerprint,
  languageMode,
  strict,
}: {
  persona: PersonaProfile;
  userMessage: string;
  candidateReply: string;
  rules: SpeakingRules;
  fingerprint: PersonaFingerprint;
  languageMode: LanguageMode;
  strict?: boolean;
}): Promise<"PASS" | "FAIL"> {
  if (hasAiTell(candidateReply)) return "FAIL";

  const sentences = countSentences(candidateReply);
  if (sentences > rules.maxSentences) return "FAIL";

  const tokens = countTokens(candidateReply);
  if (tokens > rules.maxTokens) return "FAIL";

  if (rules.forbidQuestions && candidateReply.includes("?")) return "FAIL";

  if (violatesLanguage(candidateReply, languageMode)) return "FAIL";

  const bannedList = [...rules.bannedPhrases, ...EXTRA_BANNED];
  if (containsBanned(candidateReply, bannedList)) return "FAIL";

  if (!containsRequiredMarker(candidateReply, rules.requiredMarkers || [])) return "FAIL";

  if (rules.energy === "low" && /!/g.test(candidateReply)) return "FAIL";

  if (POETIC_REGEX.test(candidateReply)) return "FAIL";

  if (strict && /stay positive|you'll be fine|you will be fine|focus on your breath/i.test(candidateReply)) {
    return "FAIL";
  }

  const prompt = CRITIC_PROMPT.replace(
    "{{PERSONA_JSON}}",
    JSON.stringify(persona)
  )
    .replace("{{SPEAKING_RULES}}", JSON.stringify(rules))
    .replace("{{FINGERPRINT}}", JSON.stringify(fingerprint))
    .replace("{{USER_MESSAGE}}", userMessage)
    .replace("{{CANDIDATE_REPLY}}", candidateReply);

  const completion = await openai.chat.completions.create({
    model: process.env.OPENAI_MODEL_CRITIC || "gpt-4o-mini",
    temperature: 0,
    max_tokens: 2,
    messages: [{ role: "system", content: prompt }],
  });
  const verdict = (completion.choices[0]?.message?.content || "FAIL").trim().toUpperCase();
  return verdict.includes("PASS") ? "PASS" : "FAIL";
}
