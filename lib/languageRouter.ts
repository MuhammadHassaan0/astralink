import { SpeakingRules } from "./personaRules";

export type LanguageMode = "urdu" | "english" | "mix";

const ENGLISH_WORD_REGEX = /[a-zA-Z]+/g;
const URDU_LETTERS = /[\u0600-\u06FF]/g;
const LANGUAGE_REQUEST = /(respond|reply|speak) in (english|urdu)/i;

export function detectLanguage(message: string): LanguageMode {
  const latin = (message.match(ENGLISH_WORD_REGEX) || []).length;
  const urdu = (message.match(URDU_LETTERS) || []).length;
  if (urdu && urdu > latin) return "urdu";
  if (latin && latin > urdu) return "english";
  return "mix";
}

export function chooseLanguage(
  rules: SpeakingRules,
  userMessage: string
): LanguageMode {
  const request = userMessage.match(LANGUAGE_REQUEST);
  if (request) {
    return request[2].toLowerCase() === "urdu" ? "urdu" : "english";
  }
  const userLang = detectLanguage(userMessage);
  if (rules.defaultLanguage === "mix") {
    return userLang === "mix" ? "english" : userLang;
  }
  if (rules.defaultLanguage === "urdu" && userLang === "english") {
    return "urdu";
  }
  if (rules.defaultLanguage === "english" && userLang === "urdu") {
    return "english";
  }
  return rules.defaultLanguage || userLang;
}

export function violatesLanguage(reply: string, mode: LanguageMode): boolean {
  const latin = (reply.match(ENGLISH_WORD_REGEX) || []).length;
  const urdu = (reply.match(URDU_LETTERS) || []).length;
  const total = Math.max(1, latin + urdu);
  if (mode === "urdu") {
    return latin / total > 0.2;
  }
  if (mode === "english") {
    return urdu / total > 0.2;
  }
  return false;
}
