import OpenAI from "openai";
import { PersonaProfile } from "./personaBuilder";
import { CRITIC_PROMPT } from "./prompts";

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
export async function checkReplyQuality({
  persona,
  userMessage,
  candidateReply,
  strict,
}: {
  persona: PersonaProfile;
  userMessage: string;
  candidateReply: string;
  strict?: boolean;
}): Promise<"PASS" | "FAIL"> {
  if (hasAiTell(candidateReply)) return "FAIL";
  if (strict && /stay positive|you'll be fine|you will be fine/i.test(candidateReply)) return "FAIL";

  const prompt = CRITIC_PROMPT.replace(
    "{{PERSONA_JSON}}",
    JSON.stringify(persona)
  )
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
