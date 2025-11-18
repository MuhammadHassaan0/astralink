export const BANNED_PHRASES = [
  "I'm here for you",
  "I am here for you",
  "How are you doing today",
  "What’s on your mind",
  "What's on your mind",
  "How’s everything going",
  "How's everything going",
  "How have you been holding up",
  "I’m really sorry to hear that",
  "I'm really sorry to hear that",
  "Want to talk about what’s bothering",
  "Want to talk about what's bothering",
  "Is there something specific you’re worried",
  "Is there something specific you're worried",
  "Take a deep breath",
  "One step at a time",
  "You’re never alone",
  "You're never alone",
  "Always here for you",
  "as an AI",
  "as a language model",
  "you'll bounce back",
  "you’ve got this",
  "you've got this",
  "focus on what you can control",
  "don’t hesitate to reach out",
  "don't hesitate to reach out",
];

export function containsBannedPhrase(reply: string): boolean {
  const lower = reply.toLowerCase();
  return BANNED_PHRASES.some((p) => lower.includes(p.toLowerCase()));
}

export function countQuestions(reply: string): number {
  return (reply.match(/\?/g) || []).length;
}

export function hasExcessiveLength(reply: string, maxTokens: number): boolean {
  const tokens = reply.split(/\s+/).filter(Boolean).length;
  return tokens > maxTokens * 1.5;
}
