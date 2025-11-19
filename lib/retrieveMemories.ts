import OpenAI from "openai";
import { vectorSearch } from "./memoryStore";
import { ROUTER_CLASSIFIER_PROMPT } from "./prompts";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function retrieveRelevantMemories({
  personaId,
  userId,
  userMessage,
  maxMemories = 5,
}: {
  personaId: string;
  userId: string;
  userMessage: string;
  maxMemories?: number;
}) {
  const routerPrompt = ROUTER_CLASSIFIER_PROMPT.replace(
    "{{USER_MESSAGE}}",
    userMessage
  );
  await openai.chat.completions
    .create({
      model: process.env.OPENAI_MODEL_CLASSIFIER || "gpt-4o-mini",
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [{ role: "system", content: routerPrompt }],
    })
    .catch(() => null);

  const embedResp = await openai.embeddings.create({
    model: process.env.OPENAI_MODEL_EMBED || "text-embedding-3-small",
    input: userMessage,
  });
  const queryEmbedding = embedResp.data[0]?.embedding || [];
  return vectorSearch(personaId, userId, queryEmbedding, maxMemories);
}

export async function retrieveMemoriesForChat(params: {
  personaId: string;
  userId: string;
  userMessage: string;
  maxMemories?: number;
}) {
  return retrieveRelevantMemories(params);
}
