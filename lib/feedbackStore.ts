import { randomUUID } from "crypto";

export type FeedbackRating = "up" | "down";

export type FeedbackTag =
  | "too_generic"
  | "too_long"
  | "too_emotional"
  | "too_cold"
  | "wrong_language"
  | "not_like_them"
  | "good_match"
  | "other";

export interface ReplyFeedback {
  id: string;
  userId: string;
  personaId: string;
  replyId: string;
  rating: FeedbackRating;
  tags: FeedbackTag[];
  comment?: string;
  createdAt: Date;
}

// In-memory store (TODO: replace with DB/ORM)
const feedbackStore: ReplyFeedback[] = [];

export async function saveFeedback(feedback: ReplyFeedback): Promise<void> {
  feedbackStore.push(feedback);
}

export async function getFeedbackForPersona(personaId: string): Promise<ReplyFeedback[]> {
  return feedbackStore.filter((f) => f.personaId === personaId);
}

export async function getFeedbackForReply(replyId: string): Promise<ReplyFeedback[]> {
  return feedbackStore.filter((f) => f.replyId === replyId);
}

// Helper to create feedback ids if needed elsewhere
export function newFeedbackId(): string {
  return randomUUID();
}
