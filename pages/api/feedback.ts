import type { NextApiRequest, NextApiResponse } from "next";
import { randomUUID } from "crypto";
import {
  FeedbackRating,
  FeedbackTag,
  ReplyFeedback,
  saveFeedback,
} from "../../lib/feedbackStore";

const VALID_TAGS: FeedbackTag[] = [
  "too_generic",
  "too_long",
  "too_emotional",
  "too_cold",
  "wrong_language",
  "not_like_them",
  "good_match",
  "other",
];

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { userId, personaId, replyId, rating, tags, comment } = req.body || {};

    if (!userId || !personaId || !replyId || !rating) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    if (!(["up", "down"] as FeedbackRating[]).includes(rating)) {
      return res.status(400).json({ error: "Invalid rating" });
    }

    const tagArray: FeedbackTag[] = Array.isArray(tags)
      ? tags.filter((t: string) => VALID_TAGS.includes(t as FeedbackTag))
      : [];

    const feedback: ReplyFeedback = {
      id: randomUUID(),
      userId: String(userId),
      personaId: String(personaId),
      replyId: String(replyId),
      rating: rating as FeedbackRating,
      tags: tagArray,
      comment,
      createdAt: new Date(),
    };

    await saveFeedback(feedback);

    return res.status(200).json({ ok: true });
  } catch (err) {
    console.error("Error in /api/feedback:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
}
