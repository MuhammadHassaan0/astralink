"""Vercel serverless entrypoint that re-exports the Flask app."""

from backend.server import app  # noqa: F401

__all__ = ["app"]
