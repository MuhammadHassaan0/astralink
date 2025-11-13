"""Vercel serverless entrypoint that re-exports the Flask app."""

from backend.server import app

__all__ = ["app"]
