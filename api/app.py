"""Alternative Flask entrypoint for Vercel's detector."""

from backend.server import app  # noqa: F401

__all__ = ["app"]
