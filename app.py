"""Compat Flask entrypoint for Vercel builder."""

from index import app  # noqa: F401

__all__ = ["app"]
