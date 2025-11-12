"""Alternative Flask entrypoint for Vercel's detector."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backend.server import app as backend_app  # noqa: E402

app = backend_app

__all__ = ["app"]
