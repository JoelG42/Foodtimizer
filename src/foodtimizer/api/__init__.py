"""HTTP API for a future phone client.

The Streamlit tracker does not import this package. Install it with the
``api`` extra (``pip install -e ".[api]"``). It is intentionally absent from
``requirements.txt``, which is what Streamlit Cloud installs.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
