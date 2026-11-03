"""Console script: ``foodtimizer-api``.

Reads configuration from the environment and serves the API. This process is
separate from ``foodtimizer-track``; starting it does not change the
Streamlit app.
"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    from .app import create_app

    host = os.environ.get("FOODTIMIZER_API_HOST", "127.0.0.1")
    port = int(os.environ.get("FOODTIMIZER_API_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)
