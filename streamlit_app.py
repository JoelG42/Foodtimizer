"""Entry point for hosted deployment (e.g. Streamlit Community Cloud).

Streamlit Cloud auto-detects a top-level ``streamlit_app.py``. This thin
wrapper makes the ``src/`` package importable and launches the real app.

Configuration is read from Streamlit secrets / environment variables:
- ``[database] url`` (or ``DATABASE_URL``): if set, data is stored in that
  database (recommended for the cloud, where the disk is ephemeral).
- otherwise the app falls back to local files (great for running on your PC).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from foodtimizer.streamlit_app import main  # noqa: E402

main()
