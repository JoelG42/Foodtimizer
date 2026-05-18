"""Console-script entry point: ``foodtimizer-track``.

Locates the bundled Streamlit app inside the installed package and runs
it with the user's arguments. We do this (rather than asking the user to
remember the path to ``streamlit_app.py``) so the command works the same
on every machine after ``pip install -e .``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _streamlit_app_path() -> Path:
    return Path(__file__).resolve().parent / "streamlit_app.py"


def main() -> None:
    try:
        from streamlit.web import cli as stcli  # type: ignore
    except ImportError as e:  # pragma: no cover - user-facing error path
        raise SystemExit(
            "Streamlit is not installed. Install it with:  "
            'pip install "foodtimizer[app]"'
        ) from e

    app = _streamlit_app_path()
    if not app.exists():  # pragma: no cover - shouldn't happen post-install
        raise SystemExit(f"Bundled tracker app not found at {app}")

    # Streamlit's CLI reads sys.argv; rewrite it so the user's flags after
    # `foodtimizer-track` end up as the app's argv (separated by `--`).
    sys.argv = ["streamlit", "run", str(app), "--"] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":  # pragma: no cover
    main()
