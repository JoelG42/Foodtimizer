"""Console-script entry point: ``foodtimizer-track``.

Locates the bundled Streamlit app inside the installed package and runs
it with the user's arguments. Using a wrapper (instead of asking the
user to remember the path to ``streamlit_app.py``) means the command
works the same on every machine after ``pip install -e .``.

Argument routing
----------------

Streamlit and the app have different flag namespaces; Streamlit's CLI
separates them with ``--``. We split the user's argv:

- Streamlit config flags (``--server.*``, ``--browser.*``, ``--theme.*``,
  ``--global.*``, ``--logger.*``, ``--runner.*``, ``--client.*``) and the
  ``--lan`` shortcut go *before* the ``--`` so Streamlit consumes them.
- Everything else (``--config``, ``--logs-dir``, ...) goes *after* the
  ``--`` so the app's argparse sees it.

This used to be wrong: every user flag was dumped after ``--``, which
silently broke things like ``--server.address 0.0.0.0`` (Streamlit kept
binding to localhost no matter what the user asked for).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Flag prefixes that belong to Streamlit's own config system, not our
# app. Anything starting with one of these is forwarded verbatim.
_STREAMLIT_FLAG_PREFIXES: tuple[str, ...] = (
    "--server.",
    "--browser.",
    "--theme.",
    "--global.",
    "--logger.",
    "--runner.",
    "--client.",
)


def _streamlit_app_path() -> Path:
    return Path(__file__).resolve().parent / "streamlit_app.py"


def _split_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Return ``(streamlit_flags, app_flags)`` from the user's argv.

    Handles both joined (``--server.port=8888``) and split
    (``--server.port 8888``) forms. ``--lan`` is a convenience shortcut
    that expands to ``--server.address 0.0.0.0`` so users don't have to
    remember the Streamlit-specific name when sharing on a LAN.
    """
    streamlit_flags: list[str] = []
    app_flags: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--lan":
            # Bind to all interfaces so phones / tablets on the same Wi-Fi
            # can reach the tracker via the laptop's LAN IP. Firewall
            # rules on the host machine still need to allow the port.
            streamlit_flags += ["--server.address", "0.0.0.0"]
            i += 1
            continue
        if any(arg.startswith(p) for p in _STREAMLIT_FLAG_PREFIXES):
            streamlit_flags.append(arg)
            # Split form needs the next token consumed as the value;
            # joined "--key=value" is already self-contained.
            if (
                "=" not in arg
                and i + 1 < len(argv)
                and not argv[i + 1].startswith("--")
            ):
                streamlit_flags.append(argv[i + 1])
                i += 2
            else:
                i += 1
            continue
        app_flags.append(arg)
        i += 1
    return streamlit_flags, app_flags


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

    streamlit_flags, app_flags = _split_args(sys.argv[1:])
    sys.argv = ["streamlit", "run", str(app), *streamlit_flags, "--"] + app_flags
    sys.exit(stcli.main())


if __name__ == "__main__":  # pragma: no cover
    main()
