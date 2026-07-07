"""Pluggable persistence for the config and daily logs.

The app used to read/write plain files (a YAML config + one JSON file per
day). That's perfect locally, but breaks on hosts with an ephemeral disk
(e.g. Streamlit Community Cloud), where anything written is wiped on restart.

This module abstracts persistence behind a tiny interface with two backends:

- :class:`FileStorage` — the original file behavior (default, for local use).
- :class:`SqlStorage` — a database-backed key/value store (for the cloud),
  working over any SQLAlchemy URL (Postgres in production, SQLite in tests).

Both store the *same* plain dicts:

- the config document (as produced by :func:`config.problem_to_dict`), and
- one day-log document per date (as produced by
  :func:`tracker.daylog_to_data`).

A process-wide "active" backend is configured once at startup via
:func:`configure`; the module-level helpers then dispatch to it, so callers
don't have to thread a storage object through every function.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Protocol

import yaml

from .config import config_dict_to_yaml, validate_config_dict


class Storage(Protocol):
    """Persistence backend interface."""

    def load_config(self) -> Optional[dict[str, Any]]: ...
    def save_config(self, data: dict[str, Any]) -> None: ...
    def load_day(self, on_date: date) -> Optional[dict[str, Any]]: ...
    def save_day(self, on_date: date, data: dict[str, Any]) -> None: ...
    def logged_dates(self) -> list[date]: ...
    def location_label(self) -> str: ...
    def is_file_backed(self) -> bool: ...


class FileStorage:
    """Original behavior: a YAML config file + one JSON file per day."""

    def __init__(self, config_path: str | Path, logs_dir: str | Path) -> None:
        self.config_path = Path(config_path)
        self.logs_dir = Path(logs_dir)

    def load_config(self) -> Optional[dict[str, Any]]:
        if not self.config_path.exists():
            return None
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def save_config(self, data: dict[str, Any]) -> None:
        text = config_dict_to_yaml(data)  # validates by re-parsing
        target = self.config_path
        # Keep a one-off timestamped backup the first time we overwrite a
        # hand-curated file, so nothing is ever lost.
        if target.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            bak = target.with_name(f"{target.name}.{stamp}.bak")
            if not bak.exists():
                bak.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)

    def load_day(self, on_date: date) -> Optional[dict[str, Any]]:
        path = self.logs_dir / f"{on_date.isoformat()}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_day(self, on_date: date, data: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / f"{on_date.isoformat()}.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(path)

    def logged_dates(self) -> list[date]:
        if not self.logs_dir.exists():
            return []
        out: list[date] = []
        for f in self.logs_dir.glob("*.json"):
            try:
                out.append(date.fromisoformat(f.stem))
            except ValueError:
                continue
        return sorted(out, reverse=True)

    def location_label(self) -> str:
        return str(self.config_path)

    def is_file_backed(self) -> bool:
        return True


# Key used for the single config document in the SQL key/value table.
_CONFIG_KEY = "config"
_LOG_PREFIX = "log:"


class SqlStorage:
    """Database-backed store: one row per document in a simple key/value table.

    Works on any SQLAlchemy URL. Values are stored as JSON text (portable
    across SQLite and Postgres), and upserts use the ``ON CONFLICT`` syntax
    supported by both.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._ensure_table()

    def _ensure_table(self) -> None:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS foodtimizer_kv ("
                    "k TEXT PRIMARY KEY, v TEXT NOT NULL)"
                )
            )

    def _get(self, key: str) -> Optional[dict[str, Any]]:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT v FROM foodtimizer_kv WHERE k = :k"), {"k": key}
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _set(self, key: str, data: dict[str, Any]) -> None:
        from sqlalchemy import text

        payload = json.dumps(data, ensure_ascii=False)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO foodtimizer_kv (k, v) VALUES (:k, :v) "
                    "ON CONFLICT (k) DO UPDATE SET v = :v"
                ),
                {"k": key, "v": payload},
            )

    def load_config(self) -> Optional[dict[str, Any]]:
        return self._get(_CONFIG_KEY)

    def save_config(self, data: dict[str, Any]) -> None:
        validate_config_dict(data)
        self._set(_CONFIG_KEY, data)

    def load_day(self, on_date: date) -> Optional[dict[str, Any]]:
        return self._get(f"{_LOG_PREFIX}{on_date.isoformat()}")

    def save_day(self, on_date: date, data: dict[str, Any]) -> None:
        self._set(f"{_LOG_PREFIX}{on_date.isoformat()}", data)

    def logged_dates(self) -> list[date]:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT k FROM foodtimizer_kv WHERE k LIKE :p"),
                {"p": f"{_LOG_PREFIX}%"},
            ).fetchall()
        out: list[date] = []
        for (k,) in rows:
            try:
                out.append(date.fromisoformat(k[len(_LOG_PREFIX):]))
            except ValueError:
                continue
        return sorted(out, reverse=True)

    def location_label(self) -> str:
        return "the cloud database"

    def is_file_backed(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Process-wide active backend + dispatch helpers
# ---------------------------------------------------------------------------

_active: Optional[Storage] = None


def configure(storage: Storage) -> None:
    global _active
    _active = storage


def active() -> Storage:
    if _active is None:
        raise RuntimeError("storage backend not configured; call configure() first")
    return _active


def load_config() -> Optional[dict[str, Any]]:
    return active().load_config()


def save_config(data: dict[str, Any]) -> None:
    active().save_config(data)


def load_day(on_date: date) -> Optional[dict[str, Any]]:
    return active().load_day(on_date)


def save_day(on_date: date, data: dict[str, Any]) -> None:
    active().save_day(on_date, data)


def logged_dates() -> list[date]:
    return active().logged_dates()


def location_label() -> str:
    return active().location_label()


def is_file_backed() -> bool:
    return active().is_file_backed()
