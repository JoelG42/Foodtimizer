"""Manual food-log tracker.

Lets you write down what you actually ate today (grams of each ingredient
from the config) and compute live macro totals against the targets you
already defined for the optimizer.

Storage is one JSON file per day in a user-chosen ``logs/`` directory.
The on-disk shape is intentionally simple and human-editable::

    {
      "date": "2026-05-18",
      "weight_kg": 78.4,
      "entries": [
        {"ingredient": "chicken_breast", "grams": 150,
         "slot": "lunch", "note": null, "eaten_at": "2026-05-18T13:30:00"},
        ...
      ]
    }

The tracker is intentionally decoupled from the optimizer: it only needs
an ingredient-name -> :class:`Ingredient` mapping (so the same config that
drives the optimizer also drives the tracker).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model import Ingredient


@dataclass(frozen=True)
class LogEntry:
    """A single thing-you-ate event.

    ``slot`` is freeform (typically ``breakfast`` / ``lunch`` / ``dinner`` /
    ``snack`` but anything goes). ``eaten_at`` is an ISO 8601 timestamp set
    when the entry is created; it survives reloads so the UI can render a
    chronological log.
    """

    ingredient: str
    grams: float
    slot: str | None = None
    note: str | None = None
    eaten_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingredient": self.ingredient,
            "grams": float(self.grams),
            "slot": self.slot,
            "note": self.note,
            "eaten_at": self.eaten_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LogEntry":
        return cls(
            ingredient=str(data["ingredient"]),
            grams=float(data["grams"]),
            slot=(str(data["slot"]) if data.get("slot") else None),
            note=(str(data["note"]) if data.get("note") else None),
            eaten_at=(str(data["eaten_at"]) if data.get("eaten_at") else None),
        )


@dataclass(frozen=True)
class DayLog:
    """All entries for a single date.

    Immutable: every mutation returns a fresh ``DayLog``. Makes Streamlit
    rerun semantics predictable and lets the UI compare snapshots cheaply.
    """

    log_date: date
    entries: tuple[LogEntry, ...] = field(default_factory=tuple)
    weight_kg: float | None = None

    def with_added(self, entry: LogEntry) -> "DayLog":
        return replace(self, entries=self.entries + (entry,))

    def with_removed(self, index: int) -> "DayLog":
        if not 0 <= index < len(self.entries):
            raise IndexError(f"Entry index {index} out of range")
        return replace(self, entries=self.entries[:index] + self.entries[index + 1 :])

    def with_updated(self, index: int, entry: LogEntry) -> "DayLog":
        if not 0 <= index < len(self.entries):
            raise IndexError(f"Entry index {index} out of range")
        return replace(
            self,
            entries=self.entries[:index] + (entry,) + self.entries[index + 1 :],
        )

    def with_weight(self, kg: float | None) -> "DayLog":
        """Set or clear the day's body weight in kilograms."""
        if kg is not None and kg <= 0:
            raise ValueError("Body weight must be greater than zero.")
        return replace(self, weight_kg=None if kg is None else float(kg))


def compute_totals(
    log: DayLog,
    ingredients: Mapping[str, Ingredient],
) -> dict[str, float]:
    """Sum every macro across the log's entries (grams * macro-per-100 / 100).

    Entries that reference an unknown ingredient name are silently skipped;
    the UI surfaces them separately so the user can clean them up.
    """
    out: dict[str, float] = {}
    for entry in log.entries:
        ing = ingredients.get(entry.ingredient)
        if ing is None:
            continue
        for macro, per_100g in ing.macros.items():
            out[macro] = out.get(macro, 0.0) + entry.grams * float(per_100g) / 100.0
    return out


def unknown_ingredients(log: DayLog, ingredients: Mapping[str, Ingredient]) -> list[str]:
    """Names referenced by log entries that aren't in the ingredient map."""
    out: list[str] = []
    seen: set[str] = set()
    for entry in log.entries:
        if entry.ingredient not in ingredients and entry.ingredient not in seen:
            out.append(entry.ingredient)
            seen.add(entry.ingredient)
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def daylog_to_data(log: DayLog) -> dict[str, Any]:
    """Serialize a day log to the plain dict shape used on disk / in the DB."""
    return {
        "date": log.log_date.isoformat(),
        "weight_kg": log.weight_kg,
        "entries": [e.to_dict() for e in log.entries],
    }


def daylog_from_data(on_date: date, data: Mapping[str, Any] | None) -> DayLog:
    """Rebuild a :class:`DayLog` from stored data (``None`` -> empty day)."""
    raw = data or {}
    entries_raw = raw.get("entries", []) or []
    raw_weight = raw.get("weight_kg")
    weight = float(raw_weight) if raw_weight is not None else None
    return DayLog(
        log_date=on_date,
        entries=tuple(LogEntry.from_dict(e) for e in entries_raw),
        weight_kg=weight,
    )


def log_path(logs_dir: Path | str, on_date: date) -> Path:
    """Return the on-disk path for the day log (no I/O performed)."""
    return Path(logs_dir) / f"{on_date.isoformat()}.json"


def load_day_log(logs_dir: Path | str, on_date: date) -> DayLog:
    """Load the log for ``on_date``. Returns an empty log if no file exists."""
    path = log_path(logs_dir, on_date)
    if not path.exists():
        return DayLog(log_date=on_date)
    data = json.loads(path.read_text(encoding="utf-8"))
    return daylog_from_data(on_date, data)


def save_day_log(logs_dir: Path | str, log: DayLog) -> Path:
    """Atomically write ``log`` to disk, creating directories as needed."""
    path = log_path(logs_dir, log.log_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = daylog_to_data(log)
    # Atomic write: serialize to a sibling tempfile then rename. Prevents
    # half-written JSON if Streamlit reruns mid-write or the process dies.
    # Using ``with_name`` instead of ``with_suffix`` keeps this safe across
    # Python versions regardless of how strict ``with_suffix`` is about
    # multi-dot suffixes.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def list_logged_dates(logs_dir: Path | str) -> list[date]:
    """Return all dates that have a log file (sorted newest-first)."""
    d = Path(logs_dir)
    if not d.exists():
        return []
    out: list[date] = []
    for f in d.glob("*.json"):
        try:
            out.append(date.fromisoformat(f.stem))
        except ValueError:
            continue
    return sorted(out, reverse=True)


def now_iso() -> str:
    """Current local timestamp, second precision, ISO 8601.

    Wrapped in a function so tests can monkey-patch it.
    """
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def make_entry(
    ingredient: str,
    grams: float,
    *,
    slot: str | None = None,
    note: str | None = None,
    eaten_at: str | None = None,
) -> LogEntry:
    """Build a :class:`LogEntry`, auto-stamping ``eaten_at`` if not given."""
    return LogEntry(
        ingredient=ingredient,
        grams=float(grams),
        slot=(slot or None),
        note=(note or None),
        eaten_at=eaten_at or now_iso(),
    )


def load_weight_history(logs_dir: Path | str) -> list[tuple[date, float]]:
    """Body-weight points from every day log, oldest first.

    Days with no ``weight_kg`` are skipped.
    """
    points: list[tuple[date, float]] = []
    for on_date in sorted(list_logged_dates(logs_dir)):
        log = load_day_log(logs_dir, on_date)
        if log.weight_kg is not None:
            points.append((on_date, log.weight_kg))
    return points


def rolling_average(values: list[float], window: int) -> list[float | None]:
    """Trailing mean of ``window`` weigh-ins, aligned to each value.

    The first ``window - 1`` positions are ``None``. ``values`` must already
    be in chronological order.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
            continue
        chunk = values[i + 1 - window : i + 1]
        out.append(sum(chunk) / window)
    return out


def bulk_add(log: DayLog, entries: Iterable[LogEntry]) -> DayLog:
    """Add many entries in one go (useful when copying from another day)."""
    out = log
    for e in entries:
        out = out.with_added(e)
    return out
