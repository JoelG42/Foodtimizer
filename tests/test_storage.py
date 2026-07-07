"""Tests for the pluggable persistence layer (:mod:`foodtimizer.storage`).

Verifies that both backends round-trip the config and daily logs identically:

- :class:`FileStorage` (YAML config + one JSON file per day), and
- :class:`SqlStorage` over an in-file SQLite database (a stand-in for the
  Postgres database used in the cloud; both speak the same SQLAlchemy API).
"""

from __future__ import annotations

from datetime import date

import pytest

from foodtimizer.config import problem_to_dict
from foodtimizer.model import (
    Ingredient,
    LibraryMeal,
    MacroTarget,
    Problem,
)
from foodtimizer.storage import FileStorage
from foodtimizer.tracker import DayLog, daylog_from_data, daylog_to_data, make_entry


def _problem() -> Problem:
    ings = (
        Ingredient(
            name="eggs",
            macros={"kcal": 155.0, "protein": 13.0, "carbs": 1.1, "fat": 11.0},
        ),
        Ingredient(
            name="banana",
            macros={"kcal": 89.0, "protein": 1.1, "carbs": 23.0, "fat": 0.3},
        ),
    )
    return Problem(
        ingredients=ings,
        targets=(MacroTarget(name="kcal", value=2000.0),),
        meal_library=(LibraryMeal(name="m", tag="snack", ingredients=("eggs",)),),
    )


def _day() -> DayLog:
    log = DayLog(log_date=date(2026, 7, 7))
    log = log.with_added(make_entry(ingredient="eggs", grams=100.0, slot="breakfast"))
    log = log.with_added(make_entry(ingredient="banana", grams=120.0, slot="snack"))
    return log


def _sqlite_storage(tmp_path):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from foodtimizer.storage import SqlStorage

    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'store.db'}")
    return SqlStorage(engine)


# --- config round-trip ------------------------------------------------------


def test_config_round_trip_file(tmp_path):
    storage = FileStorage(tmp_path / "day.yaml", tmp_path / "logs")
    _check_config_round_trip(storage)


def test_config_round_trip_sql(tmp_path):
    _check_config_round_trip(_sqlite_storage(tmp_path))


def _check_config_round_trip(storage):
    assert storage.load_config() is None  # empty to start

    problem = _problem()
    storage.save_config(problem_to_dict(problem))

    loaded = storage.load_config()
    assert loaded is not None
    # ``ingredients`` is a mapping keyed by name in the YAML schema.
    assert set(loaded["ingredients"]) == {"eggs", "banana"}


# --- day-log round-trip -----------------------------------------------------


def test_day_round_trip_file(tmp_path):
    storage = FileStorage(tmp_path / "day.yaml", tmp_path / "logs")
    _check_day_round_trip(storage)


def test_day_round_trip_sql(tmp_path):
    _check_day_round_trip(_sqlite_storage(tmp_path))


def _check_day_round_trip(storage):
    d = date(2026, 7, 7)
    assert storage.load_day(d) is None

    log = _day()
    storage.save_day(d, daylog_to_data(log))

    restored = daylog_from_data(d, storage.load_day(d))
    assert [e.ingredient for e in restored.entries] == ["eggs", "banana"]
    assert [e.grams for e in restored.entries] == [100.0, 120.0]
    assert restored.entries[0].slot == "breakfast"

    assert storage.logged_dates() == [d]


# --- update semantics -------------------------------------------------------


def test_config_overwrite_sql(tmp_path):
    storage = _sqlite_storage(tmp_path)
    storage.save_config(problem_to_dict(_problem()))

    p2 = _problem()
    p2 = Problem(
        ingredients=p2.ingredients
        + (
            Ingredient(
                name="oats",
                macros={"kcal": 375.0, "protein": 13.0, "carbs": 60.0, "fat": 7.0},
            ),
        ),
        targets=p2.targets,
        meal_library=p2.meal_library,
    )
    storage.save_config(problem_to_dict(p2))

    loaded = storage.load_config()
    assert "oats" in set(loaded["ingredients"])  # upsert replaced, not duplicated


def test_logged_dates_sorted_desc(tmp_path):
    storage = _sqlite_storage(tmp_path)
    for d in (date(2026, 7, 5), date(2026, 7, 9), date(2026, 7, 7)):
        storage.save_day(d, daylog_to_data(DayLog(log_date=d)))
    assert storage.logged_dates() == [
        date(2026, 7, 9),
        date(2026, 7, 7),
        date(2026, 7, 5),
    ]


def test_file_backup_created_on_overwrite(tmp_path):
    cfg = tmp_path / "day.yaml"
    storage = FileStorage(cfg, tmp_path / "logs")
    storage.save_config(problem_to_dict(_problem()))
    storage.save_config(problem_to_dict(_problem()))  # second write -> backup

    backups = list(tmp_path.glob("day.yaml.*.bak"))
    assert backups, "expected a timestamped backup on overwrite"
