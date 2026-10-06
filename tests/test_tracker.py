"""Tests for the manual food-log tracker.

The tracker has no LP / scipy involvement — these tests are pure data
plumbing: roundtrip serialization, totals arithmetic, edge cases on
unknown / deleted ingredients, atomic save.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from foodtimizer.model import Ingredient
from foodtimizer.tracker import (
    DayLog,
    LogEntry,
    bulk_add,
    compute_totals,
    list_logged_dates,
    daylog_from_data,
    daylog_to_data,
    load_day_log,
    load_weight_history,
    make_entry,
    rolling_average,
    save_day_log,
    unknown_ingredients,
)


def _ings() -> dict[str, Ingredient]:
    return {
        "chicken": Ingredient(name="chicken", macros={"kcal": 100, "protein": 20, "fat": 1}),
        "rice": Ingredient(name="rice", macros={"kcal": 350, "protein": 9, "carbs": 77}),
    }


def test_log_entry_roundtrip():
    e = LogEntry(
        ingredient="chicken",
        grams=150.0,
        slot="lunch",
        note="cold leftovers",
        eaten_at="2026-05-18T13:30:00",
    )
    rebuilt = LogEntry.from_dict(e.to_dict())
    assert rebuilt == e


def test_log_entry_from_dict_tolerates_missing_optionals():
    e = LogEntry.from_dict({"ingredient": "rice", "grams": 100})
    assert e.ingredient == "rice"
    assert e.grams == 100.0
    assert e.slot is None and e.note is None and e.eaten_at is None


def test_day_log_with_added_remove_update_are_immutable():
    log = DayLog(log_date=date(2026, 5, 18))
    a = LogEntry(ingredient="rice", grams=100)
    b = LogEntry(ingredient="chicken", grams=150)

    log2 = log.with_added(a)
    log3 = log2.with_added(b)

    assert log.entries == ()  # original untouched
    assert log2.entries == (a,)
    assert log3.entries == (a, b)

    log4 = log3.with_removed(0)
    assert log4.entries == (b,)
    assert log3.entries == (a, b)  # untouched

    log5 = log3.with_updated(1, LogEntry(ingredient="chicken", grams=200))
    assert log5.entries[1].grams == 200
    assert log3.entries[1].grams == 150


def test_with_removed_and_with_updated_validate_index():
    log = DayLog(log_date=date(2026, 5, 18)).with_added(LogEntry(ingredient="rice", grams=10))
    with pytest.raises(IndexError):
        log.with_removed(5)
    with pytest.raises(IndexError):
        log.with_updated(5, LogEntry(ingredient="rice", grams=20))


def test_compute_totals_basic():
    log = (
        DayLog(log_date=date(2026, 5, 18))
        .with_added(LogEntry(ingredient="chicken", grams=150))
        .with_added(LogEntry(ingredient="rice", grams=100))
    )
    totals = compute_totals(log, _ings())
    # chicken: 150g -> 150 kcal, 30 g protein
    # rice:    100g -> 350 kcal,  9 g protein, 77 g carbs
    assert totals["kcal"] == pytest.approx(150 + 350)
    assert totals["protein"] == pytest.approx(30 + 9)
    assert totals["carbs"] == pytest.approx(77)
    # Macros absent on an ingredient just don't add anything.
    assert totals.get("fibre", 0.0) == 0.0


def test_compute_totals_skips_unknown_ingredient():
    log = (
        DayLog(log_date=date(2026, 5, 18))
        .with_added(LogEntry(ingredient="chicken", grams=100))
        .with_added(LogEntry(ingredient="cosmic_ray", grams=999))
    )
    totals = compute_totals(log, _ings())
    # cosmic_ray contributes nothing
    assert totals["kcal"] == pytest.approx(100)
    assert unknown_ingredients(log, _ings()) == ["cosmic_ray"]


def test_save_and_load_roundtrip(tmp_path):
    log = (
        DayLog(log_date=date(2026, 5, 18))
        .with_added(LogEntry(ingredient="chicken", grams=150, slot="lunch", eaten_at="2026-05-18T13:30:00"))
        .with_added(LogEntry(ingredient="rice", grams=120, slot="lunch", eaten_at="2026-05-18T13:30:30"))
    )
    save_day_log(tmp_path, log)
    loaded = load_day_log(tmp_path, date(2026, 5, 18))
    assert loaded == log


def test_load_day_log_missing_file_returns_empty(tmp_path):
    log = load_day_log(tmp_path, date(2026, 1, 1))
    assert log.entries == ()
    assert log.log_date == date(2026, 1, 1)


def test_save_writes_atomically(tmp_path):
    """A save replaces the file in-place; no .tmp leftover after success."""
    log = DayLog(log_date=date(2026, 5, 18)).with_added(LogEntry(ingredient="rice", grams=50))
    path = save_day_log(tmp_path, log)
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()

    # The on-disk file is parseable JSON with the expected shape.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["date"] == "2026-05-18"
    assert len(data["entries"]) == 1
    assert data["entries"][0]["ingredient"] == "rice"


def test_list_logged_dates_finds_logs_newest_first(tmp_path):
    for d in (date(2026, 5, 16), date(2026, 5, 18), date(2026, 5, 17)):
        save_day_log(
            tmp_path,
            DayLog(log_date=d).with_added(LogEntry(ingredient="rice", grams=10)),
        )
    # Drop a non-date file to confirm it's ignored.
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")

    dates = list_logged_dates(tmp_path)
    assert dates == [date(2026, 5, 18), date(2026, 5, 17), date(2026, 5, 16)]


def test_list_logged_dates_empty_dir(tmp_path):
    assert list_logged_dates(tmp_path / "nonexistent") == []
    assert list_logged_dates(tmp_path) == []


def test_make_entry_auto_stamps_eaten_at():
    e = make_entry("chicken", 100)
    assert e.eaten_at is not None and "T" in e.eaten_at


def test_make_entry_uses_explicit_timestamp_when_given():
    e = make_entry("chicken", 100, eaten_at="2026-05-18T10:00:00")
    assert e.eaten_at == "2026-05-18T10:00:00"


def test_weight_roundtrip_and_old_logs(tmp_path):
    log = DayLog(log_date=date(2026, 5, 18)).with_weight(78.4)
    assert daylog_from_data(log.log_date, daylog_to_data(log)).weight_kg == pytest.approx(78.4)
    save_day_log(tmp_path, log)
    assert load_day_log(tmp_path, date(2026, 5, 18)).weight_kg == pytest.approx(78.4)

    (tmp_path / "2026-05-17.json").write_text(
        json.dumps({"date": "2026-05-17", "entries": []}),
        encoding="utf-8",
    )
    assert load_day_log(tmp_path, date(2026, 5, 17)).weight_kg is None
    assert load_weight_history(tmp_path) == [(date(2026, 5, 18), 78.4)]


def test_with_weight_rejects_non_positive():
    with pytest.raises(ValueError):
        DayLog(log_date=date(2026, 5, 18)).with_weight(0)


def test_rolling_average_waits_for_full_window():
    values = [80.0, 81.0, 79.0, 78.0, 77.0]
    assert rolling_average(values, 5) == [None, None, None, None, 79.0]


def test_bulk_add_preserves_order():
    log = DayLog(log_date=date(2026, 5, 18))
    entries = [LogEntry(ingredient="rice", grams=i * 10) for i in range(1, 4)]
    log2 = bulk_add(log, entries)
    assert [e.grams for e in log2.entries] == [10, 20, 30]
