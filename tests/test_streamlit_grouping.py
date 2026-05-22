"""Tests for the per-slot grouping helper in the Streamlit log view.

We only test the pure helper, not Streamlit rendering — running the
app requires the Streamlit runtime. Grouping is what could silently
break the delete-button behaviour if the original index threading
were ever wrong, so it's the bit worth pinning down.
"""

from __future__ import annotations

from foodtimizer.tracker import LogEntry
from foodtimizer.streamlit_app import (
    _LOG_SLOT_ORDER,
    _UNASSIGNED_SLOT_LABEL,
    _base_slot,
    _expand_slot_labels,
    _group_entries_by_slot,
    _is_first_instance,
)


def _entries(*specs: tuple[str, str | None]) -> tuple[LogEntry, ...]:
    """Build a tuple of LogEntries from ``(ingredient, slot)`` pairs."""
    return tuple(
        LogEntry(ingredient=name, grams=100.0, slot=slot)
        for name, slot in specs
    )


def test_canonical_slot_order_regardless_of_log_order():
    """Entries logged in random order still render in
    breakfast -> lunch -> dinner -> snack order."""
    entries = _entries(
        ("oats", "snack"),
        ("rice", "lunch"),
        ("eggs", "breakfast"),
        ("salmon", "dinner"),
    )
    groups = _group_entries_by_slot(entries)
    labels = [slot for slot, _ in groups]
    assert labels == list(_LOG_SLOT_ORDER)


def test_original_indices_are_preserved():
    """The 2nd field of each (idx, entry) tuple must match the entry's
    position in the input — otherwise delete buttons would target the
    wrong row after grouping."""
    entries = _entries(
        ("oats", "snack"),       # original idx 0
        ("rice", "lunch"),       # original idx 1
        ("eggs", "breakfast"),   # original idx 2
        ("salmon", "dinner"),    # original idx 3
    )
    groups = dict(_group_entries_by_slot(entries))
    assert groups["breakfast"][0][0] == 2
    assert groups["lunch"][0][0] == 1
    assert groups["dinner"][0][0] == 3
    assert groups["snack"][0][0] == 0
    # And the entry references are intact.
    assert groups["breakfast"][0][1].ingredient == "eggs"


def test_unassigned_entries_go_last_under_their_own_header():
    entries = _entries(
        ("eggs", "breakfast"),
        ("mystery_bar", None),
        ("rice", "lunch"),
    )
    groups = _group_entries_by_slot(entries)
    labels = [slot for slot, _ in groups]
    assert labels[-1] == _UNASSIGNED_SLOT_LABEL
    assert labels[:-1] == ["breakfast", "lunch"]


def test_unknown_slot_names_render_after_canonical_alphabetically():
    """A user-defined slot like ``pre_workout`` should appear after the
    canonical ones and be alphabetized vs other custom slots."""
    entries = _entries(
        ("toast", "breakfast"),
        ("banana", "pre_workout"),
        ("protein", "post_workout"),
        ("apple", "snack"),
    )
    labels = [slot for slot, _ in _group_entries_by_slot(entries)]
    # Canonical first, then custom slots alphabetized.
    assert labels == ["breakfast", "snack", "post_workout", "pre_workout"]


def test_multiple_entries_per_slot_kept_in_log_order():
    """Within one slot, entries stay in the order they were logged.
    That's the contract delete buttons + the visual log rely on."""
    entries = _entries(
        ("toast", "breakfast"),
        ("eggs", "breakfast"),
        ("banana", "breakfast"),
    )
    groups = dict(_group_entries_by_slot(entries))
    indices = [idx for idx, _ in groups["breakfast"]]
    assert indices == [0, 1, 2]


def test_expand_slot_labels_keeps_bare_name_when_count_is_one():
    out = _expand_slot_labels(["breakfast", "snack"], {"breakfast": 1, "snack": 1})
    assert out == ["breakfast", "snack"]


def test_expand_slot_labels_numbers_instances_when_count_above_one():
    out = _expand_slot_labels(["breakfast", "snack"], {"breakfast": 1, "snack": 3})
    assert out == ["breakfast", "snack #1", "snack #2", "snack #3"]


def test_expand_slot_labels_treats_missing_or_zero_count_as_one():
    assert _expand_slot_labels(["snack"], {}) == ["snack"]
    assert _expand_slot_labels(["snack"], {"snack": 0}) == ["snack"]


def test_base_slot_recovers_type_from_instance_label():
    assert _base_slot("snack") == "snack"
    assert _base_slot("snack #1") == "snack"
    assert _base_slot("snack #5") == "snack"
    assert _base_slot("lunch") == "lunch"


def test_is_first_instance_for_default_meal_seeding():
    assert _is_first_instance("snack") is True
    assert _is_first_instance("snack #1") is True
    assert _is_first_instance("snack #2") is False
    assert _is_first_instance("snack #5") is False
