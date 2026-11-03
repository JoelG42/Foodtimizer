"""Per-meal totals shown on the right of each meal card header."""

from __future__ import annotations

import pytest

from foodtimizer.model import Ingredient
from foodtimizer.streamlit_app import _meal_header_html, _slot_macro_totals
from foodtimizer.tracker import LogEntry


_INGREDIENTS = {
    "eggs": Ingredient(
        name="eggs",
        macros={"kcal": 155.0, "protein": 13.0, "carbs": 1.1, "fat": 11.0},
    ),
    "oats": Ingredient(
        name="oats",
        macros={"kcal": 389.0, "protein": 16.9, "carbs": 66.3, "fat": 6.9},
    ),
}


def test_totals_sum_every_entry_in_the_meal():
    entries = [
        (0, LogEntry(ingredient="eggs", grams=100.0, slot="breakfast")),
        (3, LogEntry(ingredient="oats", grams=50.0, slot="breakfast")),
    ]
    totals = _slot_macro_totals(entries, _INGREDIENTS)
    assert totals["kcal"] == pytest.approx(155.0 + 194.5)
    assert totals["protein"] == pytest.approx(13.0 + 8.45)
    assert totals["carbs"] == pytest.approx(1.1 + 33.15)
    assert totals["fat"] == pytest.approx(11.0 + 3.45)


def test_unknown_ingredients_and_missing_macros_count_as_zero():
    ingredients = {"water": Ingredient(name="water", macros={"kcal": 0.0})}
    entries = [
        (0, LogEntry(ingredient="water", grams=500.0)),
        (1, LogEntry(ingredient="deleted_food", grams=100.0)),
    ]
    assert _slot_macro_totals(entries, ingredients) == {
        "kcal": 0.0,
        "protein": 0.0,
        "carbs": 0.0,
        "fat": 0.0,
    }


def test_empty_meal_shows_zeros():
    header = _meal_header_html("🌅 Breakfast", _slot_macro_totals([], _INGREDIENTS))
    assert "0 kcal" in header
    assert "🥩&nbsp;0g" in header
    assert "🌾&nbsp;0g" in header
    assert "🫒&nbsp;0g" in header


def test_header_shows_rounded_macros_with_emojis():
    header = _meal_header_html(
        "🥗 Lunch", {"kcal": 512.4, "protein": 41.6, "carbs": 52.2, "fat": 14.5}
    )
    assert "🥗 Lunch" in header
    assert "512 kcal" in header
    assert "🥩&nbsp;42g" in header
    assert "🌾&nbsp;52g" in header
    assert "🫒&nbsp;14g" in header


def test_custom_slot_names_are_escaped():
    header = _meal_header_html("🍴 <b>late</b>", {"kcal": 0.0})
    assert "<b>late</b>" not in header
    assert "&lt;b&gt;late&lt;/b&gt;" in header
