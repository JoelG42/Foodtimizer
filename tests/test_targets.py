"""Tests for editing macro goals (targets) via the editor helpers."""

from __future__ import annotations

import pytest
import yaml

from foodtimizer.config import _problem_from_dict, dump_problem
from foodtimizer.editor import delete_target, upsert_target
from foodtimizer.model import (
    Ingredient,
    LibraryMeal,
    MacroTarget,
    Problem,
)


def _problem() -> Problem:
    ings = (
        Ingredient(
            name="chicken",
            macros={"kcal": 165.0, "protein": 31.0, "carbs": 0.0, "fat": 3.6},
        ),
    )
    return Problem(
        ingredients=ings,
        targets=(
            MacroTarget(name="kcal", value=2000.0),
            MacroTarget(name="protein", value=150.0, weight=100.0),
        ),
        meal_library=(LibraryMeal(name="m", tag="snack", ingredients=("chicken",)),),
    )


def test_upsert_updates_value_and_preserves_weight():
    p = upsert_target(_problem(), "protein", value=180.0, weight=100.0)
    protein = next(t for t in p.targets if t.name == "protein")
    assert protein.value == 180.0
    assert protein.weight == 100.0  # preserved priority


def test_upsert_adds_new_target():
    p = upsert_target(_problem(), "fat", value=70.0)
    assert any(t.name == "fat" and t.value == 70.0 for t in p.targets)
    assert len(p.targets) == 3


def test_upsert_requires_a_value_or_bound():
    with pytest.raises(ValueError):
        upsert_target(_problem(), "carbs")


def test_delete_target_removes_it():
    p = delete_target(_problem(), "protein")
    assert {t.name for t in p.targets} == {"kcal"}


def test_delete_last_target_raises():
    p = _problem()
    p = delete_target(p, "protein")
    with pytest.raises(ValueError):
        delete_target(p, "kcal")


def test_goal_change_survives_yaml_round_trip():
    p = upsert_target(_problem(), "kcal", value=1800.0)
    reloaded = _problem_from_dict(yaml.safe_load(dump_problem(p)))
    kcal = next(t for t in reloaded.targets if t.name == "kcal")
    assert kcal.value == 1800.0
