"""Tests for composite foods (recipes).

A recipe is a composite food built from base ingredients (or entered as
totals). It is persisted in the ``recipes:`` section and materialized into a
derived per-100 g :class:`Ingredient` at load time, so the rest of the system
treats it like any other food.
"""

from __future__ import annotations

import math

import pytest
import yaml

from foodtimizer.config import _problem_from_dict, dump_problem
from foodtimizer.editor import delete_recipe, real_ingredients, upsert_recipe
from foodtimizer.model import (
    Ingredient,
    LibraryMeal,
    MacroTarget,
    Problem,
    Recipe,
    derive_recipe_macros,
)


def _base_problem(recipes: tuple[Recipe, ...] = ()) -> Problem:
    ings = (
        Ingredient(name="eggs", macros={"kcal": 155.0, "protein": 13.0, "carbs": 1.1, "fat": 11.0}),
        Ingredient(name="banana", macros={"kcal": 89.0, "protein": 1.1, "carbs": 23.0, "fat": 0.3}),
    )
    return Problem(
        ingredients=ings,
        targets=(MacroTarget(name="kcal", value=2000.0),),
        meal_library=(LibraryMeal(name="m", tag="snack", ingredients=("eggs",)),),
        recipes=recipes,
    )


def _materialize(problem: Problem) -> Problem:
    """Round-trip through YAML so derived ingredients are rebuilt."""
    return _problem_from_dict(yaml.safe_load(dump_problem(problem)))


def test_derive_from_components_averages_per_100g():
    ing_map = {
        "eggs": Ingredient("eggs", {"kcal": 155.0, "protein": 13.0}),
        "banana": Ingredient("banana", {"kcal": 89.0, "protein": 1.1}),
    }
    recipe = Recipe(name="mix", components={"eggs": 100.0, "banana": 100.0})
    per100 = derive_recipe_macros(recipe, ing_map)
    # 244 kcal across 200 g -> 122 per 100 g.
    assert math.isclose(per100["kcal"], 122.0)
    assert math.isclose(per100["protein"], (13.0 + 1.1) / 2)


def test_finished_weight_override_concentrates_macros():
    ing_map = {"eggs": Ingredient("eggs", {"kcal": 155.0})}
    # 100 g of eggs baked down to a 50 g result: kcal conserved -> doubles.
    recipe = Recipe(name="mix", components={"eggs": 100.0}, total_grams=50.0)
    per100 = derive_recipe_macros(recipe, ing_map)
    assert math.isclose(per100["kcal"], 310.0)


def test_manual_totals_mode():
    recipe = Recipe(name="cake", total_grams=500.0, total_macros={"kcal": 1500.0, "protein": 20.0})
    per100 = derive_recipe_macros(recipe, {})
    assert math.isclose(per100["kcal"], 300.0)
    assert math.isclose(per100["protein"], 4.0)


def test_roundtrip_materializes_and_does_not_duplicate():
    problem = _base_problem(
        recipes=(Recipe(name="mix", components={"eggs": 100.0, "banana": 100.0}),)
    )
    text = dump_problem(problem)
    raw = yaml.safe_load(text)

    # Persisted under recipes, NOT duplicated as a plain ingredient.
    assert "mix" in raw.get("recipes", {})
    assert "mix" not in raw.get("ingredients", {})

    reloaded = _problem_from_dict(raw)
    derived = reloaded.ingredient_by_name("mix")
    assert math.isclose(derived.macros["kcal"], 122.0)
    assert any(r.name == "mix" for r in reloaded.recipes)
    # The Ingredients editor hides recipe-derived foods.
    assert "mix" not in [i.name for i in real_ingredients(reloaded)]


def test_upsert_then_delete_recipe():
    problem = upsert_recipe(_base_problem(), "mix", components={"eggs": 100.0})
    problem = _materialize(problem)
    assert "mix" in [i.name for i in problem.ingredients]

    problem = delete_recipe(problem, "mix")
    problem = _materialize(problem)
    assert "mix" not in [i.name for i in problem.ingredients]
    assert all(r.name != "mix" for r in problem.recipes)


def test_recipe_can_be_used_in_a_meal():
    problem = upsert_recipe(_base_problem(), "mix", components={"eggs": 100.0})
    problem = _materialize(problem)
    # A meal may reference the derived recipe food like any ingredient.
    from foodtimizer.editor import upsert_meal
    from foodtimizer.model import MealIngredient

    problem = upsert_meal(
        problem, "treat", "snack", {"mix": MealIngredient(anchor=50.0)}
    )
    problem = _materialize(problem)
    assert problem.meal_by_name("treat").ingredients == ("mix",)


def test_recipe_name_cannot_clash_with_ingredient():
    with pytest.raises(ValueError):
        upsert_recipe(_base_problem(), "eggs", components={"banana": 100.0})


def test_unknown_component_is_rejected():
    with pytest.raises(ValueError):
        upsert_recipe(_base_problem(), "mix", components={"does_not_exist": 100.0})
