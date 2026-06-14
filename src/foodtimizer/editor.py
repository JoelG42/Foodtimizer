"""Pure, in-memory edits to a :class:`Problem`.

The Streamlit UI lets the user manage their ingredient database and meal
library without hand-editing YAML. Every function here is side-effect free:
it takes a :class:`Problem` and returns a *new* :class:`Problem` with the
requested change applied. Persistence is handled separately by
:func:`foodtimizer.config.save_problem`.

Keeping these as plain data transforms (rather than methods on the frozen
dataclasses) makes them trivial to unit-test and keeps the model module
focused on describing the optimization problem.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .model import (
    Ingredient,
    IngredientBound,
    LibraryMeal,
    MealIngredient,
    Problem,
)

# Macros shown by default in the UI even when an ingredient hasn't set them.
DEFAULT_MACROS: tuple[str, ...] = ("kcal", "protein", "carbs", "fat", "fibre")

# Bound fields a user can attach to an ingredient from the UI, in display order.
INGREDIENT_BOUND_FIELDS: tuple[str, ...] = (
    "step",
    "per_meal_min",
    "per_meal_max",
    "total_min",
    "total_max",
)


def all_macro_keys(problem: Problem) -> list[str]:
    """Every macro name in use, default macros first, then any extras."""
    seen: list[str] = list(DEFAULT_MACROS)
    for ing in problem.ingredients:
        for key in ing.macros:
            if key not in seen:
                seen.append(key)
    return seen


# ---------------------------------------------------------------------------
# Ingredients
# ---------------------------------------------------------------------------


def upsert_ingredient(
    problem: Problem,
    name: str,
    macros: Mapping[str, float],
    bounds: Mapping[str, float] | None = None,
    *,
    original_name: str | None = None,
) -> Problem:
    """Add or update an ingredient (and its optional bounds).

    ``original_name`` lets the caller rename an ingredient in place: the old
    entry (and its bounds) is removed and references in the meal library are
    rewritten to the new name. ``bounds`` keys are the grams-valued bound
    fields (``step``, ``per_meal_min`` …); empty/``None`` values are dropped.
    """
    name = name.strip()
    if not name:
        raise ValueError("Ingredient name cannot be empty.")

    clean_macros = {
        str(k): float(v)
        for k, v in macros.items()
        if v is not None and str(k).strip()
    }

    old = (original_name or name).strip()
    new_ing = Ingredient(name=name, macros=clean_macros)

    kept = [i for i in problem.ingredients if i.name not in {name, old}]
    kept.append(new_ing)

    new_bounds = dict(problem.bounds)
    new_bounds.pop(old, None)
    new_bounds.pop(name, None)
    if bounds:
        bound = _bound_from_fields(bounds)
        if bound is not None:
            new_bounds[name] = bound

    new_problem = replace(
        problem, ingredients=tuple(kept), bounds=new_bounds
    )

    if old != name:
        new_problem = _rename_ingredient_in_meals(new_problem, old, name)
    return new_problem


def delete_ingredient(problem: Problem, name: str) -> Problem:
    """Remove an ingredient, its bounds, and strip it from every meal."""
    ingredients = tuple(i for i in problem.ingredients if i.name != name)
    bounds = {k: v for k, v in problem.bounds.items() if k != name}

    meals = []
    for meal in problem.meal_library:
        if name in meal.ingredients:
            new_ings = tuple(i for i in meal.ingredients if i != name)
            new_specs = {
                k: v for k, v in meal.ingredient_specs.items() if k != name
            }
            meals.append(
                replace(meal, ingredients=new_ings, ingredient_specs=new_specs)
            )
        else:
            meals.append(meal)
    return replace(
        problem,
        ingredients=ingredients,
        bounds=bounds,
        meal_library=tuple(meals),
    )


def _bound_from_fields(fields: Mapping[str, float]) -> IngredientBound | None:
    vals = {
        k: float(v)
        for k, v in fields.items()
        if v is not None and k in INGREDIENT_BOUND_FIELDS
    }
    if not vals:
        return None
    return IngredientBound(
        total_min=vals.get("total_min"),
        total_max=vals.get("total_max"),
        per_meal_min=vals.get("per_meal_min"),
        per_meal_max=vals.get("per_meal_max"),
        step=vals.get("step"),
    )


def _rename_ingredient_in_meals(
    problem: Problem, old: str, new: str
) -> Problem:
    meals = []
    for meal in problem.meal_library:
        if old not in meal.ingredients:
            meals.append(meal)
            continue
        new_ings = tuple(new if i == old else i for i in meal.ingredients)
        new_specs = {
            (new if k == old else k): v for k, v in meal.ingredient_specs.items()
        }
        meals.append(replace(meal, ingredients=new_ings, ingredient_specs=new_specs))
    return replace(problem, meal_library=tuple(meals))


# ---------------------------------------------------------------------------
# Meals
# ---------------------------------------------------------------------------


def upsert_meal(
    problem: Problem,
    name: str,
    tag: str,
    ingredient_specs: Mapping[str, MealIngredient],
    macro_min: Mapping[str, float] | None = None,
    macro_max: Mapping[str, float] | None = None,
    *,
    original_name: str | None = None,
) -> Problem:
    """Add or update a meal in the library.

    ``ingredient_specs`` is an *ordered* mapping of ingredient name to its
    per-meal spec; the keys define the meal's ingredient list and order.
    Every referenced ingredient must already exist in the problem.
    """
    name = name.strip()
    tag = tag.strip()
    if not name:
        raise ValueError("Meal name cannot be empty.")
    if not tag:
        raise ValueError("Meal must have a tag (e.g. breakfast, lunch_dinner, snack).")
    if not ingredient_specs:
        raise ValueError("A meal needs at least one ingredient.")

    known = {i.name for i in problem.ingredients}
    for ing in ingredient_specs:
        if ing not in known:
            raise ValueError(f"Unknown ingredient in meal: {ing!r}")

    meal = LibraryMeal(
        name=name,
        tag=tag,
        ingredients=tuple(ingredient_specs.keys()),
        macro_min=dict(macro_min or {}),
        macro_max=dict(macro_max or {}),
        ingredient_specs=dict(ingredient_specs),
    )

    old = (original_name or name).strip()
    kept = [m for m in problem.meal_library if m.name not in {name, old}]
    kept.append(meal)
    return replace(problem, meal_library=tuple(kept))


def delete_meal(problem: Problem, name: str) -> Problem:
    """Remove a meal and drop it from the default day plan."""
    meals = tuple(m for m in problem.meal_library if m.name != name)
    day = {k: v for k, v in problem.default_day.items() if v != name}
    return replace(problem, meal_library=meals, default_day=day)
