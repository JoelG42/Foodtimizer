"""Data model for the nutrition optimization problem.

All macro quantities on `Ingredient` are expressed *per 100 g* of the
ingredient. Decision variables (in `Plan`) are in **grams**.

Concept
-------

You maintain a **library of meals**. Each meal is a list of ingredients plus
a `tag` (e.g. ``"breakfast"``, ``"lunch_dinner"``, ``"snack"``). Tag-level
constraints (`TagConstraints`) like ``kcal_max`` apply to any slot whose
meal carries that tag.

At optimization time you supply a **day plan** — a mapping
``{slot_name: meal_name}`` (e.g. ``{"breakfast": "oats_breakfast",
"lunch": "chicken_rice", "dinner": "gnocchi_bolognese",
"snack": "protein_shake"}``). The optimizer then computes the gram amount
of every ingredient in every chosen meal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Ingredient:
    """An ingredient with its macro profile per 100 g."""

    name: str
    macros: Mapping[str, float]

    def amount_for(self, macro: str) -> float:
        return float(self.macros.get(macro, 0.0))


@dataclass(frozen=True)
class MacroTarget:
    """A target for a single macro.

    Soft targets: deviations are penalized in the objective with `weight`.
    Hard targets: enforced as an equality constraint (no slack).

    `lower` / `upper`, when set, add hard inequality bounds. They are
    independent of `value` and may be set without it (`value=None`).
    """

    name: str
    value: float | None = None
    weight: float = 1.0
    hard: bool = False
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class TagConstraints:
    """Constraints applied to any slot whose meal carries this tag.

    `macro_min` / `macro_max` express per-slot minimums and maximums for any
    macro (kcal, protein, carbs, fat, fibre, …). The YAML may also use the
    shortcut keys `kcal_max` / `kcal_min` at the top level; these are folded
    into `macro_max` / `macro_min` by the parser.
    """

    macro_min: Mapping[str, float] = field(default_factory=dict)
    macro_max: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LibraryMeal:
    """A named, tagged meal in the user's library.

    `ingredients` is the (ordered) list of ingredient *names* in the meal.
    `ingredient_specs` optionally carries per-(meal, ingredient) bounds and
    role labels (e.g. ``main``); missing entries cascade to the global
    `ingredient_bounds` and then to `defaults`.

    Per-meal `macro_min` / `macro_max` override the tag defaults for the
    specific macros they mention.
    """

    name: str
    tag: str
    ingredients: tuple[str, ...]
    macro_min: Mapping[str, float] = field(default_factory=dict)
    macro_max: Mapping[str, float] = field(default_factory=dict)
    ingredient_specs: Mapping[str, MealIngredient] = field(default_factory=dict)


@dataclass(frozen=True)
class Defaults:
    """Global default bounds applied to every (slot, ingredient) variable.

    See :class:`Problem` for the full resolution semantics. In short:
    ``min`` is the *max* of every applicable lower bound (defaults, global
    ingredient bound, per-meal spec, `main` flag) and ``max`` is the *min*
    of every applicable upper bound. If the resulting min exceeds the max,
    min is silently capped at max so the LP stays feasible.

    Use ``per_meal_min`` to enforce "any listed ingredient must actually
    be used" (e.g. 5 g). Use ``per_meal_max`` as a global sanity ceiling
    against ballooning low-density ingredients (e.g. 300 g). Use
    ``main_min`` to set the default minimum for ingredients labelled
    ``main`` in a meal definition (e.g. 80 g).
    """

    per_meal_min: float | None = None
    per_meal_max: float | None = None
    main_min: float | None = None
    # L1 penalty per gram of deviation from a meal-ingredient's ``anchor``.
    # Small relative to macro target weights so the anchor only matters when
    # macros are otherwise indifferent. ``None`` disables anchor penalties.
    anchor_weight: float | None = None


@dataclass(frozen=True)
class MealIngredient:
    """Per-(meal, ingredient) spec from a meal definition.

    Each field is optional; absent fields fall through to the global
    ingredient bound, then to the defaults.

    ``anchor`` is the *typical* amount (in grams) you'd use in this meal.
    The optimizer pays an L1 penalty ``defaults.anchor_weight * |x - anchor|``
    for each gram it deviates, which keeps amounts recipe-shaped without
    making the problem infeasible. ``anchor`` does not impose any min/max:
    use ``min``/``max`` (or the inline ingredient bounds) for hard limits.
    """

    min: float | None = None
    max: float | None = None
    main: bool = False
    anchor: float | None = None


@dataclass(frozen=True)
class IngredientBound:
    """Optional bounds and quantization for a single ingredient.

    - ``total_*`` apply across all slots (in grams).
    - ``per_meal_*`` apply within a single slot (in grams).
    - ``step`` quantizes the gram amount: the ingredient is consumed in
      non-negative integer multiples of ``step`` grams (e.g. eggs in 55-g
      units, tortillas in 60-g units). When ``step`` is set, the variable
      for this ingredient becomes integer in the MILP.
    """

    total_min: float | None = None
    total_max: float | None = None
    per_meal_min: float | None = None
    per_meal_max: float | None = None
    step: float | None = None


@dataclass(frozen=True)
class Recipe:
    """A composite food assembled from other ingredients (or entered as
    totals), e.g. a home-baked banana bread.

    Two ways to define one:

    - **From components**: ``components`` maps ingredient name -> grams used.
      Total macros are summed from those ingredients. ``total_grams``, if
      given, overrides the finished weight (handy for baking, where water
      evaporates so the result weighs less than the raw mix — calories are
      unchanged, so per-100 g goes up).
    - **From totals**: leave ``components`` empty and give ``total_grams``
      plus ``total_macros`` (absolute macros for the whole batch).

    Either way it yields a derived :class:`Ingredient` (macros per 100 g)
    named the same, so you can log it by grams like any other food.
    """

    name: str
    components: Mapping[str, float] = field(default_factory=dict)
    total_grams: float | None = None
    total_macros: Mapping[str, float] = field(default_factory=dict)


def derive_recipe_macros(
    recipe: Recipe, ingredients: Mapping[str, Ingredient]
) -> dict[str, float]:
    """Compute a recipe's macros **per 100 g** of the finished food.

    Returns an empty dict if the recipe has no usable quantity yet (so the
    UI can show an empty-state preview without blowing up).
    """
    if recipe.components:
        totals: dict[str, float] = {}
        raw_grams = 0.0
        for ing_name, grams in recipe.components.items():
            grams = float(grams)
            raw_grams += grams
            ing = ingredients.get(ing_name)
            if ing is None:
                continue
            for macro, per_100g in ing.macros.items():
                totals[macro] = totals.get(macro, 0.0) + grams * float(per_100g) / 100.0
        total_grams = recipe.total_grams if recipe.total_grams else raw_grams
    else:
        totals = {k: float(v) for k, v in recipe.total_macros.items()}
        total_grams = recipe.total_grams or 0.0

    if not total_grams or total_grams <= 0:
        return {}
    return {macro: value / total_grams * 100.0 for macro, value in totals.items()}


@dataclass(frozen=True)
class Problem:
    """The static configuration: ingredients, targets, library, constraints.

    The *day selection* (which meal goes in which slot) is supplied to
    `optimize()` separately. A `default_day` may be provided in the config
    and used when no explicit day plan is given.

    ``recipes`` are composite foods; each is materialized into a derived
    entry in ``ingredients`` (per-100 g macros) by the config loader, so the
    rest of the system treats them like ordinary ingredients.
    """

    ingredients: tuple[Ingredient, ...]
    targets: tuple[MacroTarget, ...]
    meal_library: tuple[LibraryMeal, ...]
    tag_constraints: Mapping[str, TagConstraints] = field(default_factory=dict)
    bounds: Mapping[str, IngredientBound] = field(default_factory=dict)
    default_day: Mapping[str, str] = field(default_factory=dict)
    defaults: Defaults = field(default_factory=Defaults)
    recipes: tuple[Recipe, ...] = ()

    def ingredient_by_name(self, name: str) -> Ingredient:
        for ing in self.ingredients:
            if ing.name == name:
                return ing
        raise KeyError(f"Unknown ingredient: {name!r}")

    def recipe_by_name(self, name: str) -> Recipe:
        for recipe in self.recipes:
            if recipe.name == name:
                return recipe
        raise KeyError(f"Unknown recipe: {name!r}")

    def meal_by_name(self, name: str) -> LibraryMeal:
        for meal in self.meal_library:
            if meal.name == name:
                return meal
        raise KeyError(f"Unknown meal: {name!r}")

    def meals_by_tag(self, tag: str) -> tuple[LibraryMeal, ...]:
        return tuple(m for m in self.meal_library if m.tag == tag)


@dataclass(frozen=True)
class PlanItem:
    """An entry in the resulting plan: how many grams of an ingredient go
    into a particular slot.

    ``anchor`` is the typical-amount target this ingredient was pulled
    toward (if any), in grams. Reporting only — not a constraint.
    """

    slot: str
    meal: str
    ingredient: str
    grams: float
    anchor: float | None = None


@dataclass(frozen=True)
class Plan:
    items: tuple[PlanItem, ...]
    macro_totals: Mapping[str, float]
    target_deviations: Mapping[str, float]
    slot_meals: Mapping[str, str]
    objective_value: float
    status: str

    def items_by_slot(self) -> dict[str, list[PlanItem]]:
        out: dict[str, list[PlanItem]] = {}
        for item in self.items:
            out.setdefault(item.slot, []).append(item)
        return out
