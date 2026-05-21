"""Combine the daily tracker with the optimizer.

Given what you've actually eaten today (a :class:`DayLog`) and which slots
are still to come, :func:`plan_remaining` returns a :class:`Plan` whose
macro totals account for both already-eaten entries and the optimizer-
chosen remaining meals.

How the math is reduced to the existing optimizer
-------------------------------------------------

Rather than reformulate the LP with fixed variables, we build a *derived*
:class:`Problem` whose

- daily macro targets (``value`` / ``lower`` / ``upper``) are reduced by
  what's already on the log, and
- per-ingredient ``total_min`` / ``total_max`` bounds are reduced by what's
  already been eaten of that ingredient,

then call the regular :func:`optimize`. Soft targets stay soft, so if
you've already eaten more kcal than you should have, the LP will overshoot
the (now-zero) remaining target with a penalty — it won't go infeasible.

What's returned
---------------

A :class:`Plan` containing both logged and optimized items:

- One ``PlanItem`` per log entry with ``meal = "(logged)"`` and the entry's
  slot (or ``"logged"`` if the entry has no slot), so the UI can render
  eaten + planned together.
- The optimizer's items for each slot in ``remaining_day_plan``.
- ``macro_totals`` summed across both groups.
- ``target_deviations`` measured against the **original** daily targets,
  i.e. "did the eaten food plus this plan hit my targets?".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .model import (
    IngredientBound,
    LibraryMeal,
    MacroTarget,
    Plan,
    PlanItem,
    Problem,
)
from .optimizer import optimize
from .tracker import DayLog, compute_totals


# Slot label used for logged entries that have no explicit slot.
_DEFAULT_LOGGED_SLOT = "logged"

# Meal label used on logged PlanItems so the UI can distinguish them.
LOGGED_MEAL_LABEL = "(logged)"

# Prefix for synthetic LibraryMeal names created from CustomSlot specs.
# Exposed so UIs can detect "this is a one-off meal, render it
# differently" without parsing the rest of the name.
CUSTOM_MEAL_PREFIX = "__custom_"

# Sensible default tag when a CustomSlot does not specify one. Slots not
# in this map default to no tag (i.e. no tag-level constraints applied).
_SLOT_TAG_DEFAULTS: dict[str, str] = {
    "breakfast": "breakfast",
    "lunch": "lunch_dinner",
    "dinner": "lunch_dinner",
    "snack": "snack",
}


@dataclass(frozen=True)
class CustomSlot:
    """An ad-hoc meal for one slot: a hand-picked list of ingredients.

    Use this when:

    - You don't want to commit a meal to ``meal_library:`` yet (e.g. a
      one-off sandwich) and just want the optimizer to pick gram amounts
      across some ingredients you have in the fridge.
    - You want to remix a saved meal (e.g. swap broccoli for bell pepper)
      without editing the YAML.

    ``tag`` selects which ``tag_constraints`` apply (kcal cap, macro
    floors, etc.). If ``None``, the planner uses a sensible default based
    on the slot name (``breakfast`` -> ``breakfast``,
    ``lunch``/``dinner`` -> ``lunch_dinner``, ``snack`` -> ``snack``).
    Pass an empty string (``""``) to apply **no** tag constraints at all.

    Notes
    -----

    - Synthetic meals carry no anchors and no ``main`` flags, so the
      optimizer has maximum freedom to choose grams. Per-ingredient
      bounds (``per_meal_min``, ``per_meal_max``, ``step``, daily totals)
      still apply because they are keyed on the ingredient, not the meal.
    - The ``ingredients`` argument accepts any iterable of names; we
      coerce to a tuple for hashability.
    """

    ingredients: tuple[str, ...]
    tag: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ingredients, tuple):
            object.__setattr__(self, "ingredients", tuple(self.ingredients))


def plan_remaining(
    problem: Problem,
    day_log: DayLog,
    remaining_day_plan: Mapping[str, str] | None = None,
    *,
    anchor_weight: float | None = None,
    custom_slots: Mapping[str, CustomSlot] | None = None,
) -> Plan:
    """Plan ``remaining_day_plan`` + ``custom_slots`` given what's in ``day_log``.

    Parameters
    ----------
    problem
        Your full configured ``Problem`` (ingredients, targets, meal
        library, bounds, defaults).
    day_log
        What you've eaten so far today.
    remaining_day_plan
        ``{slot_name: meal_name}`` for slots that should be filled with a
        meal from your saved library. May be empty / ``None``.
    anchor_weight
        Per-call override of ``problem.defaults.anchor_weight``; same
        semantics as :func:`optimize`.
    custom_slots
        ``{slot_name: CustomSlot(...)}`` for slots you want to fill with
        a hand-picked list of ingredients rather than a saved meal. The
        slot names must not overlap with ``remaining_day_plan``.

    Returns
    -------
    A combined :class:`Plan`. Items are ordered: logged entries first
    (in log order), then optimized items. Custom slots are reported with
    their synthetic meal name (prefixed ``__custom_…``); UIs can detect
    these via :data:`CUSTOM_MEAL_PREFIX` and render them as "(custom)".
    """
    saved_plan = dict(remaining_day_plan or {})
    customs = dict(custom_slots or {})

    overlap = set(saved_plan) & set(customs)
    if overlap:
        raise ValueError(
            f"Slot(s) {sorted(overlap)} appear in both `remaining_day_plan` "
            "and `custom_slots`; pick one source per slot."
        )

    # Synthesize a ``LibraryMeal`` for each custom slot and merge it into
    # a working ``Problem`` so the rest of the pipeline (subtract-consumed,
    # optimize) treats custom and saved meals identically.
    extended_problem, full_day_plan = _attach_custom_slots(problem, saved_plan, customs)

    ingredient_map = {i.name: i for i in extended_problem.ingredients}
    consumed_macros = compute_totals(day_log, ingredient_map)
    consumed_grams = _consumed_grams_per_ingredient(day_log, ingredient_map)

    if full_day_plan:
        derived = _problem_minus_consumed(
            extended_problem, consumed_macros, consumed_grams
        )
        plan = optimize(derived, full_day_plan, anchor_weight=anchor_weight)
        if plan.status.startswith("FAILED"):
            # Surface the failure verbatim; caller decides how to display.
            return plan
        optimized_items = plan.items
        optimized_slot_meals = plan.slot_meals
        optimized_totals = plan.macro_totals
        objective = plan.objective_value
        status = plan.status
    else:
        optimized_items = ()
        optimized_slot_meals = {}
        optimized_totals = {}
        objective = 0.0
        status = "OK (nothing left to plan)"

    logged_items = _logged_plan_items(day_log, ingredient_map)

    combined_totals: dict[str, float] = {}
    for k, v in consumed_macros.items():
        combined_totals[k] = combined_totals.get(k, 0.0) + v
    for k, v in optimized_totals.items():
        combined_totals[k] = combined_totals.get(k, 0.0) + v

    # Deviations are measured against the ORIGINAL (un-reduced) targets so
    # the user sees the whole-day picture.
    deviations: dict[str, float] = {}
    for tgt in problem.targets:
        if tgt.value is None:
            continue
        deviations[tgt.name] = combined_totals.get(tgt.name, 0.0) - tgt.value

    return Plan(
        items=logged_items + optimized_items,
        macro_totals=combined_totals,
        target_deviations=deviations,
        slot_meals=optimized_slot_meals,
        objective_value=objective,
        status=status,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _attach_custom_slots(
    problem: Problem,
    saved_slots: dict[str, str],
    customs: dict[str, CustomSlot],
) -> tuple[Problem, dict[str, str]]:
    """Synthesize a ``LibraryMeal`` for each custom slot and return an
    extended ``Problem`` plus a unified ``{slot: meal_name}`` mapping.

    Validation done here so any bad input fails before we touch the LP:
    empty ingredient lists, unknown ingredient names, name collisions
    with existing library meals.
    """
    if not customs:
        return problem, dict(saved_slots)

    known_ings = {ing.name for ing in problem.ingredients}
    existing_meal_names = {m.name for m in problem.meal_library}

    synthetic: list[LibraryMeal] = []
    full_day_plan = dict(saved_slots)

    for slot, spec in customs.items():
        if not spec.ingredients:
            raise ValueError(
                f"Custom slot {slot!r}: ingredient list is empty. "
                "Pick at least one ingredient."
            )
        unknown = [i for i in spec.ingredients if i not in known_ings]
        if unknown:
            raise ValueError(
                f"Custom slot {slot!r}: unknown ingredient(s) {unknown}. "
                "Add them under `ingredients:` in your config first."
            )

        # Tag resolution: explicit empty string disables tag constraints
        # entirely; ``None`` falls back to the slot-name default; anything
        # else is taken verbatim.
        if spec.tag is None:
            tag = _SLOT_TAG_DEFAULTS.get(slot, "")
        else:
            tag = spec.tag

        meal_name = _synthetic_meal_name(slot, existing_meal_names)
        existing_meal_names.add(meal_name)

        # No ``ingredient_specs`` -> no anchors, no ``main`` flags. That
        # matches the intent of an ad-hoc, macro-driven meal: the LP picks
        # whatever grams hit the targets, only respecting per-ingredient
        # bounds and (optionally) the slot's tag constraints.
        synthetic.append(
            LibraryMeal(
                name=meal_name,
                tag=tag,
                ingredients=tuple(spec.ingredients),
            )
        )
        full_day_plan[slot] = meal_name

    extended = replace(
        problem,
        meal_library=problem.meal_library + tuple(synthetic),
    )
    return extended, full_day_plan


def _synthetic_meal_name(slot: str, taken: set[str]) -> str:
    """Pick a synthetic meal name that doesn't collide with anything
    already in the library. We disambiguate with a numeric suffix only
    if needed; the common case is just ``__custom_<slot>``.
    """
    base = f"{CUSTOM_MEAL_PREFIX}{slot}"
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def _consumed_grams_per_ingredient(
    day_log: DayLog, ingredient_map: Mapping[str, object]
) -> dict[str, float]:
    """Sum grams per ingredient across the log, skipping unknowns.

    Unknowns (renamed/removed ingredients) are already excluded from the
    macro totals; skipping them here keeps bound math consistent.
    """
    out: dict[str, float] = {}
    for entry in day_log.entries:
        if entry.ingredient not in ingredient_map:
            continue
        out[entry.ingredient] = out.get(entry.ingredient, 0.0) + entry.grams
    return out


def _logged_plan_items(
    day_log: DayLog, ingredient_map: Mapping[str, object]
) -> tuple[PlanItem, ...]:
    """Promote each log entry to a ``PlanItem`` for combined rendering."""
    return tuple(
        PlanItem(
            slot=entry.slot or _DEFAULT_LOGGED_SLOT,
            meal=LOGGED_MEAL_LABEL,
            ingredient=entry.ingredient,
            grams=float(entry.grams),
        )
        for entry in day_log.entries
        if entry.ingredient in ingredient_map
    )


def _problem_minus_consumed(
    problem: Problem,
    consumed_macros: Mapping[str, float],
    consumed_grams: Mapping[str, float],
) -> Problem:
    """Return a copy of ``problem`` with targets and total bounds reduced."""
    new_targets = tuple(
        _target_minus_consumed(t, consumed_macros.get(t.name, 0.0))
        for t in problem.targets
    )
    new_bounds = dict(problem.bounds)
    for name, eaten in consumed_grams.items():
        old = new_bounds.get(name)
        if old is None:
            continue
        new_bounds[name] = _bound_minus_consumed(old, eaten)
    return replace(problem, targets=new_targets, bounds=new_bounds)


def _target_minus_consumed(t: MacroTarget, eaten: float) -> MacroTarget:
    """Shrink a target by what's been eaten. Never goes negative.

    Soft targets stay soft, so if ``eaten`` already exceeds ``value`` the
    new ``value`` becomes 0 and the LP will overshoot it with the usual
    slack penalty — keeping the problem feasible.
    """
    new_value = max(0.0, t.value - eaten) if t.value is not None else None
    new_lower = max(0.0, t.lower - eaten) if t.lower is not None else None
    new_upper = max(0.0, t.upper - eaten) if t.upper is not None else None
    return replace(t, value=new_value, lower=new_lower, upper=new_upper)


def _bound_minus_consumed(b: IngredientBound, eaten: float) -> IngredientBound:
    """Shrink daily total bounds by what's been eaten. Never negative.

    ``per_meal_*`` and ``step`` are unchanged: they're per-slot rules that
    don't care how much you ate earlier in the day.
    """
    new_total_min = max(0.0, b.total_min - eaten) if b.total_min is not None else None
    new_total_max = max(0.0, b.total_max - eaten) if b.total_max is not None else None
    return replace(b, total_min=new_total_min, total_max=new_total_max)
