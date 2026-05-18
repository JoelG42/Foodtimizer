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

from dataclasses import replace
from typing import Mapping

from .model import (
    IngredientBound,
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


def plan_remaining(
    problem: Problem,
    day_log: DayLog,
    remaining_day_plan: Mapping[str, str],
    *,
    anchor_weight: float | None = None,
) -> Plan:
    """Plan ``remaining_day_plan`` given what's already in ``day_log``.

    Parameters
    ----------
    problem
        Your full configured ``Problem`` (ingredients, targets, meal
        library, bounds, defaults).
    day_log
        What you've eaten so far today.
    remaining_day_plan
        ``{slot_name: meal_name}`` for the slots you still want the
        optimizer to fill. Can be empty (then the result is just a
        report of what you've eaten vs your targets).
    anchor_weight
        Per-call override of ``problem.defaults.anchor_weight``; same
        semantics as :func:`optimize`.

    Returns
    -------
    A combined :class:`Plan`. Items are ordered: logged entries first
    (in log order), then optimized items.
    """
    ingredient_map = {i.name: i for i in problem.ingredients}
    consumed_macros = compute_totals(day_log, ingredient_map)
    consumed_grams = _consumed_grams_per_ingredient(day_log, ingredient_map)

    if remaining_day_plan:
        derived = _problem_minus_consumed(problem, consumed_macros, consumed_grams)
        plan = optimize(derived, remaining_day_plan, anchor_weight=anchor_weight)
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
