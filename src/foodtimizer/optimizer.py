"""Mixed-integer linear-programming optimizer.

You give it a :class:`Problem` (ingredients / library / constraints / targets)
and a ``day_plan`` mapping ``{slot_name: meal_name}``. It returns a
:class:`Plan` with the gram amount of every ingredient in every chosen meal.

Math
----

For each (slot ``s``, ingredient ``j``) pair in the chosen meals there is a
non-negative decision variable. Its meaning depends on whether ``j`` has a
``step`` bound:

- Continuous (no ``step``):  ``x[s, j]`` is grams.
- Stepped (``step = q``):    ``n[s, j]`` is an **integer** count of ``q``-gram
                             units; grams = ``q · n[s, j]``.

A single per-column **scale factor** lets us treat both uniformly:
``scale[s, j] = q`` (stepped) or ``1`` (continuous), and the actual gram
contribution of column ``c`` is ``scale[c] · x[c]``.

For each *soft* macro target ``m`` we add non-negative continuous slack
variables ``s_m^+`` and ``s_m^-`` and minimize

    Σ_m  w_m · (s_m^+ + s_m^-)

subject to

- Macro identities:
      ``Σ_{s,j} scale[s,j] · x[s,j] · macro_m(j)/100 − s_m^+ + s_m^- = T_m``
- Per-slot kcal cap (meal's own ``kcal_max`` if set, else from the tag):
      ``Σ_j scale[s,j] · x[s,j] · kcal(j)/100 ≤ kcal_cap(s)``
- Optional macro `lower` / `upper` hard bounds.
- Optional per-ingredient gram bounds (per-slot and totals), all in *grams*
  so they get rescaled into the variable's own units automatically.

Solver: HiGHS via :func:`scipy.optimize.milp`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .model import (
    Ingredient,
    IngredientBound,
    LibraryMeal,
    Plan,
    PlanItem,
    Problem,
    TagConstraints,
)


@dataclass(frozen=True)
class _VarLayout:
    """LP variable layout: continuous/integer gram-or-step vars, then slack pairs."""

    x_index: dict[tuple[str, str], int]  # (slot, ingredient) -> column
    slack_index: dict[str, tuple[int, int]]  # macro -> (plus_col, minus_col)
    # (slot, ingredient) -> (plus_col, minus_col) for anchor L1 slacks.
    anchor_slack_index: dict[tuple[str, str], tuple[int, int]]
    # (slot, ingredient) -> (anchor_grams, weight) for plan reporting.
    anchors: dict[tuple[str, str], tuple[float, float]]
    n_vars: int
    scale: np.ndarray  # length n_vars, multiplies coefficients touching that column
    integrality: np.ndarray  # length n_vars, 1=integer, 0=continuous


def optimize(
    problem: Problem,
    day_plan: Mapping[str, str] | None = None,
    *,
    anchor_weight: float | None = None,
) -> Plan:
    """Solve the MILP for the given day plan.

    If ``day_plan`` is None, falls back to ``problem.default_day``.

    ``anchor_weight``, if given, overrides ``problem.defaults.anchor_weight``
    for this call. Pass ``0`` to disable anchor penalties entirely without
    editing the config.
    """
    day = dict(day_plan) if day_plan is not None else dict(problem.default_day)
    if not day:
        raise ValueError(
            "No day plan provided and no `default_day` in config. "
            "Pass {slot: meal_name} or define a `day:` section in the YAML."
        )

    effective_anchor_weight = (
        anchor_weight if anchor_weight is not None else problem.defaults.anchor_weight
    )

    slot_meals = _resolve_slot_meals(problem, day)
    layout = _build_layout(problem, slot_meals, effective_anchor_weight)

    c = _build_objective(problem, layout, effective_anchor_weight)
    A_rows, lb_rows, ub_rows = _build_linear_constraints(problem, slot_meals, layout)
    var_lb, var_ub = _build_variable_bounds(problem, slot_meals, layout)

    constraints = (
        LinearConstraint(A_rows, lb_rows, ub_rows) if A_rows.size else None
    )

    res = milp(
        c=c,
        constraints=constraints,
        integrality=layout.integrality,
        bounds=Bounds(lb=var_lb, ub=var_ub),
    )

    if not res.success or res.x is None:
        return Plan(
            items=(),
            macro_totals={},
            target_deviations={},
            slot_meals={s: m.name for s, m in slot_meals.items()},
            objective_value=float("nan"),
            status=f"FAILED: {res.message}",
        )

    return _build_plan(problem, slot_meals, layout, np.asarray(res.x), res.fun, res.message)


def _resolve_slot_meals(
    problem: Problem, day: Mapping[str, str]
) -> dict[str, LibraryMeal]:
    out: dict[str, LibraryMeal] = {}
    for slot, meal_name in day.items():
        try:
            out[slot] = problem.meal_by_name(meal_name)
        except KeyError as e:
            available = sorted(m.name for m in problem.meal_library)
            raise ValueError(
                f"Slot {slot!r}: meal {meal_name!r} not in library. "
                f"Available: {available}"
            ) from e
    return out


def _slot_macro_max(
    meal: LibraryMeal,
    tag_constraints: Mapping[str, TagConstraints],
    macro: str,
) -> float | None:
    """Return the upper bound for `macro` on this slot, or None.

    Per-meal `macro_max` overrides the tag-level `macro_max` per-macro.
    """
    if macro in meal.macro_max:
        return meal.macro_max[macro]
    tc = tag_constraints.get(meal.tag)
    if tc and macro in tc.macro_max:
        return tc.macro_max[macro]
    return None


def _slot_macro_min(
    meal: LibraryMeal,
    tag_constraints: Mapping[str, TagConstraints],
    macro: str,
) -> float | None:
    if macro in meal.macro_min:
        return meal.macro_min[macro]
    tc = tag_constraints.get(meal.tag)
    if tc and macro in tc.macro_min:
        return tc.macro_min[macro]
    return None


def _slot_constrained_macros(
    meal: LibraryMeal, tag_constraints: Mapping[str, TagConstraints]
) -> set[str]:
    """Set of macros that have any per-slot bound at this meal."""
    macros = set(meal.macro_min) | set(meal.macro_max)
    tc = tag_constraints.get(meal.tag)
    if tc:
        macros |= set(tc.macro_min) | set(tc.macro_max)
    return macros


def _slot_kcal_cap(
    meal: LibraryMeal, tag_constraints: Mapping[str, TagConstraints]
) -> float | None:
    """Convenience wrapper used by the CLI's header rendering."""
    return _slot_macro_max(meal, tag_constraints, "kcal")


def _build_layout(
    problem: Problem,
    slot_meals: dict[str, LibraryMeal],
    anchor_weight: float | None,
) -> _VarLayout:
    bounds = problem.bounds
    x_index: dict[tuple[str, str], int] = {}
    scales: list[float] = []
    integrality: list[int] = []

    for slot, meal in slot_meals.items():
        for ing in meal.ingredients:
            x_index[(slot, ing)] = len(x_index)
            b = bounds.get(ing)
            if b is not None and b.step is not None:
                scales.append(float(b.step))
                integrality.append(1)
            else:
                scales.append(1.0)
                integrality.append(0)

    slack_index: dict[str, tuple[int, int]] = {}
    n = len(x_index)
    for tgt in problem.targets:
        if tgt.value is not None and not tgt.hard:
            slack_index[tgt.name] = (n, n + 1)
            scales.extend([1.0, 1.0])
            integrality.extend([0, 0])
            n += 2

    # Anchor L1 slacks: one pair per (slot, ingredient) whose meal spec
    # declares an anchor (and only when anchor_weight is enabled).
    anchor_slack_index: dict[tuple[str, str], tuple[int, int]] = {}
    anchors: dict[tuple[str, str], tuple[float, float]] = {}
    if anchor_weight is not None and anchor_weight > 0:
        for slot, meal in slot_meals.items():
            for ing_name in meal.ingredients:
                spec = meal.ingredient_specs.get(ing_name)
                if spec is None or spec.anchor is None:
                    continue
                anchor_slack_index[(slot, ing_name)] = (n, n + 1)
                anchors[(slot, ing_name)] = (float(spec.anchor), float(anchor_weight))
                scales.extend([1.0, 1.0])
                integrality.extend([0, 0])
                n += 2

    return _VarLayout(
        x_index=x_index,
        slack_index=slack_index,
        anchor_slack_index=anchor_slack_index,
        anchors=anchors,
        n_vars=n,
        scale=np.array(scales, dtype=float),
        integrality=np.array(integrality, dtype=int),
    )


def _build_objective(
    problem: Problem, layout: _VarLayout, anchor_weight: float | None
) -> np.ndarray:
    c = np.zeros(layout.n_vars)
    for tgt in problem.targets:
        if tgt.name in layout.slack_index:
            plus, minus = layout.slack_index[tgt.name]
            c[plus] = tgt.weight
            c[minus] = tgt.weight
    if anchor_weight is not None and anchor_weight > 0:
        for key, (plus, minus) in layout.anchor_slack_index.items():
            c[plus] = anchor_weight
            c[minus] = anchor_weight
    return c


def _ingredient_map(problem: Problem) -> dict[str, Ingredient]:
    return {ing.name: ing for ing in problem.ingredients}


def _build_linear_constraints(
    problem: Problem,
    slot_meals: dict[str, LibraryMeal],
    layout: _VarLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a single LinearConstraint(A, lb, ub) matrix covering everything."""
    ings = _ingredient_map(problem)
    scale = layout.scale
    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def _empty_row() -> np.ndarray:
        return np.zeros(layout.n_vars)

    for tgt in problem.targets:
        if tgt.value is not None:
            row = _empty_row()
            for (slot, ing_name), col in layout.x_index.items():
                row[col] = scale[col] * ings[ing_name].amount_for(tgt.name) / 100.0
            if tgt.hard:
                rows.append(row)
                lbs.append(tgt.value)
                ubs.append(tgt.value)
            else:
                plus, minus = layout.slack_index[tgt.name]
                row[plus] = -1.0
                row[minus] = 1.0
                rows.append(row)
                lbs.append(tgt.value)
                ubs.append(tgt.value)

        if tgt.upper is not None:
            row = _empty_row()
            for (slot, ing_name), col in layout.x_index.items():
                row[col] = scale[col] * ings[ing_name].amount_for(tgt.name) / 100.0
            rows.append(row)
            lbs.append(-np.inf)
            ubs.append(tgt.upper)
        if tgt.lower is not None:
            row = _empty_row()
            for (slot, ing_name), col in layout.x_index.items():
                row[col] = scale[col] * ings[ing_name].amount_for(tgt.name) / 100.0
            rows.append(row)
            lbs.append(tgt.lower)
            ubs.append(np.inf)

    for slot, meal in slot_meals.items():
        for macro in _slot_constrained_macros(meal, problem.tag_constraints):
            cap = _slot_macro_max(meal, problem.tag_constraints, macro)
            floor = _slot_macro_min(meal, problem.tag_constraints, macro)
            if cap is None and floor is None:
                continue
            row = _empty_row()
            for ing_name in meal.ingredients:
                col = layout.x_index[(slot, ing_name)]
                row[col] = scale[col] * ings[ing_name].amount_for(macro) / 100.0
            rows.append(row)
            lbs.append(floor if floor is not None else -np.inf)
            ubs.append(cap if cap is not None else np.inf)

    for ing_name, bound in problem.bounds.items():
        cols = [
            layout.x_index[(slot, ing_name)]
            for slot in slot_meals
            if (slot, ing_name) in layout.x_index
        ]
        if not cols:
            continue
        if bound.total_max is not None:
            row = _empty_row()
            for col in cols:
                row[col] = scale[col]
            rows.append(row)
            lbs.append(-np.inf)
            ubs.append(bound.total_max)
        if bound.total_min is not None:
            row = _empty_row()
            for col in cols:
                row[col] = scale[col]
            rows.append(row)
            lbs.append(bound.total_min)
            ubs.append(np.inf)

    # Anchor L1 equalities:  scale[col] * x[col] - s_plus + s_minus = anchor_g.
    # Penalizing s_plus + s_minus in the objective then drives the variable
    # toward the anchor (typical recipe amount).
    for (slot, ing_name), (plus, minus) in layout.anchor_slack_index.items():
        col = layout.x_index[(slot, ing_name)]
        anchor_g, _w = layout.anchors[(slot, ing_name)]
        row = _empty_row()
        row[col] = scale[col]
        row[plus] = -1.0
        row[minus] = 1.0
        rows.append(row)
        lbs.append(anchor_g)
        ubs.append(anchor_g)

    if not rows:
        return np.empty((0, layout.n_vars)), np.empty((0,)), np.empty((0,))
    return np.vstack(rows), np.array(lbs), np.array(ubs)


def _build_variable_bounds(
    problem: Problem,
    slot_meals: dict[str, LibraryMeal],
    layout: _VarLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve per-column ``(lb, ub)`` from defaults + global ingredient
    bounds + per-meal ingredient specs.

    The effective minimum is the **max** of every applicable lower bound:
    ``defaults.per_meal_min``, ``ingredient_bounds.per_meal_min``,
    ``meal.ingredient_specs[ing].min``, and ``defaults.main_min`` when
    ``meal.ingredient_specs[ing].main`` is true. Labelling something as
    "main" therefore only ever *tightens* a constraint.

    The effective maximum is the **min** of every applicable upper bound.

    If the resulting min exceeds the max (e.g. ``main_min`` of 80 g vs a
    hard pin at 60 g), we silently cap min at max to keep the LP feasible.
    Bounds are user-supplied in grams; we divide by the column's scale so
    stepped (integer) variables use the same gram semantics.
    """
    lb = np.zeros(layout.n_vars)
    ub = np.full(layout.n_vars, np.inf)

    for (slot, ing_name), col in layout.x_index.items():
        s = layout.scale[col]
        meal = slot_meals[slot]
        eff_min = _resolve_meal_min(meal, ing_name, problem)
        eff_max = _resolve_meal_max(meal, ing_name, problem)
        if eff_max is not None and eff_min is not None and eff_min > eff_max:
            eff_min = eff_max
        if eff_min is not None:
            lb[col] = eff_min / s
        if eff_max is not None:
            ub[col] = eff_max / s
    return lb, ub


def _resolve_meal_min(
    meal: LibraryMeal, ing_name: str, problem: Problem
) -> float | None:
    """Maximum of every applicable lower bound (max wins -> tightest constraint)."""
    candidates: list[float] = []
    if problem.defaults.per_meal_min is not None:
        candidates.append(problem.defaults.per_meal_min)
    ib = problem.bounds.get(ing_name)
    if ib is not None and ib.per_meal_min is not None:
        candidates.append(ib.per_meal_min)
    spec = meal.ingredient_specs.get(ing_name)
    if spec is not None:
        if spec.min is not None:
            candidates.append(spec.min)
        if spec.main and problem.defaults.main_min is not None:
            candidates.append(problem.defaults.main_min)
    return max(candidates) if candidates else None


def _resolve_meal_max(
    meal: LibraryMeal, ing_name: str, problem: Problem
) -> float | None:
    """Minimum of every applicable upper bound (min wins -> tightest constraint)."""
    candidates: list[float] = []
    if problem.defaults.per_meal_max is not None:
        candidates.append(problem.defaults.per_meal_max)
    ib = problem.bounds.get(ing_name)
    if ib is not None and ib.per_meal_max is not None:
        candidates.append(ib.per_meal_max)
    spec = meal.ingredient_specs.get(ing_name)
    if spec is not None and spec.max is not None:
        candidates.append(spec.max)
    return min(candidates) if candidates else None


def _build_plan(
    problem: Problem,
    slot_meals: dict[str, LibraryMeal],
    layout: _VarLayout,
    x: np.ndarray,
    objective_value: float,
    status: str,
) -> Plan:
    ings = _ingredient_map(problem)
    items: list[PlanItem] = []
    EPS = 1e-6
    for (slot, ing_name), col in layout.x_index.items():
        grams = float(x[col] * layout.scale[col])
        if grams > EPS:
            anchor_info = layout.anchors.get((slot, ing_name))
            items.append(
                PlanItem(
                    slot=slot,
                    meal=slot_meals[slot].name,
                    ingredient=ing_name,
                    grams=grams,
                    anchor=anchor_info[0] if anchor_info else None,
                )
            )

    macro_keys = sorted({k for ing in problem.ingredients for k in ing.macros})
    totals: dict[str, float] = {m: 0.0 for m in macro_keys}
    for item in items:
        for m in macro_keys:
            totals[m] += item.grams * ings[item.ingredient].amount_for(m) / 100.0

    deviations: dict[str, float] = {}
    for tgt in problem.targets:
        if tgt.value is None:
            continue
        deviations[tgt.name] = totals.get(tgt.name, 0.0) - tgt.value

    return Plan(
        items=tuple(items),
        macro_totals=totals,
        target_deviations=deviations,
        slot_meals={s: m.name for s, m in slot_meals.items()},
        objective_value=float(objective_value),
        status=status,
    )
