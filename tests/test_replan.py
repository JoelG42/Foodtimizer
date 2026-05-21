"""Tests for combining the tracker with the optimizer (``plan_remaining``).

These exercise the *math* of subtracting consumed macros / grams from the
problem before optimizing, not the Streamlit UI.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from foodtimizer import (
    CustomSlot,
    DayLog,
    Ingredient,
    IngredientBound,
    LibraryMeal,
    LogEntry,
    MacroTarget,
    MealIngredient,
    Problem,
    TagConstraints,
    plan_remaining,
)
from foodtimizer.replan import (
    CUSTOM_MEAL_PREFIX,
    LOGGED_MEAL_LABEL,
    diagnose_infeasibility,
)


def _basic_problem() -> Problem:
    """Two ingredients, two soft macro targets, two interchangeable meals."""
    chicken = Ingredient(
        name="chicken", macros={"kcal": 100.0, "protein": 20.0, "carbs": 0.0}
    )
    rice = Ingredient(
        name="rice", macros={"kcal": 350.0, "protein": 9.0, "carbs": 77.0}
    )
    meal = LibraryMeal(
        name="ck", tag="lunch_dinner", ingredients=("chicken", "rice")
    )
    return Problem(
        ingredients=(chicken, rice),
        targets=(
            MacroTarget(name="kcal", value=600.0, weight=1.0),
            MacroTarget(name="protein", value=40.0, weight=100.0),
        ),
        meal_library=(meal,),
    )


def test_consumed_reduces_macro_target():
    """If you already ate 300 kcal toward a 600-kcal goal, the optimizer
    should plan ~300 kcal of remaining food, not 600."""
    problem = _basic_problem()
    log = DayLog(log_date=date(2026, 5, 18)).with_added(
        # chicken @ 100 kcal/100g -> 300 g = 300 kcal, 60 g protein
        LogEntry(ingredient="chicken", grams=300.0, slot="breakfast")
    )
    plan = plan_remaining(problem, log, {"lunch": "ck"})

    # macro totals = logged + optimized; should land near the original 600 kcal
    assert math.isclose(plan.macro_totals["kcal"], 600.0, abs_tol=1.0)

    # And the *optimized* portion only (everything not labelled (logged))
    # should contribute ~300 kcal.
    optimized_kcal = sum(
        it.grams * problem.ingredient_by_name(it.ingredient).amount_for("kcal") / 100.0
        for it in plan.items
        if it.meal != LOGGED_MEAL_LABEL
    )
    assert math.isclose(optimized_kcal, 300.0, abs_tol=1.0)


def test_logged_items_are_returned_with_special_meal_label():
    """The combined plan exposes logged entries as PlanItems so a UI
    can render eaten + planned together."""
    problem = _basic_problem()
    log = (
        DayLog(log_date=date(2026, 5, 18))
        .with_added(LogEntry(ingredient="rice", grams=100, slot="breakfast"))
        .with_added(LogEntry(ingredient="chicken", grams=150))  # no slot
    )
    plan = plan_remaining(problem, log, {"lunch": "ck"})

    logged = [it for it in plan.items if it.meal == LOGGED_MEAL_LABEL]
    optimized = [it for it in plan.items if it.meal != LOGGED_MEAL_LABEL]

    assert len(logged) == 2
    # The first logged item has its original slot, the second falls back
    # to the generic "logged" slot.
    assert logged[0].slot == "breakfast"
    assert logged[0].ingredient == "rice"
    assert logged[0].grams == 100
    assert logged[1].slot == "logged"
    assert logged[1].ingredient == "chicken"
    assert logged[1].grams == 150

    # The optimized half exists separately.
    assert optimized
    assert all(it.slot == "lunch" for it in optimized)


def test_no_remaining_slots_just_reports_logged_totals():
    """Calling with an empty plan returns a no-op result that still
    reports what's been eaten and how it compares to targets."""
    problem = _basic_problem()
    log = DayLog(log_date=date(2026, 5, 18)).with_added(
        LogEntry(ingredient="chicken", grams=200)
    )
    plan = plan_remaining(problem, log, {})

    # No optimization happened: status is informational, no items beyond logged.
    assert plan.status.startswith("OK")
    assert {it.meal for it in plan.items} == {LOGGED_MEAL_LABEL}

    # 200 g chicken -> 200 kcal, 40 g protein
    assert math.isclose(plan.macro_totals["kcal"], 200.0)
    assert math.isclose(plan.macro_totals["protein"], 40.0)
    # Deviation vs the ORIGINAL 600/40 targets.
    assert math.isclose(plan.target_deviations["kcal"], -400.0)
    assert math.isclose(plan.target_deviations["protein"], 0.0)


def test_overshoot_keeps_problem_feasible():
    """Already exceeding a soft target shouldn't make the LP infeasible.

    The remaining target is clamped to 0 and the optimizer pays a slack
    penalty for unavoidable overshoot from per-meal minima.
    """
    problem = _basic_problem()
    log = DayLog(log_date=date(2026, 5, 18)).with_added(
        # 800 g chicken -> 800 kcal (already over the 600 target)
        LogEntry(ingredient="chicken", grams=800)
    )
    plan = plan_remaining(problem, log, {"lunch": "ck"})

    assert not plan.status.startswith("FAILED")
    # Combined kcal is at least what we ate.
    assert plan.macro_totals["kcal"] >= 800.0 - 1e-6


def test_total_max_shrinks_by_consumed_grams():
    """A daily ``total_max`` cap should account for what's already eaten:
    eating 50 g whey before lunch must reduce the cap available to the
    remaining slots by 50 g.
    """
    whey = Ingredient(name="whey", macros={"kcal": 400.0, "protein": 75.0})
    rice = Ingredient(name="rice", macros={"kcal": 350.0, "carbs": 77.0})
    meal = LibraryMeal(name="m", tag="lunch_dinner", ingredients=("whey", "rice"))
    problem = Problem(
        ingredients=(whey, rice),
        targets=(MacroTarget(name="protein", value=100.0, weight=100.0),),
        meal_library=(meal,),
        bounds={"whey": IngredientBound(total_max=60.0)},
    )

    log = DayLog(log_date=date(2026, 5, 18)).with_added(
        LogEntry(ingredient="whey", grams=50.0, slot="breakfast")
    )
    plan = plan_remaining(problem, log, {"lunch": "m"})

    optimized_whey = sum(
        it.grams for it in plan.items
        if it.ingredient == "whey" and it.meal != LOGGED_MEAL_LABEL
    )
    # Only 10 g of headroom remain in the daily cap.
    assert optimized_whey <= 10.0 + 1e-6

    # And the combined whey across logged + optimized respects the original cap.
    total_whey = sum(it.grams for it in plan.items if it.ingredient == "whey")
    assert total_whey <= 60.0 + 1e-6


def test_total_max_already_exceeded_blocks_more():
    """If you've blown past total_max, the LP must not add any more."""
    whey = Ingredient(name="whey", macros={"kcal": 400.0, "protein": 75.0})
    rice = Ingredient(name="rice", macros={"kcal": 350.0, "carbs": 77.0})
    meal = LibraryMeal(name="m", tag="lunch_dinner", ingredients=("whey", "rice"))
    problem = Problem(
        ingredients=(whey, rice),
        targets=(MacroTarget(name="protein", value=200.0, weight=100.0),),
        meal_library=(meal,),
        bounds={"whey": IngredientBound(total_max=60.0)},
    )
    log = DayLog(log_date=date(2026, 5, 18)).with_added(
        LogEntry(ingredient="whey", grams=80.0)
    )
    plan = plan_remaining(problem, log, {"lunch": "m"})

    optimized_whey = sum(
        it.grams for it in plan.items
        if it.ingredient == "whey" and it.meal != LOGGED_MEAL_LABEL
    )
    assert optimized_whey == pytest.approx(0.0, abs=1e-6)


def test_total_min_credits_already_eaten():
    """A daily ``total_min`` should also shrink: if your floor is 200 g
    of rice/day and you've eaten 150 g, the LP only needs 50 g more.
    """
    rice = Ingredient(name="rice", macros={"kcal": 350.0, "carbs": 77.0})
    meal = LibraryMeal(name="m", tag="lunch_dinner", ingredients=("rice",))
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=100.0, weight=1.0),),
        meal_library=(meal,),
        bounds={"rice": IngredientBound(total_min=200.0)},
    )
    log = DayLog(log_date=date(2026, 5, 18)).with_added(
        LogEntry(ingredient="rice", grams=150.0, slot="breakfast")
    )
    plan = plan_remaining(problem, log, {"lunch": "m"})

    optimized_rice = sum(
        it.grams for it in plan.items
        if it.ingredient == "rice" and it.meal != LOGGED_MEAL_LABEL
    )
    # At least 50 g more to satisfy the daily floor.
    assert optimized_rice >= 50.0 - 1e-6


def test_unknown_ingredient_in_log_is_excluded_from_math_and_items():
    """Stale log entries referencing a renamed/deleted ingredient are
    skipped (mirrors what ``compute_totals`` already does)."""
    problem = _basic_problem()
    log = (
        DayLog(log_date=date(2026, 5, 18))
        .with_added(LogEntry(ingredient="chicken", grams=100))
        .with_added(LogEntry(ingredient="cosmic_ray", grams=999))
    )
    plan = plan_remaining(problem, log, {"lunch": "ck"})

    # Only the known logged item appears in the combined plan.
    logged_names = [it.ingredient for it in plan.items if it.meal == LOGGED_MEAL_LABEL]
    assert logged_names == ["chicken"]

    # And the macro totals don't include cosmic_ray's bogus contribution.
    # 100 g chicken alone = 100 kcal; the optimizer fills the rest.
    optimized_kcal = sum(
        it.grams * problem.ingredient_by_name(it.ingredient).amount_for("kcal") / 100.0
        for it in plan.items
        if it.meal != LOGGED_MEAL_LABEL
    )
    assert math.isclose(optimized_kcal, 500.0, abs_tol=1.0)


def test_original_targets_used_for_deviation_report():
    """Deviations should be measured against the user's *full-day* targets,
    not the reduced ones — that's what they care about.
    """
    problem = _basic_problem()  # targets: 600 kcal, 40 g protein
    log = DayLog(log_date=date(2026, 5, 18)).with_added(
        LogEntry(ingredient="chicken", grams=200)  # 200 kcal, 40 g protein
    )
    plan = plan_remaining(problem, log, {"lunch": "ck"})

    # Protein is already met by the log; combined should be on target.
    assert math.isclose(plan.target_deviations["protein"], 0.0, abs_tol=0.5)
    # kcal: combined should hit the FULL 600 target, deviation ~0.
    assert math.isclose(plan.target_deviations["kcal"], 0.0, abs_tol=1.0)


def test_original_problem_is_not_mutated():
    """`replace`-based copying must leave the input ``Problem`` untouched."""
    problem = _basic_problem()
    original_targets = problem.targets
    original_bounds = dict(problem.bounds)

    log = DayLog(log_date=date(2026, 5, 18)).with_added(
        LogEntry(ingredient="chicken", grams=300)
    )
    plan_remaining(problem, log, {"lunch": "ck"})

    assert problem.targets is original_targets
    assert dict(problem.bounds) == original_bounds


def test_custom_slot_only_no_saved_meal():
    """A custom slot alone (no saved meals) plans an ad-hoc 'meal' that
    contains exactly the ingredients you picked."""
    chicken = Ingredient(name="chicken", macros={"kcal": 100.0, "protein": 20.0})
    rice = Ingredient(name="rice", macros={"kcal": 350.0, "protein": 9.0, "carbs": 77.0})
    problem = Problem(
        ingredients=(chicken, rice),
        targets=(
            MacroTarget(name="kcal", value=500.0, weight=1.0),
            MacroTarget(name="protein", value=40.0, weight=100.0),
        ),
        meal_library=(),
    )

    plan = plan_remaining(
        problem,
        DayLog(log_date=date(2026, 5, 21)),
        remaining_day_plan=None,
        custom_slots={"snack": CustomSlot(ingredients=("chicken", "rice"))},
    )

    assert not plan.status.startswith("FAILED")
    # Slot is in slot_meals, and the meal name is the synthetic one.
    assert "snack" in plan.slot_meals
    assert plan.slot_meals["snack"].startswith(CUSTOM_MEAL_PREFIX)
    # And the optimized items only contain the chosen ingredients.
    ingredients = {it.ingredient for it in plan.items}
    assert ingredients == {"chicken", "rice"}


def test_custom_slot_mixed_with_saved_meal():
    """You can use a saved meal for one slot and a custom slot for another
    in the same call."""
    chicken = Ingredient(name="chicken", macros={"kcal": 100.0, "protein": 20.0})
    rice = Ingredient(name="rice", macros={"kcal": 350.0, "protein": 9.0, "carbs": 77.0})
    bread = Ingredient(name="bread", macros={"kcal": 250.0, "protein": 9.0, "carbs": 49.0})

    meal = LibraryMeal(name="ck", tag="lunch_dinner", ingredients=("chicken", "rice"))
    problem = Problem(
        ingredients=(chicken, rice, bread),
        targets=(MacroTarget(name="kcal", value=900.0, weight=1.0),),
        meal_library=(meal,),
    )

    plan = plan_remaining(
        problem,
        DayLog(log_date=date(2026, 5, 21)),
        remaining_day_plan={"lunch": "ck"},
        custom_slots={"snack": CustomSlot(ingredients=("bread", "chicken"))},
    )

    assert not plan.status.startswith("FAILED")
    assert plan.slot_meals["lunch"] == "ck"
    assert plan.slot_meals["snack"].startswith(CUSTOM_MEAL_PREFIX)

    lunch_ings = {it.ingredient for it in plan.items if it.slot == "lunch"}
    snack_ings = {it.ingredient for it in plan.items if it.slot == "snack"}
    assert lunch_ings.issubset({"chicken", "rice"})
    assert snack_ings.issubset({"bread", "chicken"})


def test_custom_slot_tag_inferred_from_slot_name():
    """A snack-slot custom meal picks up the ``snack`` tag's kcal cap."""
    rice = Ingredient(name="rice", macros={"kcal": 350.0, "carbs": 77.0})
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=1000.0, weight=1.0),),
        meal_library=(),
        tag_constraints={"snack": TagConstraints(macro_max={"kcal": 200.0})},
    )

    plan = plan_remaining(
        problem,
        DayLog(log_date=date(2026, 5, 21)),
        custom_slots={"snack": CustomSlot(ingredients=("rice",))},
    )
    # Snack tag caps kcal at 200; even though the daily target is 1000,
    # the snack alone can't exceed 200 kcal.
    snack_kcal = sum(
        it.grams * rice.amount_for("kcal") / 100.0
        for it in plan.items
        if it.slot == "snack"
    )
    assert snack_kcal <= 200.0 + 1e-6


def test_custom_slot_explicit_tag_overrides_inference():
    """Passing an explicit tag (or empty string for 'no tag') wins over
    the slot-name default."""
    rice = Ingredient(name="rice", macros={"kcal": 350.0, "carbs": 77.0})
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=1000.0, weight=1.0),),
        meal_library=(),
        # Snack tag would cap at 200, but we override with "" (no tag).
        tag_constraints={"snack": TagConstraints(macro_max={"kcal": 200.0})},
    )

    plan = plan_remaining(
        problem,
        DayLog(log_date=date(2026, 5, 21)),
        custom_slots={"snack": CustomSlot(ingredients=("rice",), tag="")},
    )
    snack_kcal = sum(
        it.grams * rice.amount_for("kcal") / 100.0
        for it in plan.items
        if it.slot == "snack"
    )
    # With tag overridden to "", the snack cap doesn't apply.
    assert snack_kcal > 200.0


def test_custom_slot_rejects_unknown_ingredient():
    """A bad ingredient name should error before the LP runs."""
    rice = Ingredient(name="rice", macros={"kcal": 350.0})
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=200.0, weight=1.0),),
        meal_library=(),
    )
    with pytest.raises(ValueError, match=r"unknown ingredient"):
        plan_remaining(
            problem,
            DayLog(log_date=date(2026, 5, 21)),
            custom_slots={
                "snack": CustomSlot(ingredients=("rice", "moondust")),
            },
        )


def test_custom_slot_rejects_empty_ingredients():
    rice = Ingredient(name="rice", macros={"kcal": 350.0})
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=200.0, weight=1.0),),
        meal_library=(),
    )
    with pytest.raises(ValueError, match=r"empty"):
        plan_remaining(
            problem,
            DayLog(log_date=date(2026, 5, 21)),
            custom_slots={"snack": CustomSlot(ingredients=())},
        )


def test_custom_slot_overlap_with_saved_plan_errors():
    """The same slot in both maps is ambiguous and must error."""
    rice = Ingredient(name="rice", macros={"kcal": 350.0})
    meal = LibraryMeal(name="m", tag="snack", ingredients=("rice",))
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=200.0, weight=1.0),),
        meal_library=(meal,),
    )
    with pytest.raises(ValueError, match=r"both"):
        plan_remaining(
            problem,
            DayLog(log_date=date(2026, 5, 21)),
            remaining_day_plan={"snack": "m"},
            custom_slots={"snack": CustomSlot(ingredients=("rice",))},
        )


def test_custom_slot_accepts_list_or_tuple():
    """``CustomSlot.ingredients`` may be any iterable; we coerce to tuple."""
    spec_from_list = CustomSlot(ingredients=["a", "b"])
    spec_from_tuple = CustomSlot(ingredients=("a", "b"))
    assert spec_from_list.ingredients == ("a", "b")
    assert spec_from_tuple.ingredients == ("a", "b")
    # Frozen dataclass: equal specs hash the same -> usable as dict keys.
    assert spec_from_list == spec_from_tuple
    assert hash(spec_from_list) == hash(spec_from_tuple)


def test_custom_slot_does_not_mutate_input_problem():
    """The synthetic meal is added to a copy; the caller's Problem
    keeps its original ``meal_library``."""
    rice = Ingredient(name="rice", macros={"kcal": 350.0})
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=200.0, weight=1.0),),
        meal_library=(),
    )
    original_library = problem.meal_library

    plan_remaining(
        problem,
        DayLog(log_date=date(2026, 5, 21)),
        custom_slots={"snack": CustomSlot(ingredients=("rice",))},
    )
    assert problem.meal_library is original_library
    assert problem.meal_library == ()


def test_per_meal_bounds_are_unchanged_by_consumed():
    """``per_meal_*`` bounds are per-slot facts and must NOT shrink based
    on what was eaten earlier in the day."""
    rice = Ingredient(name="rice", macros={"kcal": 350.0, "carbs": 77.0})
    meal = LibraryMeal(
        name="m",
        tag="lunch_dinner",
        ingredients=("rice",),
        ingredient_specs={"rice": MealIngredient(main=True)},
    )
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=1000.0, weight=1.0),),
        meal_library=(meal,),
        tag_constraints={"lunch_dinner": TagConstraints(macro_max={"kcal": 800.0})},
        bounds={"rice": IngredientBound(per_meal_max=150.0)},
    )

    log = DayLog(log_date=date(2026, 5, 18)).with_added(
        LogEntry(ingredient="rice", grams=100)
    )
    plan = plan_remaining(problem, log, {"lunch": "m"})

    optimized_rice = sum(
        it.grams for it in plan.items
        if it.ingredient == "rice" and it.meal != LOGGED_MEAL_LABEL
    )
    # Per-meal cap of 150 g still applies, irrespective of the 100 g already eaten.
    assert optimized_rice <= 150.0 + 1e-6


# ---------------------------------------------------------------------------
# diagnose_infeasibility
# ---------------------------------------------------------------------------


def _toast_problem():
    """Reusable: a snack-cap world matching the kind of setup the user
    has in examples/day.yaml. Forces realistic per-meal floors via
    ``per_meal_min`` so we can trigger the snack-cap conflict."""
    deli = Ingredient(name="deli_chicken", macros={"kcal": 102.0, "protein": 20.0})
    toast = Ingredient(name="toast_bread", macros={"kcal": 250.0, "carbs": 42.0})
    tomato = Ingredient(name="tomato", macros={"kcal": 18.0})
    cheese = Ingredient(name="cheese", macros={"kcal": 234.0, "protein": 34.0})
    return Problem(
        ingredients=(deli, toast, tomato, cheese),
        targets=(MacroTarget(name="kcal", value=1800.0, weight=1.0),),
        meal_library=(),
        tag_constraints={"snack": TagConstraints(macro_max={"kcal": 300.0})},
        bounds={
            "deli_chicken": IngredientBound(per_meal_min=100.0),
            "toast_bread": IngredientBound(per_meal_min=50.0, step=25.0),
            "tomato": IngredientBound(per_meal_min=60.0),
            "cheese": IngredientBound(per_meal_min=30.0),
        },
    )


def test_diagnose_snack_kcal_cap_conflict():
    """The exact failure mode the user hit: many ingredients in a custom
    snack whose serving floors sum past the snack's kcal cap."""
    problem = _toast_problem()
    msgs = diagnose_infeasibility(
        problem,
        DayLog(log_date=date(2026, 5, 21)),
        custom_slots={
            "snack": CustomSlot(
                ingredients=("deli_chicken", "toast_bread", "tomato", "cheese"),
            ),
        },
    )
    assert msgs, "diagnostic should report at least one cause"
    joined = " ".join(msgs)
    assert "kcal" in joined.lower()
    assert "300" in joined  # the cap
    # The biggest contributor at the floor is toast_bread (50 g × 250 = 125).
    assert "toast_bread" in joined


def test_diagnose_returns_empty_for_feasible_setup():
    """When the problem is actually feasible, the diagnostic stays silent
    (no false positives)."""
    problem = _toast_problem()
    msgs = diagnose_infeasibility(
        problem,
        DayLog(log_date=date(2026, 5, 21)),
        custom_slots={
            "snack": CustomSlot(ingredients=("deli_chicken", "tomato")),
            # 100×1.02 + 60×0.18 = 113 kcal min, well under 300.
        },
    )
    assert msgs == []


def test_diagnose_daily_total_max_conflict():
    """If a stepped ingredient's daily cap is already used up by what's
    logged, the diagnostic should flag it instead of an opaque solver
    error."""
    tortilla = Ingredient(name="tortilla", macros={"kcal": 292.0, "carbs": 45.0})
    chicken = Ingredient(name="chicken", macros={"kcal": 100.0, "protein": 20.0})
    meal = LibraryMeal(
        name="wrap",
        tag="lunch_dinner",
        ingredients=("tortilla", "chicken"),
    )
    problem = Problem(
        ingredients=(tortilla, chicken),
        targets=(MacroTarget(name="kcal", value=1800.0, weight=1.0),),
        meal_library=(meal,),
        bounds={
            # One tortilla per day total; pinned to one 60 g unit per meal.
            "tortilla": IngredientBound(
                step=60.0, per_meal_min=60.0, per_meal_max=60.0, total_max=60.0
            ),
        },
    )
    log = DayLog(log_date=date(2026, 5, 21)).with_added(
        LogEntry(ingredient="tortilla", grams=60.0)
    )
    msgs = diagnose_infeasibility(problem, log, {"lunch": "wrap"})
    assert msgs, "should flag the daily tortilla cap"
    joined = " ".join(msgs)
    assert "tortilla" in joined.lower()
    assert "daily" in joined.lower()


def test_diagnose_respects_step_rounding():
    """A step=55 ingredient with per_meal_min=5 actually contributes 55 g
    (next reachable step), not 5 g. The diagnostic must use the rounded
    value when reporting / detecting conflicts."""
    eggs = Ingredient(name="eggs", macros={"kcal": 155.0, "protein": 13.0})
    problem = Problem(
        ingredients=(eggs,),
        targets=(MacroTarget(name="kcal", value=1800.0, weight=1.0),),
        meal_library=(),
        # Eggs alone: min 55 g (one egg) → 85 kcal. Snack cap 80 → conflict.
        tag_constraints={"snack": TagConstraints(macro_max={"kcal": 80.0})},
        bounds={"eggs": IngredientBound(step=55.0, per_meal_min=5.0)},
    )
    msgs = diagnose_infeasibility(
        problem,
        DayLog(log_date=date(2026, 5, 21)),
        custom_slots={"snack": CustomSlot(ingredients=("eggs",))},
    )
    assert msgs
    joined = " ".join(msgs)
    # Either the eggs grams (55) or the kcal at floor (~85) should appear.
    assert "55" in joined or "85" in joined


def test_diagnose_empty_when_nothing_planned():
    """No saved meals and no custom slots → nothing to diagnose."""
    problem = _toast_problem()
    assert (
        diagnose_infeasibility(problem, DayLog(log_date=date(2026, 5, 21))) == []
    )
