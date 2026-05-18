"""Unit tests for the optimizer.

Small, hand-checkable problems with the new (tagged meal library) schema.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from foodtimizer import load_problem, optimize
from foodtimizer.model import (
    Defaults,
    Ingredient,
    IngredientBound,
    LibraryMeal,
    MacroTarget,
    MealIngredient,
    Problem,
    TagConstraints,
)


def _macro_total(plan, ingredients: dict[str, Ingredient], macro: str) -> float:
    return sum(
        it.grams * ingredients[it.ingredient].amount_for(macro) / 100.0
        for it in plan.items
    )


def test_single_ingredient_hits_kcal_exactly():
    """One slot, one-ingredient meal, soft kcal target."""
    ing = Ingredient(name="rice", macros={"kcal": 100.0, "protein": 2.5})
    meal = LibraryMeal(name="rice_only", tag="lunch_dinner", ingredients=("rice",))
    problem = Problem(
        ingredients=(ing,),
        targets=(MacroTarget(name="kcal", value=200.0, weight=1.0),),
        meal_library=(meal,),
    )
    plan = optimize(problem, {"lunch": "rice_only"})
    assert plan.items
    assert plan.items[0].ingredient == "rice"
    assert math.isclose(plan.items[0].grams, 200.0, rel_tol=1e-6)


def test_two_ingredients_match_two_targets():
    """Closed-form check: 20 g protein and 50 g carbs from chicken+rice."""
    chicken = Ingredient(name="chicken", macros={"protein": 30.0, "carbs": 0.0})
    rice = Ingredient(name="rice", macros={"protein": 2.0, "carbs": 28.0})
    meal = LibraryMeal(name="ck", tag="lunch_dinner", ingredients=("chicken", "rice"))
    problem = Problem(
        ingredients=(chicken, rice),
        targets=(
            MacroTarget(name="protein", value=20.0, weight=100.0),
            MacroTarget(name="carbs", value=50.0, weight=100.0),
        ),
        meal_library=(meal,),
    )
    plan = optimize(problem, {"lunch": "ck"})

    ings = {"chicken": chicken, "rice": rice}
    assert math.isclose(_macro_total(plan, ings, "protein"), 20.0, abs_tol=1e-6)
    assert math.isclose(_macro_total(plan, ings, "carbs"), 50.0, abs_tol=1e-6)


def test_tag_kcal_cap_is_respected():
    """The breakfast tag's kcal cap caps the breakfast slot's total kcal."""
    rice = Ingredient(name="rice", macros={"kcal": 100.0, "protein": 2.5})
    meal = LibraryMeal(name="rice_only", tag="breakfast", ingredients=("rice",))
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=500.0, weight=1.0),),
        meal_library=(meal,),
        tag_constraints={"breakfast": TagConstraints(macro_max={"kcal": 150.0})},
    )
    plan = optimize(problem, {"breakfast": "rice_only"})
    total_kcal = sum(it.grams for it in plan.items)  # 100 kcal/100g => grams==kcal
    assert total_kcal <= 150.0 + 1e-6
    assert math.isclose(total_kcal, 150.0, abs_tol=1e-4)


def test_meal_kcal_override_beats_tag():
    """A meal's own macro_max.kcal overrides the tag default."""
    rice = Ingredient(name="rice", macros={"kcal": 100.0})
    meal = LibraryMeal(
        name="capped",
        tag="breakfast",
        ingredients=("rice",),
        macro_max={"kcal": 80.0},
    )
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=500.0, weight=1.0),),
        meal_library=(meal,),
        tag_constraints={"breakfast": TagConstraints(macro_max={"kcal": 300.0})},
    )
    plan = optimize(problem, {"breakfast": "capped"})
    total_grams = sum(it.grams for it in plan.items)
    assert total_grams <= 80.0 + 1e-6


def test_same_meal_in_two_slots():
    """Putting the same library meal in two slots creates two independent
    instances of the same gram variables (lunch + dinner can hold the same dish)."""
    chicken = Ingredient(name="chicken", macros={"kcal": 165.0, "protein": 31.0})
    meal = LibraryMeal(name="chicken_only", tag="lunch_dinner", ingredients=("chicken",))
    problem = Problem(
        ingredients=(chicken,),
        targets=(MacroTarget(name="protein", value=62.0, weight=10.0),),
        meal_library=(meal,),
        tag_constraints={"lunch_dinner": TagConstraints(macro_max={"kcal": 200.0})},
    )
    plan = optimize(problem, {"lunch": "chicken_only", "dinner": "chicken_only"})

    slots = {it.slot for it in plan.items}
    assert slots == {"lunch", "dinner"}
    ings = {"chicken": chicken}
    assert math.isclose(_macro_total(plan, ings, "protein"), 62.0, abs_tol=1e-4)


def test_lower_bound_is_respected():
    """A target `lower` is a hard inequality."""
    spinach = Ingredient(name="spinach", macros={"kcal": 23.0, "fibre": 2.2})
    meal = LibraryMeal(name="spinach_only", tag="lunch_dinner", ingredients=("spinach",))
    problem = Problem(
        ingredients=(spinach,),
        targets=(
            MacroTarget(name="kcal", value=100.0, weight=1.0),
            MacroTarget(name="fibre", lower=10.0),
        ),
        meal_library=(meal,),
    )
    plan = optimize(problem, {"lunch": "spinach_only"})
    ings = {"spinach": spinach}
    assert _macro_total(plan, ings, "fibre") >= 10.0 - 1e-6


def test_ingredient_per_meal_max_is_respected():
    oil = Ingredient(name="oil", macros={"kcal": 884.0, "fat": 100.0})
    rice = Ingredient(name="rice", macros={"kcal": 100.0, "fat": 0.3})
    meal = LibraryMeal(name="oil_rice", tag="lunch_dinner", ingredients=("oil", "rice"))
    problem = Problem(
        ingredients=(oil, rice),
        targets=(MacroTarget(name="kcal", value=2000.0, weight=1.0),),
        meal_library=(meal,),
        bounds={"oil": IngredientBound(per_meal_max=15.0)},
    )
    plan = optimize(problem, {"lunch": "oil_rice"})
    grams = {it.ingredient: it.grams for it in plan.items}
    assert grams.get("oil", 0.0) <= 15.0 + 1e-6


def test_unknown_meal_in_day_raises():
    chicken = Ingredient(name="chicken", macros={"kcal": 165.0})
    meal = LibraryMeal(name="cm", tag="lunch_dinner", ingredients=("chicken",))
    problem = Problem(
        ingredients=(chicken,),
        targets=(MacroTarget(name="kcal", value=100.0, weight=1.0),),
        meal_library=(meal,),
    )
    with pytest.raises(ValueError, match="not in library"):
        optimize(problem, {"lunch": "does_not_exist"})


def test_yaml_example_runs_with_default_day():
    """Smoke test: load the example YAML and run with its default day plan."""
    here = Path(__file__).resolve().parents[1]
    cfg = here / "examples" / "day.yaml"
    problem = load_problem(cfg)
    plan = optimize(problem)  # uses default_day

    assert not plan.status.startswith("FAILED")
    assert plan.items
    assert set(plan.slot_meals) == {"breakfast", "lunch", "dinner", "snack"}

    # Protein target is heavily weighted; expect it to be hit closely.
    assert math.isclose(plan.macro_totals["protein"], 135.0, abs_tol=2.0)

    # Breakfast slot must obey its 400 kcal tag cap.
    ings = {ing.name: ing for ing in problem.ingredients}
    breakfast_kcal = sum(
        it.grams * ings[it.ingredient].amount_for("kcal") / 100.0
        for it in plan.items
        if it.slot == "breakfast"
    )
    assert breakfast_kcal <= 400.0 + 1e-4


def test_step_constraint_forces_integer_multiples():
    """An ingredient with `step` must be consumed in whole multiples of step.

    Eggs (155 kcal / 100 g, step=55 g) into a meal with kcal_max=400. The
    optimizer should pick an integer number of egg units (0, 55, 110, 165, ...)
    that, multiplied by 1.55 kcal/g, stays below 400 kcal and is as close as
    possible to a tight kcal target.
    """
    eggs = Ingredient(name="eggs", macros={"kcal": 155.0, "protein": 13.0})
    meal = LibraryMeal(name="eggs_only", tag="breakfast", ingredients=("eggs",))
    problem = Problem(
        ingredients=(eggs,),
        targets=(MacroTarget(name="kcal", value=400.0, weight=1.0),),
        meal_library=(meal,),
        tag_constraints={"breakfast": TagConstraints(macro_max={"kcal": 400.0})},
        bounds={"eggs": IngredientBound(step=55.0, per_meal_max=275.0)},
    )
    plan = optimize(problem, {"breakfast": "eggs_only"})

    grams = {it.ingredient: it.grams for it in plan.items}
    egg_g = grams.get("eggs", 0.0)
    # Must be an integer multiple of 55 g, within (0, 275].
    assert egg_g > 0
    assert math.isclose(egg_g / 55.0, round(egg_g / 55.0), abs_tol=1e-6)
    # 165 g = 3 eggs => 255.75 kcal (closest under cap)
    # 220 g = 4 eggs => 341 kcal (closest to 400 under the kcal_max=400 cap)
    # Pick whichever is closer to 400; 4 eggs at 341 kcal wins.
    assert math.isclose(egg_g, 220.0, abs_tol=1e-4)


def test_step_with_two_stepped_ingredients_and_continuous_fill():
    """Tortilla (60 g step) + cheddar (continuous) hit a 600 kcal target.

    The cheddar should fill the gap left after the integer-quantized tortilla.
    """
    tortilla = Ingredient(name="tortilla", macros={"kcal": 310.0, "protein": 9.0})
    cheddar = Ingredient(name="cheddar", macros={"kcal": 402.0, "protein": 25.0})
    meal = LibraryMeal(
        name="wrap", tag="lunch_dinner", ingredients=("tortilla", "cheddar")
    )
    problem = Problem(
        ingredients=(tortilla, cheddar),
        targets=(MacroTarget(name="kcal", value=600.0, weight=1.0),),
        meal_library=(meal,),
        tag_constraints={"lunch_dinner": TagConstraints(macro_max={"kcal": 900.0})},
        bounds={
            "tortilla": IngredientBound(step=60.0, per_meal_max=180.0),
            "cheddar": IngredientBound(per_meal_max=100.0),
        },
    )
    plan = optimize(problem, {"lunch": "wrap"})

    grams = {it.ingredient: it.grams for it in plan.items}
    t_g = grams.get("tortilla", 0.0)
    c_g = grams.get("cheddar", 0.0)
    assert math.isclose(t_g / 60.0, round(t_g / 60.0), abs_tol=1e-6), (
        f"tortilla {t_g} g is not a multiple of 60 g"
    )
    total_kcal = t_g * 3.10 + c_g * 4.02
    assert math.isclose(total_kcal, 600.0, abs_tol=0.5), (
        f"expected ~600 kcal, got {total_kcal:.2f}"
    )


def test_daily_unit_cap_via_units_field(tmp_path):
    """`total_max_units: 3` with `step: 55` should cap eggs at 3 × 55 = 165 g/day.

    We load via YAML to exercise the new convenience parsing.
    """
    yaml_text = """
ingredients:
  eggs: { kcal: 155, protein: 13.0 }
targets:
  kcal: { value: 600, weight: 1.0 }
meal_library:
  lunch_eggs:
    tag: lunch_dinner
    ingredients: [eggs]
  dinner_eggs:
    tag: lunch_dinner
    ingredients: [eggs]
ingredient_bounds:
  eggs: { step: 55, per_meal_max_units: 4, total_max_units: 3 }
day:
  lunch: lunch_eggs
  dinner: dinner_eggs
"""
    cfg = tmp_path / "eggs.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")

    problem = load_problem(cfg)
    plan = optimize(problem)

    total_eggs_g = sum(it.grams for it in plan.items if it.ingredient == "eggs")
    assert total_eggs_g <= 3 * 55.0 + 1e-6
    # Each slot's grams must be an integer multiple of 55.
    for it in plan.items:
        if it.ingredient == "eggs":
            assert math.isclose(it.grams / 55.0, round(it.grams / 55.0), abs_tol=1e-6)


def test_units_field_requires_step():
    """Specifying *_units without `step` should be a config error."""
    yaml_text = """
ingredients:
  rice: { kcal: 100, protein: 2.5 }
targets:
  kcal: { value: 200 }
meal_library:
  m:
    tag: lunch_dinner
    ingredients: [rice]
ingredient_bounds:
  rice: { total_max_units: 3 }
"""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        path = fh.name
    with pytest.raises(ValueError, match=r"\*_units"):
        load_problem(path)


def test_tag_macro_min_is_respected():
    """A `macro_min` on a tag enforces a per-slot minimum for that macro."""
    chicken = Ingredient(name="chicken", macros={"kcal": 165.0, "protein": 31.0, "carbs": 0.0})
    rice = Ingredient(name="rice", macros={"kcal": 130.0, "protein": 2.7, "carbs": 28.0})
    meal = LibraryMeal(
        name="ck", tag="lunch_dinner", ingredients=("chicken", "rice")
    )
    problem = Problem(
        ingredients=(chicken, rice),
        targets=(
            MacroTarget(name="protein", value=40.0, weight=100.0),
            MacroTarget(name="kcal", value=600.0, weight=1.0),
        ),
        meal_library=(meal,),
        tag_constraints={
            "lunch_dinner": TagConstraints(macro_min={"carbs": 50.0}),
        },
    )
    plan = optimize(problem, {"lunch": "ck"})

    ings = {"chicken": chicken, "rice": rice}
    assert _macro_total(plan, ings, "carbs") >= 50.0 - 1e-6


def test_meal_macro_max_overrides_tag():
    """Per-meal `macro_max` for a specific macro overrides the tag default."""
    rice = Ingredient(name="rice", macros={"kcal": 100.0, "carbs": 30.0})
    # tag says carbs<=100; meal says carbs<=20 -> meal wins
    meal = LibraryMeal(
        name="capped",
        tag="lunch_dinner",
        ingredients=("rice",),
        macro_max={"carbs": 20.0},
    )
    problem = Problem(
        ingredients=(rice,),
        targets=(MacroTarget(name="kcal", value=500.0, weight=1.0),),
        meal_library=(meal,),
        tag_constraints={
            "lunch_dinner": TagConstraints(macro_max={"carbs": 100.0}),
        },
    )
    plan = optimize(problem, {"lunch": "capped"})
    ings = {"rice": rice}
    assert _macro_total(plan, ings, "carbs") <= 20.0 + 1e-6


def test_pinned_step_ingredient():
    """`per_meal_min_units = per_meal_max_units = 1` pins to exactly 1 unit."""
    tortilla = Ingredient(name="tortilla", macros={"kcal": 310.0, "carbs": 50.0})
    rice = Ingredient(name="rice", macros={"kcal": 130.0, "carbs": 28.0})
    meal = LibraryMeal(
        name="wrap", tag="lunch_dinner", ingredients=("tortilla", "rice")
    )
    problem = Problem(
        ingredients=(tortilla, rice),
        targets=(MacroTarget(name="kcal", value=500.0, weight=1.0),),
        meal_library=(meal,),
        bounds={
            "tortilla": IngredientBound(
                step=60.0, per_meal_min=60.0, per_meal_max=60.0
            ),
        },
    )
    plan = optimize(problem, {"lunch": "wrap"})
    grams = {it.ingredient: it.grams for it in plan.items}
    assert math.isclose(grams.get("tortilla", 0.0), 60.0, abs_tol=1e-6)


def test_kcal_max_shortcut_equivalent_to_macro_max_kcal(tmp_path):
    """The YAML shortcut `kcal_max` parses into `macro_max.kcal`."""
    yaml_text = """
ingredients:
  rice: { kcal: 100, protein: 2.5, carbs: 28 }
targets:
  kcal: { value: 500 }
tag_constraints:
  lunch_dinner: { kcal_max: 200 }
meal_library:
  m:
    tag: lunch_dinner
    ingredients: [rice]
day:
  lunch: m
"""
    cfg = tmp_path / "k.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)
    assert problem.tag_constraints["lunch_dinner"].macro_max == {"kcal": 200.0}


def test_defaults_per_meal_min_forces_presence():
    """`defaults.per_meal_min` ensures every ingredient in a meal is used."""
    a = Ingredient(name="a", macros={"kcal": 100.0, "protein": 30.0})
    b = Ingredient(name="b", macros={"kcal": 100.0, "protein": 5.0})
    meal = LibraryMeal(name="m", tag="lunch_dinner", ingredients=("a", "b"))
    problem = Problem(
        ingredients=(a, b),
        targets=(MacroTarget(name="protein", value=30.0, weight=10.0),),
        meal_library=(meal,),
        defaults=Defaults(per_meal_min=10.0),
    )
    plan = optimize(problem, {"lunch": "m"})
    grams = {it.ingredient: it.grams for it in plan.items}
    # Both ingredients must appear with >= 10 g, even though `a` alone could
    # satisfy the protein target more cheaply.
    assert grams.get("a", 0.0) >= 10.0 - 1e-6
    assert grams.get("b", 0.0) >= 10.0 - 1e-6


def test_defaults_per_meal_max_caps_low_density():
    """`defaults.per_meal_max` kills the unlimited-broccoli case."""
    broccoli = Ingredient(
        name="broccoli", macros={"kcal": 34.0, "protein": 2.8, "carbs": 7.0}
    )
    meal = LibraryMeal(name="b", tag="lunch_dinner", ingredients=("broccoli",))
    problem = Problem(
        ingredients=(broccoli,),
        # 1000 kcal would naively require ~2941 g of broccoli.
        targets=(MacroTarget(name="kcal", value=1000.0, weight=1.0),),
        meal_library=(meal,),
        defaults=Defaults(per_meal_max=300.0),
    )
    plan = optimize(problem, {"lunch": "b"})
    grams = {it.ingredient: it.grams for it in plan.items}
    assert grams.get("broccoli", 0.0) <= 300.0 + 1e-6


def test_ingredient_bound_overrides_default():
    """`ingredient_bounds.<name>.per_meal_max` wins over `defaults.per_meal_max`."""
    oil = Ingredient(name="oil", macros={"kcal": 884.0})
    meal = LibraryMeal(name="m", tag="lunch_dinner", ingredients=("oil",))
    problem = Problem(
        ingredients=(oil,),
        targets=(MacroTarget(name="kcal", value=5000.0, weight=1.0),),
        meal_library=(meal,),
        defaults=Defaults(per_meal_max=300.0),
        bounds={"oil": IngredientBound(per_meal_max=15.0)},
    )
    plan = optimize(problem, {"lunch": "m"})
    grams = {it.ingredient: it.grams for it in plan.items}
    assert grams.get("oil", 0.0) <= 15.0 + 1e-6


def test_defaults_with_stepped_ingredient():
    """`defaults.per_meal_min: 5` with eggs (step=55) forces at least 1 egg (55 g)."""
    eggs = Ingredient(name="eggs", macros={"kcal": 155.0, "protein": 13.0})
    meal = LibraryMeal(name="m", tag="breakfast", ingredients=("eggs",))
    problem = Problem(
        ingredients=(eggs,),
        targets=(MacroTarget(name="protein", value=10.0, weight=1.0),),
        meal_library=(meal,),
        defaults=Defaults(per_meal_min=5.0),
        bounds={"eggs": IngredientBound(step=55.0)},
    )
    plan = optimize(problem, {"breakfast": "m"})
    grams = {it.ingredient: it.grams for it in plan.items}
    assert grams.get("eggs", 0.0) >= 55.0 - 1e-6


def test_main_label_applies_main_min():
    """An ingredient marked `main` in a meal gets `defaults.main_min` as floor."""
    bulk = Ingredient(name="bulk", macros={"kcal": 50.0, "protein": 1.0})
    flav = Ingredient(name="flav", macros={"kcal": 200.0, "protein": 10.0})
    meal = LibraryMeal(
        name="m",
        tag="lunch_dinner",
        ingredients=("bulk", "flav"),
        ingredient_specs={
            "bulk": MealIngredient(main=True),
            "flav": MealIngredient(),
        },
    )
    problem = Problem(
        ingredients=(bulk, flav),
        targets=(MacroTarget(name="protein", value=20.0, weight=100.0),),
        meal_library=(meal,),
        defaults=Defaults(per_meal_min=5.0, main_min=80.0),
    )
    plan = optimize(problem, {"lunch": "m"})
    grams = {it.ingredient: it.grams for it in plan.items}
    assert grams["bulk"] >= 80.0 - 1e-6, (
        f"bulk should hit main_min=80 g, got {grams['bulk']}"
    )
    assert grams["flav"] >= 5.0 - 1e-6


def test_max_of_mins_does_not_weaken_global():
    """Marking `main` (main_min=80) never weakens a global per_meal_min=120."""
    chicken = Ingredient(name="chicken", macros={"kcal": 100.0, "protein": 30.0})
    meal = LibraryMeal(
        name="m",
        tag="lunch_dinner",
        ingredients=("chicken",),
        ingredient_specs={"chicken": MealIngredient(main=True)},
    )
    problem = Problem(
        ingredients=(chicken,),
        targets=(MacroTarget(name="protein", value=20.0, weight=1.0),),
        meal_library=(meal,),
        defaults=Defaults(main_min=80.0),
        bounds={"chicken": IngredientBound(per_meal_min=120.0)},
    )
    plan = optimize(problem, {"lunch": "m"})
    grams = {it.ingredient: it.grams for it in plan.items}
    assert grams["chicken"] >= 120.0 - 1e-6


def test_meal_spec_min_overrides_main_default():
    """Per-meal explicit `min` tightens beyond `main_min` (both apply, max wins)."""
    a = Ingredient(name="a", macros={"kcal": 100.0, "protein": 20.0})
    meal = LibraryMeal(
        name="m",
        tag="lunch_dinner",
        ingredients=("a",),
        ingredient_specs={"a": MealIngredient(main=True, min=150.0)},
    )
    problem = Problem(
        ingredients=(a,),
        targets=(MacroTarget(name="protein", value=20.0, weight=1.0),),
        meal_library=(meal,),
        defaults=Defaults(main_min=80.0),
    )
    plan = optimize(problem, {"lunch": "m"})
    grams = {it.ingredient: it.grams for it in plan.items}
    assert grams["a"] >= 150.0 - 1e-6


def test_min_capped_at_max_to_keep_feasible():
    """If main_min > global max, min is silently capped at max (no infeasibility)."""
    tortilla = Ingredient(name="tortilla", macros={"kcal": 300.0, "carbs": 50.0})
    meal = LibraryMeal(
        name="m",
        tag="lunch_dinner",
        ingredients=("tortilla",),
        ingredient_specs={"tortilla": MealIngredient(main=True)},
    )
    problem = Problem(
        ingredients=(tortilla,),
        targets=(MacroTarget(name="carbs", value=30.0, weight=1.0),),
        meal_library=(meal,),
        defaults=Defaults(main_min=80.0),
        bounds={"tortilla": IngredientBound(per_meal_max=60.0)},
    )
    plan = optimize(problem, {"lunch": "m"})
    grams = {it.ingredient: it.grams for it in plan.items}
    assert math.isclose(grams["tortilla"], 60.0, abs_tol=1e-6)


def test_yaml_rich_ingredients_form(tmp_path):
    """Parse a meal_library entry with the rich mapping form."""
    yaml_text = """
ingredients:
  a: { kcal: 100, protein: 20 }
  b: { kcal: 50,  protein: 5 }
targets:
  protein: { value: 30, weight: 1.0 }
defaults:
  per_meal_min: 5
  main_min: 70
meal_library:
  m:
    tag: lunch_dinner
    ingredients:
      a: main
      b: { min: 20, max: 100 }
day:
  lunch: m
"""
    cfg = tmp_path / "rich.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)
    meal = problem.meal_by_name("m")
    assert meal.ingredients == ("a", "b")
    assert meal.ingredient_specs["a"].main is True
    assert meal.ingredient_specs["b"].min == 20.0
    assert meal.ingredient_specs["b"].max == 100.0

    plan = optimize(problem)
    grams = {it.ingredient: it.grams for it in plan.items}
    assert grams["a"] >= 70.0 - 1e-6  # main_min applied
    assert grams["b"] >= 20.0 - 1e-6  # meal-level min applied
    assert grams["b"] <= 100.0 + 1e-6  # meal-level max applied


def test_inline_serving_is_treated_as_per_meal_min(tmp_path):
    """`serving: N` on an ingredient acts as `per_meal_min: N`.

    Without `serving`, the global default (5 g) would let the optimizer
    drop the sauce to 5 g; with `serving: 80`, the sauce stays >= 80 g.
    """
    yaml_text = """
ingredients:
  rice:         { kcal: 350, carbs: 77 }
  tomato_sauce: { kcal: 27,  carbs: 4.1, serving: 80 }
targets:
  kcal: { value: 500, weight: 1.0 }
defaults:
  per_meal_min: 5
meal_library:
  dish:
    tag: lunch_dinner
    ingredients: [rice, tomato_sauce]
day:
  lunch: dish
"""
    cfg = tmp_path / "inline.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)

    assert "tomato_sauce" in problem.bounds
    assert problem.bounds["tomato_sauce"].per_meal_min == 80.0

    plan = optimize(problem)
    grams = {it.ingredient: it.grams for it in plan.items}
    assert grams["tomato_sauce"] >= 80.0 - 1e-6


def test_inline_step_and_units_on_ingredient(tmp_path):
    """`step` and `*_units` work inline on an ingredient definition,
    no separate `ingredient_bounds:` section needed."""
    yaml_text = """
ingredients:
  eggs: { kcal: 155, protein: 13, step: 55, per_meal_max_units: 2, total_max_units: 2 }
  rice: { kcal: 350, protein: 9, carbs: 77 }
targets:
  kcal: { value: 600, weight: 1.0 }
meal_library:
  m1: { tag: lunch_dinner, ingredients: [eggs, rice] }
  m2: { tag: lunch_dinner, ingredients: [eggs, rice] }
day:
  lunch:  m1
  dinner: m2
"""
    cfg = tmp_path / "inline_step.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)

    eggs_bound = problem.bounds["eggs"]
    assert eggs_bound.step == 55.0
    assert eggs_bound.per_meal_max == 2 * 55.0
    assert eggs_bound.total_max == 2 * 55.0

    plan = optimize(problem)
    total_eggs_g = sum(it.grams for it in plan.items if it.ingredient == "eggs")
    assert total_eggs_g <= 2 * 55.0 + 1e-6
    for it in plan.items:
        if it.ingredient == "eggs":
            assert math.isclose(it.grams / 55.0, round(it.grams / 55.0), abs_tol=1e-6)


def test_explicit_ingredient_bounds_override_inline(tmp_path):
    """A field in `ingredient_bounds:` overrides the inline equivalent.

    Inline declares `serving: 50` (= per_meal_min 50) but the explicit
    section overrides it to 100 — optimizer must use 100.
    """
    yaml_text = """
ingredients:
  rice: { kcal: 350, carbs: 77, serving: 50, per_meal_max: 200 }
targets:
  kcal: { value: 350, weight: 1.0 }
meal_library:
  m: { tag: lunch_dinner, ingredients: [rice] }
day:
  lunch: m
ingredient_bounds:
  rice: { per_meal_min: 100 }
"""
    cfg = tmp_path / "override.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)

    # Explicit per_meal_min wins; the per_meal_max from inline survives
    # because the explicit section didn't redeclare it.
    assert problem.bounds["rice"].per_meal_min == 100.0
    assert problem.bounds["rice"].per_meal_max == 200.0

    plan = optimize(problem)
    grams = {it.ingredient: it.grams for it in plan.items}
    assert grams["rice"] >= 100.0 - 1e-6


def test_anchor_pulls_amount_toward_typical(tmp_path):
    """When macros are perfectly satisfiable in many ways, the anchor
    breaks the tie and pulls the amount toward the typical value."""
    yaml_text = """
ingredients:
  rice:  { kcal: 100, protein: 2 }
  sauce: { kcal: 100, protein: 2 }
targets:
  kcal: { value: 600, weight: 1.0 }
defaults:
  anchor_weight: 0.05
meal_library:
  m:
    tag: lunch_dinner
    ingredients:
      rice:  200      # anchor 200
      sauce: 400      # anchor 400
day:
  lunch: m
"""
    cfg = tmp_path / "anchor.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)

    assert problem.meal_by_name("m").ingredient_specs["rice"].anchor == 200.0
    assert problem.meal_by_name("m").ingredient_specs["sauce"].anchor == 400.0

    plan = optimize(problem)
    grams = {it.ingredient: it.grams for it in plan.items}
    # 200 g + 400 g hits 600 kcal exactly and both anchors exactly.
    assert math.isclose(grams["rice"], 200.0, abs_tol=1.0)
    assert math.isclose(grams["sauce"], 400.0, abs_tol=1.0)

    # PlanItem reports its anchor so callers can render the deviation.
    item_by_ing = {it.ingredient: it for it in plan.items}
    assert item_by_ing["rice"].anchor == 200.0


def test_anchor_yields_to_macro_target(tmp_path):
    """Macro targets have much higher weight than anchor, so the anchor
    can be pulled away when the macro requires it."""
    yaml_text = """
ingredients:
  rice:  { kcal: 100, protein: 2 }
  sauce: { kcal: 100, protein: 2 }
targets:
  kcal: { value: 1000, weight: 1.0 }
defaults:
  anchor_weight: 0.05
meal_library:
  m:
    tag: lunch_dinner
    ingredients:
      rice:  200
      sauce: 400
day:
  lunch: m
"""
    cfg = tmp_path / "anchor_yield.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)
    plan = optimize(problem)
    # 600 g total only gives 600 kcal; we need 1000 kcal, so at least one
    # variable must drift above its anchor.
    total_g = sum(it.grams for it in plan.items)
    assert total_g >= 1000.0 - 1e-3
    assert math.isclose(plan.macro_totals["kcal"], 1000.0, abs_tol=1e-3)


def test_anchor_respects_step_and_max(tmp_path):
    """Anchor can be infeasible (e.g. anchor 180 with a step of 150);
    the optimizer still respects step and snaps to the nearest legal
    multiple. The anchor only adds a penalty term, never a constraint."""
    yaml_text = """
ingredients:
  chicken: { kcal: 100, protein: 20, step: 150, per_meal_max_units: 2 }
targets:
  kcal: { value: 1000, weight: 1.0 }
defaults:
  anchor_weight: 0.05
meal_library:
  m:
    tag: lunch_dinner
    ingredients:
      chicken: 180         # anchor 180 g (between 150 and 300)
day:
  lunch: m
"""
    cfg = tmp_path / "anchor_step.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)
    plan = optimize(problem)
    grams = {it.ingredient: it.grams for it in plan.items}
    # chicken must be 150 or 300; macros want 1000 kcal => 300 wins.
    assert grams["chicken"] in (150.0, 300.0)
    assert math.isclose(grams["chicken"] / 150.0, round(grams["chicken"] / 150.0))


def test_anchor_weight_cli_override_disables_pull(tmp_path):
    """Passing ``anchor_weight=0`` to optimize disables anchor pulls and
    the optimizer is free to choose any macro-feasible point."""
    yaml_text = """
ingredients:
  a: { kcal: 100 }
  b: { kcal: 100 }
targets:
  kcal: { value: 200, weight: 1.0 }
defaults:
  anchor_weight: 0.5
meal_library:
  m:
    tag: lunch_dinner
    ingredients:
      a: 50
      b: 50
day:
  lunch: m
"""
    cfg = tmp_path / "anchor_off.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)

    plan_on = optimize(problem)
    grams_on = {it.ingredient: it.grams for it in plan_on.items}
    # With anchor pull active, both at 50 g hits 100 kcal short of target
    # but pays no anchor penalty; pulling either up to 150 satisfies macros
    # at the cost of 100 g of anchor deviation (weight 0.5 -> 50). Either
    # case at least keeps both near their anchors.
    assert all(abs(grams_on[k] - 50.0) <= 100.0 for k in grams_on)

    plan_off = optimize(problem, anchor_weight=0.0)
    # With anchors disabled, plan exists with status reporting; objective
    # only counts macro deviation.
    assert plan_off.status and not plan_off.status.startswith("FAILED")


def test_anchor_bare_number_and_dict_form_equivalent(tmp_path):
    """Bare number and ``{anchor: N}`` produce equivalent specs."""
    yaml_text = """
ingredients:
  a: { kcal: 100 }
  b: { kcal: 100 }
targets:
  kcal: { value: 200 }
meal_library:
  m1:
    tag: lunch_dinner
    ingredients:
      a: 150
      b: { anchor: 250 }
day:
  lunch: m1
"""
    cfg = tmp_path / "anchor_forms.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)
    specs = problem.meal_by_name("m1").ingredient_specs
    assert specs["a"].anchor == 150.0
    assert specs["b"].anchor == 250.0
    assert specs["a"].main is False
    assert specs["b"].main is False


def test_inline_serving_conflicts_with_per_meal_min(tmp_path):
    """Setting both `serving` and `per_meal_min` on the same ingredient
    should raise (one is an alias of the other)."""
    yaml_text = """
ingredients:
  rice: { kcal: 350, serving: 50, per_meal_min: 80 }
targets:
  kcal: { value: 100 }
meal_library:
  m: { tag: lunch_dinner, ingredients: [rice] }
"""
    cfg = tmp_path / "conflict.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"serving"):
        load_problem(cfg)


def test_cli_slot_override(tmp_path):
    """Picking a different meal at the CLI changes the plan structure.

    Robust to library edits: picks any lunch_dinner meal that's not the
    current default lunch.
    """
    here = Path(__file__).resolve().parents[1]
    cfg = here / "examples" / "day.yaml"
    problem = load_problem(cfg)

    current_lunch = problem.default_day.get("lunch")
    candidate = next(
        (m for m in problem.meal_library if m.tag == "lunch_dinner" and m.name != current_lunch),
        None,
    )
    assert candidate is not None, "example needs at least 2 lunch_dinner meals"

    custom_day = dict(problem.default_day)
    custom_day["lunch"] = candidate.name
    plan = optimize(problem, custom_day)

    assert plan.slot_meals["lunch"] == candidate.name
    lunch_ings = {it.ingredient for it in plan.items if it.slot == "lunch"}
    assert lunch_ings.issubset(set(candidate.ingredients))


def test_heuristic_suggester_returns_anchor_per_ingredient(tmp_path):
    """The offline heuristic returns a positive gram amount for every
    ingredient in the meal and bumps `main` items above auxiliary ones."""
    yaml_text = """
ingredients:
  chicken_breast: { kcal: 97, protein: 20, fat: 0.5, carbs: 0 }
  rice_white:     { kcal: 350, protein: 9, carbs: 77, fat: 1 }
  garlic:         { kcal: 149, protein: 6, carbs: 28, fat: 0.5 }
  avocado_oil:    { kcal: 824, protein: 0, carbs: 0, fat: 91 }
targets:
  kcal: { value: 600, weight: 1.0 }
meal_library:
  chicken_rice:
    tag: lunch_dinner
    ingredients:
      chicken_breast: main
      rice_white:     main
      garlic:         {}
      avocado_oil:    {}
day:
  lunch: chicken_rice
"""
    cfg = tmp_path / "h.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)

    from foodtimizer.suggest import HeuristicSuggester, suggest_anchors

    grams = suggest_anchors(problem, "chicken_rice", None, suggester=HeuristicSuggester())
    assert set(grams) == {"chicken_breast", "rice_white", "garlic", "avocado_oil"}
    # main items get bumped to a structural size
    assert grams["chicken_breast"] >= 100
    assert grams["rice_white"] >= 100
    # oil is always small in the heuristic
    assert grams["avocado_oil"] <= 20
    # main items outweigh aux items on average — that's the heuristic's job;
    # individual items (especially aromatic vegetables) may be coarse and
    # are intended to be refined by the LLM backend.
    main_avg = (grams["chicken_breast"] + grams["rice_white"]) / 2
    aux_avg = (grams["garlic"] + grams["avocado_oil"]) / 2
    assert main_avg > aux_avg


def test_suggest_anchors_rejects_unknown_ingredient(tmp_path):
    yaml_text = """
ingredients:
  rice: { kcal: 100 }
targets: { kcal: 100 }
meal_library:
  m: { tag: lunch_dinner, ingredients: [rice] }
"""
    cfg = tmp_path / "h2.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    problem = load_problem(cfg)

    from foodtimizer.suggest import HeuristicSuggester, suggest_anchors

    with pytest.raises(SystemExit, match=r"Unknown ingredient"):
        suggest_anchors(
            problem, "anything", ["rice", "no_such_thing"], suggester=HeuristicSuggester()
        )


def test_render_yaml_block_uses_rich_form_for_main():
    from foodtimizer.suggest import render_yaml_block

    out = render_yaml_block(
        "m",
        {"chicken": 150, "rice": 120, "oil": 10},
        main_flags={"chicken": True, "rice": True, "oil": False},
    )
    assert "anchor: 150, main: true" in out
    assert "anchor: 120, main: true" in out
    # aux ingredient uses bare-number form
    assert "oil: " in out and "anchor: 10" not in out


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
