"""YAML configuration loader.

The YAML schema is documented in `examples/day.yaml`. Top-level keys:

- ``ingredients``: name -> mapping of macros (per 100 g) **and/or**
  inline bound fields (``step``, ``serving``, ``per_meal_min/max``,
  ``per_meal_min/max_units``, ``total_min/max[_units]``)
- ``targets``: macro -> number (soft, weight 1) or mapping with
  ``value``, ``weight``, ``hard``, ``lower``, ``upper``
- ``tag_constraints``: tag -> ``{kcal_max, kcal_min, macro_min, macro_max}``
- ``meal_library``: meal_name -> ``{tag, ingredients, kcal_max?, kcal_min?,
  macro_min?, macro_max?}``. ``ingredients`` may be a list of names or a
  mapping with per-meal-ingredient specs (``main``, ``min``, ``max``).
- ``ingredient_bounds`` (optional): ingredient -> bound fields. Overrides
  the inline bounds field-by-field. Same keys as the inline form.
- ``day``: slot_name -> meal_name (default day plan)
- ``defaults``: ``{per_meal_min, per_meal_max, main_min}`` — global floors
  / ceilings applied when an ingredient declares no tighter value.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from .model import (
    Defaults,
    Ingredient,
    IngredientBound,
    LibraryMeal,
    MacroTarget,
    MealIngredient,
    Problem,
    TagConstraints,
)


def load_problem(path: str | Path) -> Problem:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
    return _problem_from_dict(raw)


def _problem_from_dict(data: dict[str, Any]) -> Problem:
    ingredients, inline_bound_specs = _parse_ingredients(data.get("ingredients", {}))
    targets = _parse_targets(data.get("targets", {}))
    tag_constraints = _parse_tag_constraints(data.get("tag_constraints", {}))
    library = _parse_meal_library(data.get("meal_library", data.get("meals", {})))
    # Inline bound fields on ingredients (e.g. ``chicken_breast: { ..., step: 150 }``)
    # are merged with the dedicated ``ingredient_bounds`` section. Inline values
    # provide the defaults; explicit ``ingredient_bounds`` entries override
    # them on a per-field basis.
    merged_bound_specs = _merge_bound_specs(inline_bound_specs, data.get("ingredient_bounds", {}))
    bounds = _parse_bounds(merged_bound_specs)
    day = _parse_day(data.get("day", {}))
    defaults = _parse_defaults(data.get("defaults", {}))

    _validate(ingredients, library, targets, tag_constraints, day, bounds)
    return Problem(
        ingredients=tuple(ingredients),
        targets=tuple(targets),
        meal_library=tuple(library),
        tag_constraints=tag_constraints,
        bounds=bounds,
        default_day=day,
        defaults=defaults,
    )


# Keys recognised on an ingredient definition that are *not* macros but
# describe bounds / serving info. They are stripped from the macros and
# folded into the merged ingredient_bounds.
_INGREDIENT_BOUND_KEYS: frozenset[str] = frozenset(
    {
        "step",
        "per_meal_min",
        "per_meal_max",
        "total_min",
        "total_max",
        "per_meal_min_units",
        "per_meal_max_units",
        "total_min_units",
        "total_max_units",
        "serving",  # alias for per_meal_min
    }
)


def _merge_bound_specs(
    inline: dict[str, dict[str, Any]], explicit: Any
) -> dict[str, dict[str, Any]]:
    """Combine inline ingredient bound fields with the explicit
    ``ingredient_bounds:`` section. Explicit wins per field."""
    if explicit is None:
        explicit = {}
    if not isinstance(explicit, dict):
        raise ValueError("`ingredient_bounds` must be a mapping")
    merged: dict[str, dict[str, Any]] = {}
    for name, spec in inline.items():
        merged[name] = dict(spec)
    for name, spec in explicit.items():
        if spec is None:
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"ingredient_bounds.{name} must be a mapping")
        merged.setdefault(name, {}).update(spec)
    return merged


def _parse_defaults(section: Any) -> Defaults:
    if not section:
        return Defaults()
    if not isinstance(section, dict):
        raise ValueError("`defaults` must be a mapping")
    return Defaults(
        per_meal_min=_opt_float(section.get("per_meal_min")),
        per_meal_max=_opt_float(section.get("per_meal_max")),
        main_min=_opt_float(section.get("main_min")),
        anchor_weight=_opt_float(section.get("anchor_weight")),
    )


def _parse_ingredients(
    section: Any,
) -> tuple[list[Ingredient], dict[str, dict[str, Any]]]:
    """Parse the ``ingredients:`` section.

    Each ingredient may freely interleave macro keys (e.g. ``kcal``, ``protein``)
    with bound keys (e.g. ``step``, ``per_meal_min``, ``serving``). Macro keys
    become the ``Ingredient.macros`` mapping; bound keys are split out and
    returned as inline bound specs, to be merged with ``ingredient_bounds``
    later.
    """
    if not isinstance(section, dict):
        raise ValueError("`ingredients` must be a mapping of name -> spec")
    out: list[Ingredient] = []
    inline_bounds: dict[str, dict[str, Any]] = {}
    for name, spec in section.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Ingredient {name!r} must map to a dict")
        macros: dict[str, float] = {}
        bound_fields: dict[str, Any] = {}
        # `serving` and `per_meal_min` are the same thing under two names;
        # forbid mixing them on a single ingredient so the user sees a clear
        # error rather than silent last-key-wins behaviour.
        if "serving" in spec and "per_meal_min" in spec:
            raise ValueError(
                f"Ingredient {name!r}: set only one of `serving` or `per_meal_min`"
            )
        for key, value in spec.items():
            key_s = str(key)
            if key_s in _INGREDIENT_BOUND_KEYS:
                if key_s == "serving":
                    bound_fields["per_meal_min"] = value
                else:
                    bound_fields[key_s] = value
            else:
                try:
                    macros[key_s] = float(value)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Ingredient {name!r}: macro {key_s!r} must be a number, "
                        f"got {value!r}"
                    ) from e
        out.append(Ingredient(name=str(name), macros=macros))
        if bound_fields:
            inline_bounds[str(name)] = bound_fields
    return out, inline_bounds


def _parse_targets(section: Any) -> list[MacroTarget]:
    if not isinstance(section, dict):
        raise ValueError("`targets` must be a mapping of macro -> spec")
    out: list[MacroTarget] = []
    for name, spec in section.items():
        if isinstance(spec, (int, float)):
            out.append(MacroTarget(name=str(name), value=float(spec)))
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"Target {name!r} must be a number or a mapping")
        out.append(
            MacroTarget(
                name=str(name),
                value=_opt_float(spec.get("value")),
                weight=float(spec.get("weight", 1.0)),
                hard=bool(spec.get("hard", False)),
                lower=_opt_float(spec.get("lower") if spec.get("lower") is not None else spec.get("min")),
                upper=_opt_float(spec.get("upper") if spec.get("upper") is not None else spec.get("max")),
            )
        )
    return out


def _parse_tag_constraints(section: Any) -> dict[str, TagConstraints]:
    if not section:
        return {}
    if not isinstance(section, dict):
        raise ValueError("`tag_constraints` must be a mapping of tag -> spec")
    out: dict[str, TagConstraints] = {}
    for tag, spec in section.items():
        if not isinstance(spec, dict):
            raise ValueError(f"tag_constraints.{tag} must be a mapping")
        macro_min, macro_max = _parse_macro_bounds(spec, f"tag_constraints.{tag}")
        out[str(tag)] = TagConstraints(macro_min=macro_min, macro_max=macro_max)
    return out


def _parse_macro_bounds(
    spec: dict[str, Any], context: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Parse macro_min / macro_max mappings, plus the kcal_max / kcal_min
    shortcut keys, into two dicts keyed by macro name."""
    macro_min: dict[str, float] = {}
    macro_max: dict[str, float] = {}

    raw_min = spec.get("macro_min", {})
    raw_max = spec.get("macro_max", {})
    if raw_min and not isinstance(raw_min, dict):
        raise ValueError(f"{context}.macro_min must be a mapping")
    if raw_max and not isinstance(raw_max, dict):
        raise ValueError(f"{context}.macro_max must be a mapping")
    for k, v in (raw_min or {}).items():
        macro_min[str(k)] = float(v)
    for k, v in (raw_max or {}).items():
        macro_max[str(k)] = float(v)

    if spec.get("kcal_max") is not None:
        if "kcal" in macro_max:
            raise ValueError(
                f"{context}: set only one of `kcal_max` or `macro_max.kcal`"
            )
        macro_max["kcal"] = float(spec["kcal_max"])
    if spec.get("kcal_min") is not None:
        if "kcal" in macro_min:
            raise ValueError(
                f"{context}: set only one of `kcal_min` or `macro_min.kcal`"
            )
        macro_min["kcal"] = float(spec["kcal_min"])

    return macro_min, macro_max


def _parse_meal_library(section: Any) -> list[LibraryMeal]:
    if not isinstance(section, dict):
        raise ValueError("`meal_library` must be a mapping of meal_name -> spec")
    out: list[LibraryMeal] = []
    for name, spec in section.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Meal {name!r} must be a mapping")
        tag = spec.get("tag")
        if not tag:
            raise ValueError(f"Meal {name!r} is missing required field `tag`")
        ings_raw = spec.get("ingredients")
        ingredient_names, ingredient_specs = _parse_meal_ingredients(ings_raw, name)
        macro_min, macro_max = _parse_macro_bounds(spec, f"meal_library.{name}")
        out.append(
            LibraryMeal(
                name=str(name),
                tag=str(tag),
                ingredients=ingredient_names,
                macro_min=macro_min,
                macro_max=macro_max,
                ingredient_specs=ingredient_specs,
            )
        )
    return out


def _parse_meal_ingredients(
    raw: Any, meal_name: str
) -> tuple[tuple[str, ...], dict[str, MealIngredient]]:
    """Accept either a list (simple) or a mapping (rich) of ingredients.

    Mapping values can be:
      - ``null`` / ``{}``                       -> defaults
      - the string ``"main"`` / ``"aux"``        -> main flag (anchor unset)
      - a bare number ``150``                    -> anchor = 150 g
      - a mapping ``{ min, max, main, anchor }``
    """
    if raw is None:
        raise ValueError(f"Meal {meal_name!r}: `ingredients` is required")

    if isinstance(raw, list):
        if not raw:
            raise ValueError(f"Meal {meal_name!r}: `ingredients` must be non-empty")
        names = tuple(str(i) for i in raw)
        return names, {}

    if isinstance(raw, dict):
        if not raw:
            raise ValueError(f"Meal {meal_name!r}: `ingredients` must be non-empty")
        names: list[str] = []
        specs: dict[str, MealIngredient] = {}
        for ing_name, value in raw.items():
            key = str(ing_name)
            names.append(key)
            specs[key] = _parse_meal_ingredient(value, key, meal_name)
        return tuple(names), specs

    raise ValueError(
        f"Meal {meal_name!r}: `ingredients` must be a list or a mapping, "
        f"got {type(raw).__name__}"
    )


def _parse_meal_ingredient(value: Any, ing_name: str, meal_name: str) -> MealIngredient:
    """Accept the various shorthands for a per-meal ingredient spec.

    Forms:
      - ``null`` / ``{}``                -> defaults
      - the string ``"main"`` / ``"aux"`` -> main flag (anchor unset)
      - a bare number (e.g. ``150``)     -> ``anchor=150`` (typical grams)
      - a mapping with any of ``min``, ``max``, ``main``, ``anchor``
    """
    context = f"meal_library.{meal_name}.ingredients.{ing_name}"
    if value is None:
        return MealIngredient()
    if isinstance(value, bool):
        # YAML interprets `yes`/`no`/`true`/`false` as booleans before we see
        # them; reject loudly so a typo doesn't silently become an empty spec.
        raise ValueError(
            f"{context}: bare boolean {value!r} is not a valid spec "
            f"(use a number for an anchor, or a mapping)"
        )
    if isinstance(value, (int, float)):
        return MealIngredient(anchor=float(value))
    if isinstance(value, str):
        flag = value.strip().lower()
        if flag in ("main",):
            return MealIngredient(main=True)
        if flag in ("aux", "auxiliary", ""):
            return MealIngredient()
        raise ValueError(
            f"{context}: unknown shorthand {value!r} "
            f"(expected 'main' or 'aux', a number, or a mapping)"
        )
    if isinstance(value, dict):
        return MealIngredient(
            min=_opt_float(value.get("min")),
            max=_opt_float(value.get("max")),
            main=bool(value.get("main", False)),
            anchor=_opt_float(value.get("anchor")),
        )
    raise ValueError(f"{context}: invalid spec {value!r}")


def _parse_bounds(section: dict[str, dict[str, Any]]) -> dict[str, IngredientBound]:
    if not section:
        return {}
    out: dict[str, IngredientBound] = {}
    for name, spec in section.items():
        step = _opt_float(spec.get("step"))
        if step is not None and step <= 0:
            raise ValueError(f"ingredient_bounds.{name}.step must be > 0")

        # _units fields are convenience aliases: when an ingredient has a
        # `step` (one unit = `step` grams), the user may express bounds as
        # counts of units rather than grams. They get translated to grams
        # internally so the optimizer sees one consistent representation.
        unit_fields = (
            "total_max_units",
            "total_min_units",
            "per_meal_max_units",
            "per_meal_min_units",
        )
        unit_values: dict[str, float | None] = {
            k: _opt_float(spec.get(k)) for k in unit_fields
        }
        if any(v is not None for v in unit_values.values()) and step is None:
            raise ValueError(
                f"ingredient_bounds.{name}: *_units fields require `step` to be set"
            )

        def _resolve(grams_key: str, units_key: str) -> float | None:
            g = _opt_float(spec.get(grams_key))
            u = unit_values[units_key]
            if g is not None and u is not None:
                raise ValueError(
                    f"ingredient_bounds.{name}: set only one of "
                    f"{grams_key!r} or {units_key!r}"
                )
            if u is not None:
                return u * step  # type: ignore[operator]
            return g

        out[str(name)] = IngredientBound(
            total_min=_resolve("total_min", "total_min_units"),
            total_max=_resolve("total_max", "total_max_units"),
            per_meal_min=_resolve("per_meal_min", "per_meal_min_units"),
            per_meal_max=_resolve("per_meal_max", "per_meal_max_units"),
            step=step,
        )
    return out


def _parse_day(section: Any) -> dict[str, str]:
    if not section:
        return {}
    if not isinstance(section, dict):
        raise ValueError("`day` must be a mapping of slot_name -> meal_name")
    return {str(k): str(v) for k, v in section.items()}


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    return float(v)


def _validate(
    ingredients: list[Ingredient],
    meals: list[LibraryMeal],
    targets: list[MacroTarget],
    tag_constraints: dict[str, TagConstraints],
    day: dict[str, str],
    bounds: dict[str, IngredientBound],
) -> None:
    if not ingredients:
        raise ValueError("No ingredients defined.")
    if not meals:
        raise ValueError("No meals defined.")
    if not targets:
        raise ValueError("No targets defined.")

    known_ings = {ing.name for ing in ingredients}
    known_meals = {m.name for m in meals}
    known_tags = {m.tag for m in meals}

    for meal in meals:
        for ing in meal.ingredients:
            if ing not in known_ings:
                raise ValueError(
                    f"Meal {meal.name!r} references unknown ingredient {ing!r}"
                )

    macro_keys = {k for ing in ingredients for k in ing.macros}
    for tgt in targets:
        if tgt.name not in macro_keys:
            raise ValueError(
                f"Target macro {tgt.name!r} is not present on any ingredient "
                f"(known: {sorted(macro_keys)})"
            )

    for tag in tag_constraints:
        if tag not in known_tags:
            # not fatal — a constraint may be defined for a tag not yet used
            pass

    for slot_name, meal_name in day.items():
        if meal_name not in known_meals:
            raise ValueError(
                f"Default day slot {slot_name!r} references unknown meal {meal_name!r}"
            )

    # Warn about ingredient_bounds that refer to ingredients that no longer
    # exist (e.g. left over after renaming `cheddar` to `cheese`). Silent
    # orphans like this are the most common foot-gun: the bound is ignored
    # and the optimizer falls back to the global default cap.
    unknown_bound_keys = sorted(name for name in bounds if name not in known_ings)
    if unknown_bound_keys:
        print(
            "warning: `ingredient_bounds` references unknown ingredient(s) "
            f"(will be ignored): {unknown_bound_keys}",
            file=sys.stderr,
        )
