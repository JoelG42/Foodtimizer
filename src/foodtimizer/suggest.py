"""LLM-backed (and heuristic-backed) anchor suggester.

Given a meal name and the ingredient set you've already declared in your
config, this prints a YAML snippet you can paste into ``meal_library.<meal>``
with sensible per-ingredient ``anchor`` values (typical recipe amounts in
grams, per adult portion).

Usage::

    # Suggest anchors for a meal already defined in your config:
    foodtimizer-anchors config.yaml --meal chicken_rice

    # Suggest anchors for a *new* meal you haven't added yet, by listing the
    # ingredients you intend to use (each must already exist in the config):
    foodtimizer-anchors config.yaml --meal new_pasta --ingredients pasta,ground_beef,tomato_sauce,onion,garlic

    # Offline / deterministic preview (no API key needed):
    foodtimizer-anchors config.yaml --meal chicken_rice --backend heuristic

Backends:

- ``openai`` (default): uses the OpenAI Python SDK with JSON mode. Reads
  ``OPENAI_API_KEY`` from the environment; ``OPENAI_BASE_URL`` is honoured
  so you can point at any OpenAI-compatible endpoint (Anthropic-via-proxy,
  Together, vLLM, Ollama with the OpenAI shim, …). Install with::

      pip install "foodtimizer[llm]"

- ``heuristic``: a zero-dependency fallback that picks anchors based on
  ingredient role and macro density. Useful for tests and offline previews.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .config import load_problem
from .model import Ingredient, LibraryMeal, MealIngredient, Problem


@dataclass(frozen=True)
class _AnchorRequest:
    """A single ask: 'pick gram amounts for these ingredients in this meal'."""

    meal_name: str
    ingredients: tuple[Ingredient, ...]
    main_flags: dict[str, bool]
    portions: int = 1


class AnchorSuggester(Protocol):
    """Anything that can take an ``_AnchorRequest`` and return per-ingredient
    gram anchors. Implementations: :class:`OpenAIChatSuggester`,
    :class:`HeuristicSuggester`."""

    def suggest(self, req: _AnchorRequest) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Heuristic backend
# ---------------------------------------------------------------------------


# Macro-density buckets used by the offline heuristic. Tuned for "looks like a
# real plate" rather than for any specific cuisine.
_HEURISTIC_ROLE_BUCKETS: tuple[tuple[str, float], ...] = (
    # (descriptor, default grams for an aux ingredient in this bucket)
    ("oil_or_dressing", 8.0),
    ("spice", 5.0),
    ("powder", 15.0),
    ("sauce", 30.0),
    ("dairy_dense", 30.0),
    ("dairy_liquid", 120.0),
    ("vegetable_aromatic", 40.0),
    ("vegetable_bulk", 80.0),
    ("fruit", 80.0),
    ("protein", 100.0),
    ("carb_base", 80.0),
)


def _heuristic_role(ing: Ingredient) -> str:
    """Bucket an ingredient using crude macro-density rules.

    Not perfect but good enough for an offline preview. The real backend
    (LLM) will do better on borderline items.
    """
    m = ing.macros
    kcal = float(m.get("kcal", 0.0))
    fat = float(m.get("fat", 0.0))
    protein = float(m.get("protein", 0.0))
    carbs = float(m.get("carbs", 0.0))

    if kcal >= 700 and fat >= 50:
        return "oil_or_dressing"
    if kcal >= 250 and fat >= 15 and protein < 5:
        return "sauce"
    if kcal >= 350 and protein >= 60:
        return "powder"
    if kcal >= 300 and (carbs >= 40 or fat >= 15):
        # cheese, nuts, concentrated dairy
        if protein >= 20:
            return "dairy_dense"
        return "powder"
    if kcal <= 60 and protein <= 4 and carbs <= 10 and fat <= 1:
        return "dairy_liquid" if "fibre" not in m or m.get("fibre", 0.0) < 1 else "vegetable_bulk"
    if protein >= 15 and fat <= 12:
        return "protein"
    if carbs >= 40:
        return "carb_base"
    if kcal <= 60 and carbs <= 12 and fat <= 2:
        return "vegetable_bulk"
    if kcal < 200 and carbs >= 15 and fat <= 1:
        return "fruit"
    if kcal >= 100 and protein <= 3 and fat >= 5:
        return "sauce"
    return "vegetable_aromatic"


_HEURISTIC_DEFAULTS = dict(_HEURISTIC_ROLE_BUCKETS)


class HeuristicSuggester:
    """Offline, deterministic anchor suggester. No API calls."""

    def suggest(self, req: _AnchorRequest) -> dict[str, float]:
        out: dict[str, float] = {}
        for ing in req.ingredients:
            role = _heuristic_role(ing)
            base = _HEURISTIC_DEFAULTS.get(role, 50.0)
            if req.main_flags.get(ing.name, False):
                # bump main items so they actually carry the meal
                base = max(base, 120.0)
            out[ing.name] = round(base * req.portions)
        return out


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------


_PROMPT_SYSTEM = (
    "You are a nutrition assistant that picks realistic per-portion ingredient "
    "amounts (in grams) for a home-cooked meal. You always return strict JSON."
)

_PROMPT_USER_TEMPLATE = """\
Pick typical per-portion gram amounts for the following ingredients in a single
adult portion of a dish called {meal!r}. Use European home-cooking norms.

For each ingredient I give you its macros per 100 g and whether it's flagged as
a "main" ingredient (a structural component that should carry the meal). Aux
ingredients should be smaller amounts (sauces, oils, spices, aromatics).

Hard rules:
- Output VALID JSON, no commentary.
- Top-level key "grams" maps ingredient name -> integer grams.
- Cover EVERY ingredient I list, even small amounts (>= 1 g).
- Reasonable bounds: oils 3-15 g, garlic/spices 3-10 g, sauces 30-150 g,
  proteins 100-250 g, carb bases 60-200 g, vegetables 50-200 g.

Ingredients ({n}):
{ingredient_lines}

Return JSON like:
  {{"grams": {{"chicken_breast": 150, "rice_white": 120, ...}}}}
"""


@dataclass(frozen=True)
class OpenAIChatSuggester:
    """Default backend: hits an OpenAI-compatible chat endpoint and parses
    the returned JSON. Model and base URL are configurable via the
    constructor or environment variables.
    """

    model: str = ""
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.2

    def suggest(self, req: _AnchorRequest) -> dict[str, float]:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised by users
            raise SystemExit(
                "The `openai` package is required for the LLM backend. "
                "Install it with:  pip install \"foodtimizer[llm]\"  "
                "(or switch to --backend heuristic)."
            ) from e

        client = OpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=self.base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )
        model = self.model or os.environ.get("FOODTIMIZER_LLM_MODEL", "gpt-4o-mini")

        user_msg = _PROMPT_USER_TEMPLATE.format(
            meal=req.meal_name,
            n=len(req.ingredients),
            ingredient_lines="\n".join(_describe_ingredient(i, req.main_flags) for i in req.ingredients),
        )

        resp = client.chat.completions.create(
            model=model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PROMPT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        grams_raw = data.get("grams") if isinstance(data, dict) else None
        if not isinstance(grams_raw, dict):
            raise RuntimeError(
                f"LLM returned unexpected JSON shape (no top-level 'grams'): {content!r}"
            )

        out: dict[str, float] = {}
        for ing in req.ingredients:
            v = grams_raw.get(ing.name)
            if v is None:
                raise RuntimeError(f"LLM did not return an amount for {ing.name!r}")
            out[ing.name] = float(v) * req.portions
        return out


def _describe_ingredient(ing: Ingredient, main_flags: dict[str, bool]) -> str:
    flag = "MAIN" if main_flags.get(ing.name, False) else "aux"
    parts = [f"- {ing.name} [{flag}]"]
    for macro in ("kcal", "protein", "carbs", "fat"):
        v = ing.macros.get(macro)
        if v is not None:
            parts.append(f"{macro}={v:g}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Glue: read config -> build request -> render YAML
# ---------------------------------------------------------------------------


def suggest_anchors(
    problem: Problem,
    meal_name: str,
    ingredient_names: Iterable[str] | None,
    *,
    suggester: AnchorSuggester,
    portions: int = 1,
) -> dict[str, float]:
    """High-level entry point reused by the CLI and (eventually) the app.

    If ``ingredient_names`` is None, reads them from the meal in the library.
    Otherwise treats the list as the intended composition of a new/edited
    meal. All names must exist in ``problem.ingredients``.
    """
    ing_map = {i.name: i for i in problem.ingredients}

    if ingredient_names is None:
        try:
            meal = problem.meal_by_name(meal_name)
        except KeyError as e:
            raise SystemExit(
                f"Meal {meal_name!r} not in the config. Either define it "
                f"first, or pass --ingredients to suggest anchors for a new meal."
            ) from e
        ing_seq = list(meal.ingredients)
        main_flags = _main_flags_from_meal(meal)
    else:
        ing_seq = list(ingredient_names)
        main_flags = {}

    missing = [n for n in ing_seq if n not in ing_map]
    if missing:
        raise SystemExit(
            f"Unknown ingredient(s) {missing} — add them to `ingredients:` first."
        )

    req = _AnchorRequest(
        meal_name=meal_name,
        ingredients=tuple(ing_map[n] for n in ing_seq),
        main_flags=main_flags,
        portions=portions,
    )
    return suggester.suggest(req)


def _main_flags_from_meal(meal: LibraryMeal) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for ing_name in meal.ingredients:
        spec: MealIngredient | None = meal.ingredient_specs.get(ing_name)
        out[ing_name] = bool(spec and spec.main)
    return out


def render_yaml_block(meal_name: str, grams: dict[str, float], main_flags: dict[str, bool]) -> str:
    """Render a copy-pasteable YAML snippet for the suggested anchors.

    Items flagged ``main`` get the rich ``{ anchor: N, main: true }`` form;
    everything else uses the bare-number shorthand.
    """
    lines = [f"{meal_name}:", "    tag: lunch_dinner   # adjust to your tag", "    ingredients:"]
    width = max(len(n) for n in grams) if grams else 0
    for name, g in grams.items():
        pad = " " * (width - len(name))
        if main_flags.get(name, False):
            lines.append(f"      {name}:{pad} {{ anchor: {int(round(g))}, main: true }}")
        else:
            lines.append(f"      {name}:{pad} {int(round(g))}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="foodtimizer-anchors",
        description=(
            "Suggest per-ingredient anchor (typical-amount) values for a meal, "
            "using an LLM (default) or an offline heuristic."
        ),
    )
    p.add_argument("config", type=Path, help="Path to YAML config (for the ingredient list).")
    p.add_argument("--meal", required=True, help="Meal name to suggest anchors for.")
    p.add_argument(
        "--ingredients",
        default=None,
        help=(
            "Comma-separated ingredient names for a NEW meal not yet in the "
            "config. If omitted, the meal must already exist in the library."
        ),
    )
    p.add_argument("--portions", type=int, default=1, help="Adult portions to scale to (default 1).")
    p.add_argument(
        "--backend",
        choices=("openai", "heuristic"),
        default="openai",
        help="Which suggester to use. Default openai; heuristic is offline.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("FOODTIMIZER_LLM_MODEL", "gpt-4o-mini"),
        help="Chat model for the openai backend (default: $FOODTIMIZER_LLM_MODEL or gpt-4o-mini).",
    )
    args = p.parse_args(argv)

    problem = load_problem(args.config)

    ing_names: list[str] | None
    if args.ingredients:
        ing_names = [n.strip() for n in args.ingredients.split(",") if n.strip()]
    else:
        ing_names = None

    suggester: AnchorSuggester
    if args.backend == "heuristic":
        suggester = HeuristicSuggester()
    else:
        suggester = OpenAIChatSuggester(model=args.model)

    grams = suggest_anchors(
        problem, args.meal, ing_names, suggester=suggester, portions=args.portions
    )

    if ing_names is None:
        meal = problem.meal_by_name(args.meal)
        main_flags = _main_flags_from_meal(meal)
    else:
        main_flags = {}

    print(
        "# Suggested anchors (paste into meal_library:):\n"
        + render_yaml_block(args.meal, grams, main_flags),
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
