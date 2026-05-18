"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_problem
from .model import Plan, Problem
from .optimizer import optimize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foodtimizer",
        description=(
            "Optimize a daily nutrition plan from a library of tagged meals. "
            "You pick one meal per slot; the optimizer decides the gram amounts."
        ),
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file.")
    parser.add_argument(
        "--list-meals",
        action="store_true",
        help="List all meals in the library, grouped by tag, then exit.",
    )
    parser.add_argument("--breakfast", help="Meal to use for the breakfast slot.")
    parser.add_argument("--lunch", help="Meal to use for the lunch slot.")
    parser.add_argument("--dinner", help="Meal to use for the dinner slot.")
    parser.add_argument("--snack", help="Meal to use for the snack slot.")
    parser.add_argument(
        "--slot",
        action="append",
        default=[],
        metavar="NAME=MEAL",
        help="Assign an arbitrary slot to a library meal. Can be passed multiple times.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=0,
        help="Decimal places for gram amounts in output (default: 0).",
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=None,
        help=(
            "Override `defaults.anchor_weight`. Pass 0 to disable anchor "
            "pulls (pure macro fit). Larger values keep amounts closer to "
            "each meal's anchor (typical recipe amount)."
        ),
    )
    args = parser.parse_args(argv)

    problem = load_problem(args.config)

    if args.list_meals:
        print(format_meal_library(problem))
        return 0

    day_plan = _build_day_plan(problem, args)
    plan = optimize(problem, day_plan, anchor_weight=args.anchor_weight)
    print(format_plan(problem, plan, precision=args.precision))
    return 0 if plan.status and not plan.status.startswith("FAILED") else 1


def _build_day_plan(problem: Problem, args: argparse.Namespace) -> dict[str, str]:
    day: dict[str, str] = dict(problem.default_day)
    for slot_name in ("breakfast", "lunch", "dinner", "snack"):
        value = getattr(args, slot_name)
        if value:
            day[slot_name] = value
    for entry in args.slot:
        if "=" not in entry:
            raise SystemExit(f"--slot must be NAME=MEAL, got {entry!r}")
        name, _, meal = entry.partition("=")
        name, meal = name.strip(), meal.strip()
        if not name or not meal:
            raise SystemExit(f"--slot must be NAME=MEAL, got {entry!r}")
        day[name] = meal
    return day


def format_meal_library(problem: Problem) -> str:
    by_tag: dict[str, list[str]] = {}
    for meal in problem.meal_library:
        by_tag.setdefault(meal.tag, []).append(meal.name)
    lines: list[str] = ["Available meals:"]
    for tag in sorted(by_tag):
        tc = problem.tag_constraints.get(tag)
        cap_kcal = tc.macro_max.get("kcal") if tc else None
        cap = f"  (kcal_max={cap_kcal:.0f})" if cap_kcal is not None else ""
        lines.append(f"  [{tag}]{cap}")
        for name in sorted(by_tag[tag]):
            meal = problem.meal_by_name(name)
            lines.append(f"    - {name}: {', '.join(meal.ingredients)}")
    return "\n".join(lines)


def format_plan(problem: Problem, plan: Plan, *, precision: int = 0) -> str:
    if plan.status.startswith("FAILED"):
        return f"Optimization failed: {plan.status}"

    lines: list[str] = []
    lines.append(f"Status: {plan.status}")
    lines.append(f"Objective (weighted deviation): {plan.objective_value:.3f}")
    lines.append("")

    ing_map = {ing.name: ing for ing in problem.ingredients}
    by_slot = plan.items_by_slot()
    g_fmt = f"{{:>7.{precision}f}} g"

    for slot, meal_name in plan.slot_meals.items():
        items = by_slot.get(slot, [])
        slot_kcal = sum(
            it.grams * ing_map[it.ingredient].amount_for("kcal") / 100.0 for it in items
        )
        meal = problem.meal_by_name(meal_name)
        cap_text = ""
        from .optimizer import _slot_kcal_cap

        cap = _slot_kcal_cap(meal, problem.tag_constraints)
        if cap is not None:
            cap_text = f"  (cap {cap:.0f})"
        lines.append(f"[{slot}] {meal_name} ({meal.tag})  {slot_kcal:.0f} kcal{cap_text}")
        if not items:
            lines.append("    (empty)")
        else:
            for it in sorted(items, key=lambda x: -x.grams):
                ing = ing_map[it.ingredient]
                kcal = it.grams * ing.amount_for("kcal") / 100.0
                prot = it.grams * ing.amount_for("protein") / 100.0
                anchor_note = ""
                if it.anchor is not None:
                    delta = it.grams - it.anchor
                    sign = "+" if delta >= 0 else ""
                    anchor_note = f"   [~{it.anchor:.0f} g, {sign}{delta:.0f}]"
                lines.append(
                    f"    {it.ingredient:<20s} "
                    + g_fmt.format(it.grams)
                    + f"   {kcal:>5.0f} kcal   {prot:>5.1f} g protein"
                    + anchor_note
                )
        lines.append("")

    lines.append("Totals")
    for macro in sorted(plan.macro_totals):
        target_line = ""
        for tgt in problem.targets:
            if tgt.name == macro and tgt.value is not None:
                dev = plan.target_deviations.get(macro, 0.0)
                sign = "+" if dev >= 0 else ""
                target_line = f"   target {tgt.value:.0f}  ({sign}{dev:.1f})"
                break
        lines.append(f"    {macro:<10s} {plan.macro_totals[macro]:>8.1f}{target_line}")

    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
