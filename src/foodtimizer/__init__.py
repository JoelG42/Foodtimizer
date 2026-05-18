"""Foodtimizer: linear-programming nutrition optimizer + manual tracker."""

from .config import load_problem
from .model import (
    Defaults,
    Ingredient,
    IngredientBound,
    LibraryMeal,
    MacroTarget,
    MealIngredient,
    Plan,
    PlanItem,
    Problem,
    TagConstraints,
)
from .optimizer import optimize
from .replan import plan_remaining
from .tracker import (
    DayLog,
    LogEntry,
    compute_totals,
    list_logged_dates,
    load_day_log,
    make_entry,
    save_day_log,
    unknown_ingredients,
)

__all__ = [
    "Defaults",
    "Ingredient",
    "IngredientBound",
    "LibraryMeal",
    "MacroTarget",
    "MealIngredient",
    "Plan",
    "PlanItem",
    "Problem",
    "TagConstraints",
    "load_problem",
    "optimize",
    "plan_remaining",
    # Tracker
    "DayLog",
    "LogEntry",
    "compute_totals",
    "list_logged_dates",
    "load_day_log",
    "make_entry",
    "save_day_log",
    "unknown_ingredients",
]
