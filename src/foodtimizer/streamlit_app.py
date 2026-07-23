"""Streamlit UI for the food-log tracker.

Launch via the console script::

    foodtimizer-track --config examples/day.yaml --logs-dir logs

or directly::

    streamlit run -m foodtimizer.streamlit_app -- --config examples/day.yaml

The UI is intentionally minimal: pick a date, add (ingredient, grams)
entries, see macro totals against the targets defined in the config.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from foodtimizer import auth
from foodtimizer import storage as store
from foodtimizer.config import _problem_from_dict, load_problem, problem_to_dict
from foodtimizer.editor import (
    INGREDIENT_BOUND_FIELDS,
    all_macro_keys,
    delete_ingredient,
    delete_meal,
    delete_recipe,
    real_ingredients,
    upsert_ingredient,
    upsert_meal,
    upsert_recipe,
)
from foodtimizer.model import (
    Ingredient,
    IngredientBound,
    LibraryMeal,
    MacroTarget,
    MealIngredient,
    Plan,
    Problem,
    Recipe,
    derive_recipe_macros,
)
from foodtimizer.replan import (
    CUSTOM_MEAL_PREFIX,
    LOGGED_MEAL_LABEL,
    CustomSlot,
    diagnose_infeasibility,
    plan_remaining,
)
from foodtimizer.tracker import (
    DayLog,
    compute_totals,
    daylog_from_data,
    daylog_to_data,
    make_entry,
    unknown_ingredients,
)


# Common slot vocabulary; user-typed values are also accepted.
_DEFAULT_SLOTS = ("breakfast", "lunch", "dinner", "snack")


def _parse_args() -> argparse.Namespace:
    """Parse args passed via ``streamlit run ... -- --config ...``."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", default=os.environ.get("FOODTIMIZER_CONFIG", "examples/day.yaml"))
    p.add_argument("--logs-dir", default=os.environ.get("FOODTIMIZER_LOGS_DIR", "logs"))
    known, _ = p.parse_known_args()
    return known


def _target_value(tgt: MacroTarget) -> float | None:
    """The 'goal' to compare totals against: ``value`` if set, else ``lower``.

    For lower-only targets (e.g. ``fibre: { lower: 25 }``) the UI shows it
    as a minimum-to-reach instead of an exact value.
    """
    if tgt.value is not None:
        return tgt.value
    if tgt.lower is not None:
        return tgt.lower
    return None


def _macro_color(pct: float, lower_only: bool) -> str:
    """Pick a tiny semantic colour for the totals row."""
    if lower_only:
        return "green" if pct >= 1.0 else "orange"
    if pct < 0.85:
        return "blue"
    if pct <= 1.05:
        return "green"
    return "red"


def _database_url() -> str | None:
    """Return the DB connection string, if one is configured.

    Looks first in Streamlit secrets (``[database] url = "..."`` — this is how
    you set it on Streamlit Community Cloud), then falls back to the
    ``DATABASE_URL`` environment variable. Returns ``None`` for local file mode.
    """
    try:
        url = st.secrets["database"]["url"]  # type: ignore[index]
        if url:
            return str(url)
    except Exception:  # noqa: BLE001 - no secrets file / key locally is fine
        pass
    return os.environ.get("DATABASE_URL") or None


@st.cache_resource(show_spinner=False)
def _get_engine(url: str):  # type: ignore[no-untyped-def]
    """Create (and cache for the session) a SQLAlchemy engine for ``url``."""
    from sqlalchemy import create_engine

    return create_engine(url, pool_pre_ping=True)


def _configure_storage(args: argparse.Namespace, namespace: str | None) -> None:
    """Select the persistence backend and register it as the active one.

    ``namespace`` isolates data per signed-in user (multi-user mode); it is
    ``None`` in single-user mode.
    """
    url = _database_url()
    if url:
        backend: store.Storage = store.SqlStorage(_get_engine(url), namespace=namespace)
    else:
        backend = store.FileStorage(args.config, args.logs_dir, namespace=namespace)

    # First run for this user/scope: seed from the bundled starter config so
    # they begin with a usable set of ingredients/targets rather than nothing.
    if backend.load_config() is None:
        try:
            seed = problem_to_dict(load_problem(args.config))
            backend.save_config(seed)
        except Exception:  # noqa: BLE001 - an empty seed is acceptable
            pass
    store.configure(backend)


def _render_sidebar(args: argparse.Namespace) -> tuple[Problem, str, str]:
    """Sidebar: config + logs-dir picker, lightweight library stats.

    The YAML is re-read on every Streamlit rerun (no caching), so any UI
    interaction picks up edits you made to ``day.yaml``. The reload button
    below is a no-op that just *forces* a rerun for when you'd rather not
    interact with another widget.
    """
    st.sidebar.header("Settings")
    st.sidebar.toggle(
        "Wide layout (desktop)",
        key="wide_layout",
        help="Off = phone-friendly centered layout. On = full-width desktop layout.",
    )

    # Account controls (only shown when auth is configured).
    auth.render_account_controls()

    # Pick and activate the persistence backend for this run (DB when a
    # database URL is configured, otherwise local files). Data is isolated
    # per signed-in user when auth is on.
    namespace = auth.current_namespace() if auth.auth_configured() else None
    _configure_storage(args, namespace)

    if store.is_file_backed():
        st.sidebar.caption(f"💾 Local files (`{store.location_label()}`)")
    else:
        st.sidebar.caption("🗄️ Connected to the cloud database")

    raw = store.load_config()
    if raw is None:
        st.sidebar.error("No config found in storage.")
        st.stop()
    try:
        problem = _problem_from_dict(raw)
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        st.sidebar.error(f"Could not load config: {e}")
        st.stop()

    st.sidebar.caption(
        f"{len(problem.ingredients)} ingredients · "
        f"{len(problem.targets)} targets · "
        f"{len(problem.meal_library)} meals in library"
    )

    if st.sidebar.button("🔄 Reload", help="Re-read the config from storage now."):
        st.rerun()

    logged = store.logged_dates()
    if logged:
        st.sidebar.caption(f"{len(logged)} day(s) logged so far")

    label = store.location_label()
    return problem, label, label


# ---------------------------------------------------------------------------
# Arise-inspired dashboard: a calorie ring + macro bars as the hero element.
# ---------------------------------------------------------------------------

_COLOR_HEX: dict[str, str] = {
    "green": "#3ddc97",
    "orange": "#f5a623",
    "red": "#ff5a5f",
    "blue": "#5b8def",
}


def _hex(color_name: str) -> str:
    return _COLOR_HEX.get(color_name, _COLOR_HEX["blue"])


def _kcal_budget(problem: Problem) -> float | None:
    """The daily kcal 'budget' for the ring: value, else upper, else lower."""
    for t in problem.targets:
        if t.name == "kcal":
            if t.value is not None:
                return t.value
            if t.upper is not None:
                return t.upper
            return t.lower
    return None


def _calorie_ring_html(eaten: float, budget: float | None, hole: str) -> str:
    """A CSS conic-gradient donut ring.

    ``hole`` is the colour painted in the centre cut-out — pass the app's
    background colour so the ring reads as a hollow donut. Built from plain
    ``div``s + inline styles, which Streamlit's HTML sanitizer renders
    reliably (unlike inline ``<svg>``).
    """
    if budget and budget > 0:
        pct = max(0.0, min(eaten / budget, 1.0))
        remaining = budget - eaten
        color = _COLOR_HEX["green"] if eaten <= budget else _COLOR_HEX["red"]
        center_sub = f"of {budget:.0f} kcal"
        foot = (
            f"{remaining:.0f} kcal left"
            if remaining >= 0
            else f"{-remaining:.0f} kcal over"
        )
    else:
        pct, color = 0.0, _COLOR_HEX["blue"]
        center_sub, foot = "kcal eaten", "no kcal target set"
    deg = pct * 360.0
    track = "rgba(255,255,255,0.12)"
    return f"""
<div style="display:flex;justify-content:center;margin:0.25rem 0 0.6rem;">
  <div style="width:200px;height:200px;border-radius:50%;
              background:conic-gradient({color} {deg:.1f}deg, {track} {deg:.1f}deg 360deg);
              display:flex;align-items:center;justify-content:center;">
    <div style="width:158px;height:158px;border-radius:50%;background:{hole};
                display:flex;flex-direction:column;align-items:center;
                justify-content:center;text-align:center;">
      <div style="font-size:2.5rem;font-weight:800;line-height:1;">{eaten:.0f}</div>
      <div style="font-size:0.8rem;opacity:0.6;margin-top:3px;">{center_sub}</div>
      <div style="font-size:0.8rem;color:{color};font-weight:600;margin-top:5px;">{foot}</div>
    </div>
  </div>
</div>
"""


def _macro_bar_html(name: str, val: float, goal: float | None, color: str) -> str:
    if goal and goal > 0:
        width = max(0.0, min(val / goal, 1.0)) * 100
        figure = f"{val:.0f} / {goal:.0f} g"
    else:
        width, figure = 0.0, f"{val:.0f} g"
    return f"""
<div style="text-align:center;padding:0 3px;">
  <div style="font-size:0.76rem;opacity:0.65;text-transform:capitalize;">{name}</div>
  <div style="font-weight:700;font-size:0.92rem;">{figure}</div>
  <div style="height:7px;border-radius:4px;background:rgba(255,255,255,0.12);
              margin-top:5px;overflow:hidden;">
    <div style="height:7px;width:{width:.0f}%;border-radius:4px;
                background:{color};"></div>
  </div>
</div>
"""


def _render_dashboard(problem: Problem, totals: dict[str, float]) -> None:
    """Calorie ring (kcal vs budget) + a row of macro progress bars."""
    eaten = totals.get("kcal", 0.0)
    # Paint the ring's centre with the app background so it reads as a donut.
    try:
        hole = st.get_option("theme.backgroundColor") or "#0e1117"
    except Exception:  # noqa: BLE001 - get_option is defensive only
        hole = "#0e1117"
    st.markdown(
        _calorie_ring_html(eaten, _kcal_budget(problem), hole),
        unsafe_allow_html=True,
    )

    macro_targets = [t for t in problem.targets if t.name != "kcal"]
    if not macro_targets:
        return
    cols = st.columns(len(macro_targets))
    for col, tgt in zip(cols, macro_targets):
        goal = _target_value(tgt)
        val = totals.get(tgt.name, 0.0)
        lower_only = tgt.value is None and tgt.lower is not None
        color = _hex(_macro_color(val / goal, lower_only)) if goal else _hex("blue")
        col.markdown(
            _macro_bar_html(tgt.name, val, goal, color), unsafe_allow_html=True
        )


# Short, friendly column headers for the most common macros. Anything not
# listed here is shown verbatim (e.g. ``sodium`` -> "sodium").
_MACRO_HEADERS: dict[str, str] = {
    "kcal": "kcal",
    "protein": "P (g)",
    "carbs": "C (g)",
    "fat": "F (g)",
    "fibre": "Fib (g)",
}


def _displayed_macros(problem: Problem) -> list[str]:
    """Macros to show as columns: every targeted macro, in config order.

    Driving column visibility off ``problem.targets`` means the same UI
    auto-adapts when you add/remove targets in the YAML — no code edit.
    """
    return [t.name for t in problem.targets]


def _macro_header(macro: str) -> str:
    return _MACRO_HEADERS.get(macro, macro)


def _format_macro_value(grams: float, ing: Ingredient | None, macro: str) -> str:
    """Format an entry's contribution to one macro for the row display."""
    if ing is None:
        return "?"
    val = grams * ing.amount_for(macro) / 100.0
    # kcal looks better as a whole number; macro grams get 1 decimal.
    return f"{val:.0f}" if macro == "kcal" else f"{val:.1f}"


# Slot order used to lay out the per-meal sections of the log. Anything
# not in this tuple (e.g. a user-defined slot like "second_lunch") shows
# up after these in alphabetical order; entries with no slot at all
# render last under an "Unassigned" header.
_LOG_SLOT_ORDER: tuple[str, ...] = ("breakfast", "lunch", "dinner", "snack")
_UNASSIGNED_SLOT_LABEL = "Unassigned"


def _group_entries_by_slot(
    entries: tuple[LogEntry, ...],
) -> list[tuple[str, list[tuple[int, LogEntry]]]]:
    """Group entries by slot, preserving each entry's original index.

    Returns ``[(slot_label, [(orig_idx, entry), ...]), ...]`` in the
    canonical slot order, then any other slots alphabetically, then the
    unassigned bucket. The original index is what ``DayLog.with_removed``
    and the per-row delete buttons key off, so threading it through here
    keeps delete-by-row working correctly even when groups reorder rows.
    """
    groups: dict[str, list[tuple[int, LogEntry]]] = {}
    for idx, entry in enumerate(entries):
        key = entry.slot or _UNASSIGNED_SLOT_LABEL
        groups.setdefault(key, []).append((idx, entry))

    ordered: list[tuple[str, list[tuple[int, LogEntry]]]] = []
    seen: set[str] = set()
    for slot in _LOG_SLOT_ORDER:
        if slot in groups:
            ordered.append((slot, groups[slot]))
            seen.add(slot)
    extras = sorted(
        s for s in groups if s not in seen and s != _UNASSIGNED_SLOT_LABEL
    )
    for s in extras:
        ordered.append((s, groups[s]))
    if _UNASSIGNED_SLOT_LABEL in groups:
        ordered.append((_UNASSIGNED_SLOT_LABEL, groups[_UNASSIGNED_SLOT_LABEL]))
    return ordered


def _entry_time(entry: LogEntry) -> str:
    """``hh:mm`` from an entry's ISO timestamp, or an em dash if missing."""
    if not entry.eaten_at:
        return "—"
    try:
        return datetime.fromisoformat(entry.eaten_at).strftime("%H:%M")
    except ValueError:
        return entry.eaten_at[:16]


def _macro_summary(
    macros: list[str], values: dict[str, float] | Ingredient | None, grams: float = 0.0
) -> str:
    """One-line ``kcal 520 · P 30 · C 60`` style macro summary."""
    parts: list[str] = []
    for m in macros:
        if isinstance(values, dict):
            num = f"{values.get(m, 0.0):.0f}"
        else:
            num = _format_macro_value(grams, values, m)
        parts.append(f"{_macro_header(m)} {num}")
    return "  ·  ".join(parts)


# Arise-style fixed meal slots with friendly labels. Each is rendered as a
# card with its logged items and an inline "add food" logger.
_MEAL_SLOTS: tuple[tuple[str, str], ...] = (
    ("breakfast", "🌅 Breakfast"),
    ("lunch", "🥗 Lunch"),
    ("dinner", "🍽️ Dinner"),
    ("snack", "🍎 Snack"),
)


def _load_day(on_date: date) -> DayLog:
    """Load a day's log from the active storage backend."""
    return daylog_from_data(on_date, store.load_day(on_date))


def _save_day(log: DayLog) -> None:
    """Persist a day's log to the active storage backend."""
    store.save_day(log.log_date, daylog_to_data(log))


def _render_entry_row(
    orig_idx: int,
    entry: LogEntry,
    ing: Ingredient | None,
    macros: list[str],
    log: DayLog,
    logs_dir: str,
) -> None:
    """One logged item: description + macro caption + delete button."""
    c_main, c_del = st.columns([6, 1])
    c_main.markdown(f"**{entry.grams:.0f} g · {entry.ingredient}**")
    c_main.caption(
        f"🕐 {_entry_time(entry)}  ·  " + _macro_summary(macros, ing, entry.grams)
    )
    # Key off the *original* index so deletes work after grouping.
    if c_del.button("🗑", key=f"del_{orig_idx}", help="Delete entry"):
        _save_day(log.with_removed(orig_idx))
        st.rerun()


def _render_slot_add_form(
    slot: str, log: DayLog, logs_dir: str, ing_names: list[str]
) -> None:
    """Arise-style 'tap a meal, then add food' inline logger for one slot."""
    with st.form(f"add_{slot}", clear_on_submit=True):
        ingredient = st.selectbox(
            "Food",
            options=ing_names,
            index=None,
            placeholder="Search your ingredients…",
            key=f"food_{slot}",
            label_visibility="collapsed",
        )
        c_g, c_btn = st.columns([2, 1])
        grams = c_g.number_input(
            "Grams",
            min_value=0,
            step=1,
            value=None,
            placeholder="grams",
            key=f"grams_{slot}",
            label_visibility="collapsed",
        )
        submit = c_btn.form_submit_button(
            "Add", use_container_width=True, type="primary"
        )
    if not submit:
        return
    if not ingredient:
        st.warning("Pick a food first.")
        return
    if not grams or grams <= 0:
        st.warning("Enter grams.")
        return
    _save_day(
        log.with_added(
            make_entry(ingredient=ingredient, grams=float(grams), slot=slot)
        )
    )
    st.toast(f"Added {grams:g} g {ingredient}", icon="✅")
    st.rerun()


def _render_meal_card(
    slot: str,
    label: str,
    indexed_entries: list[tuple[int, LogEntry]],
    log: DayLog,
    logs_dir: str,
    ingredient_map: dict[str, Ingredient],
    macros: list[str],
    ing_names: list[str],
    *,
    allow_add: bool = True,
) -> None:
    """A single meal card: header with kcal, logged items, and add-food."""
    slot_kcal = 0.0
    for _, e in indexed_entries:
        ing = ingredient_map.get(e.ingredient)
        if ing is not None:
            slot_kcal += e.grams * ing.amount_for("kcal") / 100.0

    with st.container(border=True):
        h_label, h_kcal = st.columns([3, 1])
        h_label.markdown(f"#### {label}")
        h_kcal.markdown(
            f"<div style='text-align:right;font-weight:700;opacity:0.75;"
            f"padding-top:0.55rem;'>{slot_kcal:.0f} kcal</div>",
            unsafe_allow_html=True,
        )

        if not indexed_entries:
            st.caption("Nothing logged yet.")
        for orig_idx, entry in indexed_entries:
            _render_entry_row(
                orig_idx,
                entry,
                ingredient_map.get(entry.ingredient),
                macros,
                log,
                logs_dir,
            )

        if allow_add:
            # Inline (no expander) so tapping the food field is a single tap
            # instead of "open expander, then tap the field".
            _render_slot_add_form(slot, log, logs_dir, ing_names)


def _render_meal_slots(
    log: DayLog,
    logs_dir: str,
    ingredient_map: dict[str, Ingredient],
    macros: list[str],
) -> None:
    """Render the day as Arise-style meal cards, each with inline add-food."""
    grouped: dict[str, list[tuple[int, LogEntry]]] = {}
    for idx, entry in enumerate(log.entries):
        grouped.setdefault(entry.slot or _UNASSIGNED_SLOT_LABEL, []).append(
            (idx, entry)
        )

    ing_names = sorted(ingredient_map)

    for slot, label in _MEAL_SLOTS:
        _render_meal_card(
            slot, label, grouped.get(slot, []), log, logs_dir,
            ingredient_map, macros, ing_names,
        )

    # Any non-standard slots (e.g. "pre_workout") or unassigned legacy
    # entries still get a card so nothing is ever hidden.
    standard = {s for s, _ in _MEAL_SLOTS}
    for slot in sorted(s for s in grouped if s not in standard):
        is_unassigned = slot == _UNASSIGNED_SLOT_LABEL
        label = "🍴 Other" if is_unassigned else f"🍴 {slot}"
        _render_meal_card(
            slot, label, grouped[slot], log, logs_dir,
            ingredient_map, macros, ing_names, allow_add=not is_unassigned,
        )


def _render_totals(problem: Problem, totals: dict[str, float]) -> None:
    """Macro totals vs daily targets, with progress bars."""
    st.subheader("Totals vs targets")

    target_macros = [t.name for t in problem.targets]
    other_macros = [m for m in sorted(totals) if m not in target_macros]

    for tgt in problem.targets:
        goal = _target_value(tgt)
        val = totals.get(tgt.name, 0.0)
        lower_only = tgt.value is None and tgt.lower is not None

        if goal and goal > 0:
            pct = val / goal
            arrow = "≥" if lower_only else "/"
            colour = _macro_color(pct, lower_only)
            # Label + value on one line, the bar full-width underneath, so it
            # reads cleanly on a phone instead of three squished columns.
            st.markdown(
                f"**{tgt.name}**  ·  "
                f":{colour}[**{val:.0f}** {arrow} {goal:.0f} g]  ·  {pct*100:.0f}%"
            )
            st.progress(min(pct, 1.0))
        else:
            st.markdown(f"**{tgt.name}**  ·  {val:.1f}")

    if other_macros:
        with st.expander("Other macros (no target set)"):
            for m in other_macros:
                st.write(f"{m}: {totals[m]:.1f}")


# Common slot -> meal-library tag. Used to populate the per-slot meal
# dropdowns in the "plan rest of day" section. Slots not in this map fall
# back to "any meal in the library".
_SLOT_TO_TAG: dict[str, str] = {
    "breakfast": "breakfast",
    "lunch": "lunch_dinner",
    "dinner": "lunch_dinner",
    "snack": "snack",
}


# Per-slot planning mode. Drives the UI branch (saved-meal dropdown vs
# ingredient multiselect) and what we hand to ``plan_remaining``.
_MODE_SAVED = "Saved meal"
_MODE_CUSTOM = "Custom ingredients"

_SLOT_INSTANCE_SEP = " #"


def _base_slot(label: str) -> str:
    return label.split(_SLOT_INSTANCE_SEP, 1)[0]


def _is_first_instance(label: str) -> bool:
    return (
        _SLOT_INSTANCE_SEP not in label
        or label.endswith(f"{_SLOT_INSTANCE_SEP}1")
    )


def _expand_slot_labels(
    selected_types: list[str], counts: dict[str, int]
) -> list[str]:
    expanded: list[str] = []
    for slot in selected_types:
        n = max(1, int(counts.get(slot, 1)))
        if n == 1:
            expanded.append(slot)
        else:
            expanded.extend(
                f"{slot}{_SLOT_INSTANCE_SEP}{i + 1}" for i in range(n)
            )
    return expanded


def _seed_ingredients_from_base(slot: str, library_meals: dict[str, tuple[str, ...]]) -> None:
    """Streamlit ``on_change`` callback: when the user picks a 'start
    from' meal for a custom slot, populate the ingredient multiselect
    with that meal's ingredient names. Reading session_state directly
    here is the canonical Streamlit way of doing cross-widget updates."""
    base_key = f"plan_base_{slot}"
    ings_key = f"plan_ingredients_{slot}"
    base = st.session_state.get(base_key, "")
    if not base:
        return
    ings = library_meals.get(base)
    if ings is None:
        return
    st.session_state[ings_key] = list(ings)


def _render_planner(
    problem: Problem,
    log: DayLog,
    ingredient_map: dict[str, Ingredient],
) -> None:
    """Render the 'Plan rest of day' section.

    Per slot the user can choose:

    - **Saved meal**: pick from ``meal_library`` (filtered by tag).
    - **Custom ingredients**: hand-pick an ingredient list; optionally
      seed it from an existing meal to remix (e.g. chicken_rice with
      bell pepper instead of broccoli). Synthetic meals carry no
      anchors / main flags, so the LP is freer to hit your macros.
    """
    meals_by_tag: dict[str, list[str]] = {}
    for m in problem.meal_library:
        meals_by_tag.setdefault(m.tag, []).append(m.name)
    # Quick lookup for the "start from" callback below.
    library_meals_ings: dict[str, tuple[str, ...]] = {
        m.name: m.ingredients for m in problem.meal_library
    }

    if not problem.ingredients:
        st.info("No ingredients defined in your config yet.")
        return

    # Default to slots not already represented in the log.
    logged_slots = {e.slot for e in log.entries if e.slot}
    default_slots = [s for s in _SLOT_TO_TAG if s not in logged_slots]

    selected_types = st.multiselect(
        "Which slots still need planning?",
        options=list(_SLOT_TO_TAG.keys()),
        default=default_slots,
        help=(
            "Slots ticked here will be filled by the optimizer. To plan "
            "two of the same kind (e.g. a morning AND an afternoon snack), "
            "bump the per-slot count below."
        ),
    )

    if not selected_types:
        st.caption("Pick at least one slot to enable planning.")
        return

    # Per-slot-type counts let the user plan multiple meals of the same
    # kind in one session (e.g. two snacks). One small number_input per
    # selected type, laid out in a single row to stay compact when the
    # default count of 1 is fine.
    slot_counts: dict[str, int] = {}
    st.caption("How many of each?")
    count_cols = st.columns(max(1, len(selected_types)))
    for col, slot in zip(count_cols, selected_types):
        slot_counts[slot] = int(
            col.number_input(
                slot,
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key=f"plan_count_{slot}",
                help=(
                    f"Number of `{slot}` slots to plan. Each one gets its "
                    "own meal / ingredient picker below."
                ),
            )
        )

    expanded_labels = _expand_slot_labels(selected_types, slot_counts)

    saved_plan: dict[str, str] = {}
    custom_plan: dict[str, CustomSlot] = {}

    all_ingredient_names = sorted(ingredient_map)
    all_meal_names = sorted(library_meals_ings)
    tag_options = [""] + sorted(problem.tag_constraints.keys())

    for label in expanded_labels:
        base = _base_slot(label)
        with st.container(border=True):
            st.markdown(f"#### {label}")
            mode = st.radio(
                f"{label}-mode",
                options=[_MODE_SAVED, _MODE_CUSTOM],
                key=f"plan_mode_{label}",
                horizontal=True,
                label_visibility="collapsed",
            )

            if mode == _MODE_SAVED:
                tag = _SLOT_TO_TAG.get(base)
                candidates = sorted(meals_by_tag.get(tag, [])) if tag else all_meal_names
                if not candidates:
                    st.warning(
                        f"No meals tagged `{tag}` in your library. "
                        "Switch this slot to *Custom ingredients* or add a saved meal."
                    )
                    continue
                default_meal = (
                    problem.default_day.get(base) if _is_first_instance(label) else None
                )
                idx = candidates.index(default_meal) if default_meal in candidates else 0
                chosen = st.selectbox(
                    "Meal",
                    options=candidates,
                    index=idx,
                    key=f"plan_meal_{label}",
                )
                if chosen:
                    saved_plan[label] = chosen
            else:
                # Custom-ingredients branch: optional base + multiselect + tag.
                base_options = [""] + all_meal_names
                st.selectbox(
                    "Start from an existing meal (optional)",
                    options=base_options,
                    index=0,
                    key=f"plan_base_{label}",
                    format_func=lambda x: x or "— none —",
                    help=(
                        "Pre-fill the ingredient list with a saved meal's "
                        "ingredients, then add / remove items below."
                    ),
                    on_change=_seed_ingredients_from_base,
                    args=(label, library_meals_ings),
                )

                picked = st.multiselect(
                    "Ingredients",
                    options=all_ingredient_names,
                    key=f"plan_ingredients_{label}",
                    help=(
                        "Pick any combination of ingredients. The optimizer "
                        "will choose gram amounts to hit your macros, "
                        "respecting per-ingredient bounds (e.g. step sizes)."
                    ),
                )

                # Tag dropdown. "" means "no tag-level constraints applied",
                # which is the most permissive setting and matches the
                # spirit of an ad-hoc meal.
                default_tag = _SLOT_TO_TAG.get(base, "")
                idx = tag_options.index(default_tag) if default_tag in tag_options else 0
                chosen_tag = st.selectbox(
                    "Apply tag constraints",
                    options=tag_options,
                    index=idx,
                    key=f"plan_tag_{label}",
                    format_func=lambda t: t or "(none — fewest restrictions)",
                    help=(
                        "Which tag's kcal cap / macro_min / macro_max should "
                        "apply to this slot. `(none)` skips all tag-level "
                        "constraints; per-ingredient bounds still apply."
                    ),
                )

                if picked:
                    # `chosen_tag` is already either a real tag name or
                    # the empty string ("(none)"). Pass it through verbatim;
                    # CustomSlot.tag == "" means "no tag constraints".
                    custom_plan[label] = CustomSlot(
                        ingredients=tuple(picked),
                        tag=chosen_tag,
                    )

    anchor_weight = st.slider(
        "Anchor weight (recipe-faithfulness)",
        min_value=0.0,
        max_value=1.0,
        value=float(problem.defaults.anchor_weight or 0.05),
        step=0.05,
        help=(
            "0 = pure macro fit. Higher = stick closer to each saved meal's "
            "typical recipe amounts. Custom slots have no anchors, so this "
            "only affects saved-meal slots."
        ),
    )

    use_fallback = st.checkbox(
        "Auto-relax constraints if the strict solve fails",
        value=True,
        help=(
            "If the strict LP says infeasible (e.g. a snack's kcal cap "
            "collides with its forced minimums), retry with per-slot "
            "caps and per-meal floors dropped. Daily total caps, step "
            "sizes and a hard daily kcal ceiling are still respected, "
            "so calories stay below the daily target."
        ),
    )

    if not st.button("🧮 Plan remaining slots", type="primary", use_container_width=True):
        return
    if not saved_plan and not custom_plan:
        st.warning("Pick a meal or some ingredients for at least one slot first.")
        return

    try:
        plan = plan_remaining(
            problem,
            log,
            saved_plan,
            anchor_weight=anchor_weight,
            custom_slots=custom_plan or None,
            fallback=use_fallback,
        )
    except Exception as e:  # noqa: BLE001 - surface to UI
        st.error(f"Could not plan: {e}")
        return

    # On infeasibility, run the heuristic diagnostic so the user sees
    # which hard constraint is conflicting instead of an opaque HiGHS
    # status line. Only on failure to avoid noise on the happy path.
    if plan.status.startswith("FAILED"):
        st.error(f"Optimizer failed: {plan.status}")
        try:
            reasons = diagnose_infeasibility(
                problem, log, saved_plan, custom_slots=custom_plan or None
            )
        except Exception as e:  # noqa: BLE001 - diagnostic must never crash UI
            reasons = [f"(diagnostic itself failed: {e})"]
        if reasons:
            st.markdown("**Likely cause(s):**")
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.caption(
                "No single obvious conflict detected — the infeasibility "
                "is from an interaction between several constraints. "
                "Try removing one ingredient at a time."
            )
        return

    if plan.status.startswith("FALLBACK"):
        st.warning(
            "Strict solve was infeasible, so the planner relaxed "
            "per-slot caps and per-meal floors to give you a best-effort "
            "plan. Daily totals and step sizes are still respected, and "
            "kcal is hard-capped at the daily target."
        )

    _render_combined_plan(plan, problem, ingredient_map)


def _display_meal_name(meal_name: str) -> str:
    """Friendlier label for synthetic ad-hoc meals."""
    if meal_name.startswith(CUSTOM_MEAL_PREFIX):
        return "(custom)"
    return meal_name


def _render_combined_plan(
    plan: Plan,
    problem: Problem,
    ingredient_map: dict[str, Ingredient],
) -> None:
    """Render the result of ``plan_remaining``: optimized slots + the
    whole-day totals vs the user's original targets."""
    if plan.status.startswith("FAILED"):
        st.error(f"Optimizer failed: {plan.status}")
        return

    st.success(plan.status)

    # Group items by slot, separating logged from optimized.
    by_slot: dict[str, list] = {}
    for it in plan.items:
        by_slot.setdefault(it.slot, []).append(it)

    # Render only the optimized slots here (logged entries are already
    # visible in the day's log section above).
    optimized_slots = [s for s, items in by_slot.items() if any(it.meal != LOGGED_MEAL_LABEL for it in items)]
    if not optimized_slots:
        st.info("Nothing optimized — check your slot picks.")
        return

    macros = _displayed_macros(problem)

    for slot in optimized_slots:
        items = [it for it in by_slot[slot] if it.meal != LOGGED_MEAL_LABEL]
        slot_kcal = sum(
            it.grams * ingredient_map[it.ingredient].amount_for("kcal") / 100.0
            for it in items
            if it.ingredient in ingredient_map
        )
        meal_name = _display_meal_name(plan.slot_meals.get(slot, "?"))
        st.markdown(f"**{slot}** · {meal_name} · **{slot_kcal:.0f} kcal**")
        rows = []
        for it in sorted(items, key=lambda x: -x.grams):
            ing = ingredient_map.get(it.ingredient)
            row: dict[str, str] = {
                "ingredient": it.ingredient,
                "grams": f"{it.grams:.0f}",
            }
            for macro in macros:
                row[_macro_header(macro)] = _format_macro_value(it.grams, ing, macro)
            row["anchor"] = f"~{it.anchor:.0f} g" if it.anchor is not None else "—"
            rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Whole-day totals (eaten + planned) vs original targets.
    st.markdown("**Whole-day totals (logged + planned) vs targets**")
    _render_totals(problem, dict(plan.macro_totals))


def _render_date_picker() -> date:
    """Date picker with quick previous/next/today buttons."""
    if "sel_date" not in st.session_state:
        st.session_state["sel_date"] = date.today()

    picked = st.date_input(
        "Date", value=st.session_state["sel_date"], label_visibility="collapsed"
    )
    if picked != st.session_state["sel_date"]:
        st.session_state["sel_date"] = picked
        st.rerun()

    c_prev, c_today, c_next = st.columns(3)
    if c_prev.button("‹ Prev", use_container_width=True):
        st.session_state["sel_date"] = date.fromordinal(
            st.session_state["sel_date"].toordinal() - 1
        )
        st.rerun()
    if c_today.button("Today", use_container_width=True):
        st.session_state["sel_date"] = date.today()
        st.rerun()
    if c_next.button("Next ›", use_container_width=True):
        st.session_state["sel_date"] = date.fromordinal(
            st.session_state["sel_date"].toordinal() + 1
        )
        st.rerun()
    return st.session_state["sel_date"]


# ---------------------------------------------------------------------------
# Library editing pages (ingredient database + meal builder)
# ---------------------------------------------------------------------------


def _optnum(v: object) -> float | None:
    """Coerce a data-editor cell to ``float | None`` (NaN / blank -> None)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and not v.strip():
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _persist_problem(problem: Problem, config_path: str, msg: str) -> None:
    """Save ``problem`` to the active storage backend, then rerun.

    ``config_path`` is kept only for the on-screen label; the actual write
    goes through the configured backend (local file or cloud database). For
    files, a timestamped ``.bak`` is kept on the first overwrite.
    """
    try:
        store.save_config(problem_to_dict(problem))
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        st.error(f"Could not save to {store.location_label()}: {e}")
        return
    st.toast(msg, icon="💾")
    st.rerun()


_BOUND_HELP: dict[str, str] = {
    "step": "Consumed in whole multiples of this many grams (e.g. eggs in 55 g units).",
    "per_meal_min": "Minimum grams in any single meal that uses it (a.k.a. serving).",
    "per_meal_max": "Maximum grams in any single meal.",
    "total_min": "Minimum grams across the whole day.",
    "total_max": "Maximum grams across the whole day (e.g. 60 g whey/day).",
}


def _render_ingredients_page(problem: Problem, config_path: str) -> None:
    """Editable ingredient database: macros per 100 g + optional bounds."""
    st.title("🥕 Ingredient database")
    st.caption(f"Your saved ingredients (macros per 100 g). Saved to `{config_path}`.")

    _render_quick_ingredient_form(problem, config_path)

    if problem.ingredients:
        st.caption(f"{len(problem.ingredients)} ingredients saved.")

    with st.expander("📋 Edit the full table (more columns; best on desktop)"):
        _render_ingredient_table(problem, config_path)


def _render_quick_ingredient_form(problem: Problem, config_path: str) -> None:
    """Compact, vertically-stacked form to add or update one ingredient.

    This is the primary path on mobile — the full spreadsheet editor is
    tucked into an expander below for power editing on a larger screen.
    """
    names = sorted(i.name for i in real_ingredients(problem))
    with st.container(border=True):
        target = st.selectbox(
            "Search an ingredient to edit (leave blank to add a new one)",
            options=names,
            index=None,
            placeholder="Search ingredients…  (blank = add new)",
            key="quick_ing_target",
        )
        editing = problem.ingredient_by_name(target) if target else None
        existing_bound = problem.bounds.get(target) if editing else None

        with st.form("quick_ingredient", clear_on_submit=False):
            name = st.text_input(
                "Name", value=(editing.name if editing else ""),
                placeholder="e.g. greek_yogurt",
            )
            c1, c2 = st.columns(2)
            kcal = c1.number_input(
                "kcal /100g", min_value=0.0, step=10.0,
                value=float(editing.macros.get("kcal", 0.0)) if editing else 0.0,
            )
            protein = c2.number_input(
                "protein /100g", min_value=0.0, step=1.0,
                value=float(editing.macros.get("protein", 0.0)) if editing else 0.0,
            )
            c3, c4 = st.columns(2)
            carbs = c3.number_input(
                "carbs /100g", min_value=0.0, step=1.0,
                value=float(editing.macros.get("carbs", 0.0)) if editing else 0.0,
            )
            fat = c4.number_input(
                "fat /100g", min_value=0.0, step=1.0,
                value=float(editing.macros.get("fat", 0.0)) if editing else 0.0,
            )
            c5, c6 = st.columns(2)
            fibre = c5.number_input(
                "fibre /100g", min_value=0.0, step=0.5,
                value=float(editing.macros.get("fibre", 0.0)) if editing else 0.0,
            )
            per_meal_max = c6.number_input(
                "max per meal (g, 0 = none)", min_value=0.0, step=10.0,
                value=float(existing_bound.per_meal_max)
                if existing_bound and existing_bound.per_meal_max is not None
                else 0.0,
                help="Optional cap on how much of this can go in a single meal.",
            )
            save = st.form_submit_button(
                "💾 Save ingredient", use_container_width=True, type="primary"
            )

        if save:
            if not name.strip():
                st.warning("Enter a name first.")
                return
            macros = {
                "kcal": kcal, "protein": protein,
                "carbs": carbs, "fat": fat, "fibre": fibre,
            }
            bounds: dict[str, float] = {}
            # Preserve any bound fields the quick form doesn't expose.
            if existing_bound is not None:
                for f in INGREDIENT_BOUND_FIELDS:
                    v = getattr(existing_bound, f)
                    if v is not None:
                        bounds[f] = v
            if per_meal_max > 0:
                bounds["per_meal_max"] = per_meal_max
            else:
                bounds.pop("per_meal_max", None)
            try:
                new_problem = upsert_ingredient(
                    problem, name, macros, bounds,
                    original_name=(editing.name if editing else None),
                )
            except ValueError as e:
                st.error(str(e))
                return
            _persist_problem(new_problem, config_path, f"Saved {name.strip()}")


def _render_ingredient_table(problem: Problem, config_path: str) -> None:
    """The full editable spreadsheet of ingredients (macros + all bounds)."""
    st.caption(
        "Add rows at the bottom, edit any cell, then Save. The bound columns "
        "on the right are optional real-life limits the optimizer respects."
    )

    macro_keys = all_macro_keys(problem)
    extra = [m for m in st.session_state.get("extra_macros", []) if m not in macro_keys]
    macro_keys = macro_keys + extra

    with st.container():
        st.markdown("**Add a new nutrient column** (e.g. sodium, sugar)")
        cols = st.columns([4, 1])
        new_macro = cols[0].text_input(
            "Nutrient name", key="new_macro_name", label_visibility="collapsed",
            placeholder="e.g. sodium",
        )
        if cols[1].button("Add column", use_container_width=True):
            nm = new_macro.strip()
            if nm and nm not in macro_keys:
                st.session_state.setdefault("extra_macros", []).append(nm)
                st.rerun()

    bounds = problem.bounds
    rows: list[dict[str, object]] = []
    for ing in real_ingredients(problem):
        row: dict[str, object] = {"name": ing.name}
        for m in macro_keys:
            row[m] = float(ing.macros[m]) if m in ing.macros else None
        b = bounds.get(ing.name)
        for f in INGREDIENT_BOUND_FIELDS:
            row[f] = float(getattr(b, f)) if b and getattr(b, f) is not None else None
        rows.append(row)

    columns = ["name"] + list(macro_keys) + list(INGREDIENT_BOUND_FIELDS)
    df = pd.DataFrame(rows, columns=columns)

    col_config: dict[str, object] = {
        "name": st.column_config.TextColumn("ingredient", required=True),
    }
    for m in macro_keys:
        fmt = "%d" if m == "kcal" else "%.1f"
        col_config[m] = st.column_config.NumberColumn(m, min_value=0.0, format=fmt)
    for f in INGREDIENT_BOUND_FIELDS:
        col_config[f] = st.column_config.NumberColumn(
            f, min_value=0.0, help=_BOUND_HELP.get(f)
        )

    edited = st.data_editor(
        df,
        column_config=col_config,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="ingredient_editor",
    )

    if st.button("💾 Save ingredients", type="primary"):
        _save_ingredient_table(problem, config_path, edited, list(macro_keys))


def _save_ingredient_table(
    problem: Problem,
    config_path: str,
    edited: pd.DataFrame,
    macro_keys: list[str],
) -> None:
    seen: set[str] = set()
    ingredients: list[Ingredient] = []
    new_bounds: dict[str, IngredientBound] = {}

    for record in edited.to_dict("records"):
        name = str(record.get("name") or "").strip()
        if not name:
            continue
        if name in seen:
            st.error(f"Duplicate ingredient name: '{name}'. Names must be unique.")
            return
        seen.add(name)

        macros: dict[str, float] = {}
        for m in macro_keys:
            v = _optnum(record.get(m))
            if v is not None:
                macros[m] = v
        ingredients.append(Ingredient(name=name, macros=macros))

        bound_vals = {f: _optnum(record.get(f)) for f in INGREDIENT_BOUND_FIELDS}
        if any(v is not None for v in bound_vals.values()):
            new_bounds[name] = IngredientBound(
                total_min=bound_vals["total_min"],
                total_max=bound_vals["total_max"],
                per_meal_min=bound_vals["per_meal_min"],
                per_meal_max=bound_vals["per_meal_max"],
                step=bound_vals["step"],
            )

    if not ingredients:
        st.error("You need at least one ingredient.")
        return

    # Recipe-derived ingredients aren't shown in this table; preserve them so
    # they aren't treated as deletions (they're rebuilt from `recipes` anyway).
    recipe_names = {r.name for r in problem.recipes}
    ingredients += [i for i in problem.ingredients if i.name in recipe_names]

    new_problem = replace(problem, ingredients=tuple(ingredients), bounds=new_bounds)

    # Anything dropped from the table is removed from meals too, so we never
    # leave a meal pointing at a non-existent ingredient. Only consider base
    # (non-recipe) ingredients here.
    old_base = {i.name for i in problem.ingredients if i.name not in recipe_names}
    removed = old_base - seen
    for name in removed:
        new_problem = delete_ingredient(new_problem, name)

    if removed:
        affected = sorted(
            m.name
            for m in problem.meal_library
            if any(i in removed for i in m.ingredients)
        )
        if affected:
            st.warning(
                "Removed ingredient(s) were also stripped from these meals: "
                + ", ".join(affected)
            )

    _persist_problem(
        new_problem, config_path, f"Saved {len(ingredients)} ingredients"
    )


_RECIPE_FROM_INGREDIENTS = "From ingredients"
_RECIPE_FROM_TOTALS = "Enter totals manually"


def _render_recipes_page(problem: Problem, config_path: str) -> None:
    """Build composite foods (recipes) out of base ingredients or totals."""
    st.title("🍰 Recipes & composite foods")
    st.caption(
        "Combine ingredients (like mom's banana bread) into a single food. "
        "Foodtimizer works out its calories & macros per 100 g, so you can "
        "just log how many grams you eat — and pick it for any meal. Saved to "
        f"`{config_path}`."
    )

    if not real_ingredients(problem):
        st.info("Add some base ingredients on the 🥕 **Ingredients** page first.")
        return

    recipe_names = [r.name for r in problem.recipes]
    _NEW = "➕ New recipe…"
    choice = st.selectbox(
        "Edit an existing recipe, or create a new one",
        options=[_NEW] + recipe_names,
        key="recipe_select",
    )
    recipe = None if choice == _NEW else problem.recipe_by_name(choice)
    _render_recipe_form(problem, config_path, recipe)


def _render_recipe_preview(
    per100: dict[str, float], total_grams: float, macro_opts: list[str]
) -> None:
    if not per100:
        st.info("Add ingredients (or totals) to see the nutrition preview.")
        return
    st.markdown("**Per 100 g** — this is what gets saved as the food:")
    shown = [m for m in macro_opts if m in per100] or list(per100.keys())
    cols = st.columns(len(shown))
    for col, m in zip(cols, shown):
        col.metric(_macro_header(m), f"{per100[m]:.1f}")
    kcal100 = per100.get("kcal", 0.0)
    st.caption(
        f"Whole batch ≈ {total_grams:.0f} g · "
        f"{kcal100 * total_grams / 100:.0f} kcal total"
    )


def _render_recipe_form(
    problem: Problem, config_path: str, recipe: Recipe | None
) -> None:
    is_edit = recipe is not None
    fkey = recipe.name if is_edit else "__new__"
    ing_map = {i.name: i for i in problem.ingredients}
    base_names = sorted(i.name for i in real_ingredients(problem))
    macro_opts = all_macro_keys(problem)

    name = st.text_input(
        "Recipe name",
        value=(recipe.name if is_edit else ""),
        placeholder="e.g. moms_banana_bread",
        key=f"recipe_name_{fkey}",
    )

    modes = [_RECIPE_FROM_INGREDIENTS, _RECIPE_FROM_TOTALS]
    default_mode = (
        _RECIPE_FROM_TOTALS if (is_edit and not recipe.components) else _RECIPE_FROM_INGREDIENTS
    )
    mode = st.radio(
        "How do you want to define it?",
        options=modes,
        index=modes.index(default_mode),
        horizontal=True,
        key=f"recipe_mode_{fkey}",
    )

    components: dict[str, float] = {}
    total_grams: float | None = None
    total_macros: dict[str, float] = {}

    if mode == _RECIPE_FROM_INGREDIENTS:
        rows = (
            [{"ingredient": k, "grams": float(v)} for k, v in recipe.components.items()]
            if is_edit and recipe.components
            else []
        )
        edited = st.data_editor(
            pd.DataFrame(rows, columns=["ingredient", "grams"]),
            column_config={
                "ingredient": st.column_config.SelectboxColumn(
                    "ingredient", options=base_names, required=True
                ),
                "grams": st.column_config.NumberColumn("grams", min_value=0.0),
            },
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key=f"recipe_components_{fkey}",
        )
        for record in edited.to_dict("records"):
            ing = record.get("ingredient")
            grams = _optnum(record.get("grams"))
            if ing and not (isinstance(ing, float) and math.isnan(ing)) and grams:
                components[str(ing)] = grams

        raw_sum = sum(components.values())
        finished = st.number_input(
            "Finished weight in g (optional)",
            min_value=0,
            step=1,
            value=(int(recipe.total_grams) if is_edit and recipe.total_grams else None),
            placeholder=(f"{raw_sum:.0f} (raw total)" if raw_sum else "e.g. 800"),
            key=f"recipe_finished_{fkey}",
            help=(
                "If the baked result weighs less than the raw ingredients "
                "(water evaporates), enter the final weight. Calories stay the "
                "same, so the per-100 g values go up."
            ),
        )
        if finished and finished > 0:
            total_grams = float(finished)

        preview = Recipe(name=name or "preview", components=components, total_grams=total_grams)
        _render_recipe_preview(
            derive_recipe_macros(preview, ing_map), total_grams or raw_sum, macro_opts
        )
    else:
        c1, c2 = st.columns(2)
        tg = c1.number_input(
            "Total grams (whole batch)", min_value=0, step=1,
            value=(int(recipe.total_grams) if is_edit and recipe.total_grams else None),
            placeholder="e.g. 800", key=f"recipe_tg_{fkey}",
        )
        kcal = c2.number_input(
            "Total kcal (whole batch)", min_value=0, step=10,
            value=(
                int(recipe.total_macros["kcal"])
                if is_edit and recipe.total_macros.get("kcal") is not None
                else None
            ),
            placeholder="e.g. 2400", key=f"recipe_kcal_{fkey}",
        )
        c3, c4, c5 = st.columns(3)
        macro_inputs = {
            "protein": c3.number_input(
                "protein g", min_value=0.0, step=1.0,
                value=(float(recipe.total_macros["protein"]) if is_edit and "protein" in recipe.total_macros else None),
                placeholder="0", key=f"recipe_p_{fkey}",
            ),
            "carbs": c4.number_input(
                "carbs g", min_value=0.0, step=1.0,
                value=(float(recipe.total_macros["carbs"]) if is_edit and "carbs" in recipe.total_macros else None),
                placeholder="0", key=f"recipe_c_{fkey}",
            ),
            "fat": c5.number_input(
                "fat g", min_value=0.0, step=1.0,
                value=(float(recipe.total_macros["fat"]) if is_edit and "fat" in recipe.total_macros else None),
                placeholder="0", key=f"recipe_f_{fkey}",
            ),
        }
        total_grams = float(tg) if tg else None
        kcal_val = _optnum(kcal)
        if kcal_val is not None:
            total_macros["kcal"] = kcal_val
        for macro, raw in macro_inputs.items():
            val = _optnum(raw)
            if val is not None:
                total_macros[macro] = val

        if total_grams and total_macros:
            preview = Recipe(name=name or "preview", total_grams=total_grams, total_macros=total_macros)
            _render_recipe_preview(
                derive_recipe_macros(preview, ing_map), total_grams, macro_opts
            )
        else:
            st.info("Enter total grams and at least the kcal to see a preview.")

    c_save, c_del, _ = st.columns([1, 1, 3])
    if c_save.button("💾 Save recipe", type="primary", key=f"save_recipe_{fkey}"):
        try:
            new_problem = upsert_recipe(
                problem,
                name,
                components=(components if mode == _RECIPE_FROM_INGREDIENTS else None),
                total_grams=total_grams,
                total_macros=(total_macros if mode == _RECIPE_FROM_TOTALS else None),
                original_name=(recipe.name if is_edit else None),
            )
        except ValueError as e:
            st.error(str(e))
            return
        _persist_problem(new_problem, config_path, f"Saved recipe '{name.strip()}'")
    if is_edit and c_del.button("🗑 Delete recipe", key=f"del_recipe_{fkey}"):
        _persist_problem(
            delete_recipe(problem, recipe.name), config_path, f"Deleted '{recipe.name}'"
        )


_TAG_CUSTOM = "➕ custom tag…"
_MEAL_NEW = "➕ New meal…"


def _render_meals_page(problem: Problem, config_path: str) -> None:
    """Meal builder: compose a tagged meal from saved ingredients."""
    st.title("📖 Meal builder")

    if not problem.ingredients:
        st.info("Add some ingredients on the 🥕 **Ingredients** page first.")
        return

    st.caption(
        "Build a meal from your saved ingredients. Pick a tag (which slot it "
        "fits), choose ingredients, and optionally set a typical amount "
        "(*anchor*), mark *main* items, or set hard min/max grams. Saved to "
        f"`{config_path}`."
    )

    meal_names = [m.name for m in problem.meal_library]
    choice = st.selectbox(
        "Edit an existing meal, or create a new one",
        options=[_MEAL_NEW] + meal_names,
        key="meal_select",
    )
    meal = None if choice == _MEAL_NEW else problem.meal_by_name(choice)
    _render_meal_form(problem, config_path, meal)


def _render_meal_form(
    problem: Problem, config_path: str, meal: LibraryMeal | None
) -> None:
    is_edit = meal is not None
    fkey = meal.name if is_edit else "__new__"
    ing_names = sorted(i.name for i in problem.ingredients)
    macro_opts = all_macro_keys(problem)
    known_tags = sorted({m.tag for m in problem.meal_library} | set(problem.tag_constraints))

    name = st.text_input(
        "Meal name", value=(meal.name if is_edit else ""), key=f"meal_name_{fkey}"
    )

    tag_options = known_tags + [_TAG_CUSTOM]
    default_idx = (
        tag_options.index(meal.tag) if is_edit and meal.tag in tag_options else 0
    )
    tag_choice = st.selectbox(
        "Tag (which slot this meal fits)",
        options=tag_options,
        index=default_idx,
        key=f"meal_tag_{fkey}",
        help="Tag-level constraints (e.g. snack kcal cap) apply by this tag.",
    )
    if tag_choice == _TAG_CUSTOM:
        tag = st.text_input("New tag name", key=f"meal_tag_custom_{fkey}").strip()
    else:
        tag = tag_choice

    st.markdown("**Ingredients**")
    ing_rows: list[dict[str, object]] = []
    if is_edit:
        for ing in meal.ingredients:
            spec = meal.ingredient_specs.get(ing)
            ing_rows.append(
                {
                    "ingredient": ing,
                    "role": "main" if (spec and spec.main) else "aux",
                    "anchor (g)": float(spec.anchor) if spec and spec.anchor is not None else None,
                    "min (g)": float(spec.min) if spec and spec.min is not None else None,
                    "max (g)": float(spec.max) if spec and spec.max is not None else None,
                }
            )
    ing_df = pd.DataFrame(
        ing_rows, columns=["ingredient", "role", "anchor (g)", "min (g)", "max (g)"]
    )
    ing_edited = st.data_editor(
        ing_df,
        column_config={
            "ingredient": st.column_config.SelectboxColumn(
                "ingredient", options=ing_names, required=True
            ),
            "role": st.column_config.SelectboxColumn(
                "role",
                options=["aux", "main"],
                help="`main` ingredients get `defaults.main_min` as their floor.",
            ),
            "anchor (g)": st.column_config.NumberColumn(
                "anchor (g)", min_value=0.0,
                help="Typical recipe amount. A soft pull — the optimizer can deviate to hit macros.",
            ),
            "min (g)": st.column_config.NumberColumn("min (g)", min_value=0.0),
            "max (g)": st.column_config.NumberColumn("max (g)", min_value=0.0),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"meal_ings_{fkey}",
    )

    macro_min, macro_max = _render_meal_macro_limits(meal, macro_opts, fkey)

    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("💾 Save meal", type="primary", key=f"save_meal_{fkey}"):
        _save_meal(
            problem, config_path, name, tag, ing_edited, macro_min, macro_max,
            original_name=(meal.name if is_edit else None),
        )
    if is_edit and c2.button("🗑 Delete meal", key=f"del_meal_{fkey}"):
        _persist_problem(
            delete_meal(problem, meal.name), config_path, f"Deleted '{meal.name}'"
        )


def _render_meal_macro_limits(
    meal: LibraryMeal | None, macro_opts: list[str], fkey: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Optional per-meal macro floors/caps (e.g. kcal max for this slot)."""
    macro_min: dict[str, float] = {}
    macro_max: dict[str, float] = {}
    with st.expander("Advanced: meal-level macro limits (optional)"):
        st.caption(
            "Per-slot floors and caps just for this meal. The common one is a "
            "`kcal` maximum. Leave empty to inherit only the tag's constraints."
        )
        min_rows = (
            [{"macro": k, "min": float(v)} for k, v in meal.macro_min.items()]
            if meal
            else []
        )
        max_rows = (
            [{"macro": k, "max": float(v)} for k, v in meal.macro_max.items()]
            if meal
            else []
        )
        cols = st.columns(2)
        with cols[0]:
            st.markdown("Minimums")
            emin = st.data_editor(
                pd.DataFrame(min_rows, columns=["macro", "min"]),
                column_config={
                    "macro": st.column_config.SelectboxColumn("macro", options=macro_opts),
                    "min": st.column_config.NumberColumn("min", min_value=0.0),
                },
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                key=f"meal_min_{fkey}",
            )
        with cols[1]:
            st.markdown("Maximums")
            emax = st.data_editor(
                pd.DataFrame(max_rows, columns=["macro", "max"]),
                column_config={
                    "macro": st.column_config.SelectboxColumn("macro", options=macro_opts),
                    "max": st.column_config.NumberColumn("max", min_value=0.0),
                },
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                key=f"meal_max_{fkey}",
            )
    for r in emin.to_dict("records"):
        macro = str(r.get("macro") or "").strip()
        val = _optnum(r.get("min"))
        if macro and val is not None:
            macro_min[macro] = val
    for r in emax.to_dict("records"):
        macro = str(r.get("macro") or "").strip()
        val = _optnum(r.get("max"))
        if macro and val is not None:
            macro_max[macro] = val
    return macro_min, macro_max


def _save_meal(
    problem: Problem,
    config_path: str,
    name: str,
    tag: str,
    ing_edited: pd.DataFrame,
    macro_min: dict[str, float],
    macro_max: dict[str, float],
    *,
    original_name: str | None,
) -> None:
    specs: dict[str, MealIngredient] = {}
    for record in ing_edited.to_dict("records"):
        ing = record.get("ingredient")
        if not ing or (isinstance(ing, float) and math.isnan(ing)):
            continue
        ing = str(ing)
        if ing in specs:
            st.error(f"Ingredient '{ing}' is listed twice in this meal.")
            return
        specs[ing] = MealIngredient(
            main=(record.get("role") == "main"),
            anchor=_optnum(record.get("anchor (g)")),
            min=_optnum(record.get("min (g)")),
            max=_optnum(record.get("max (g)")),
        )

    try:
        new_problem = upsert_meal(
            problem, name, tag, specs, macro_min, macro_max,
            original_name=original_name,
        )
    except ValueError as e:
        st.error(str(e))
        return

    _persist_problem(new_problem, config_path, f"Saved meal '{name.strip()}'")


def _render_tracker_page(problem: Problem, logs_dir: str) -> None:
    """Arise-inspired daily tracker: calorie ring, macro bars, meal cards."""
    ingredient_map = {i.name: i for i in problem.ingredients}
    macros = _displayed_macros(problem)

    st.title("🍽️ Foodtimizer")

    sel_date = _render_date_picker()
    log = _load_day(sel_date)

    # Warn about stale entries referencing ingredients no longer in the config.
    stale = unknown_ingredients(log, ingredient_map)
    if stale:
        st.warning(
            "These entries reference ingredients that aren't in your config "
            "and are excluded from totals: " + ", ".join(stale)
        )

    totals = compute_totals(log, ingredient_map)
    _render_dashboard(problem, totals)

    st.markdown("### Meals")
    _render_meal_slots(log, logs_dir, ingredient_map, macros)

    with st.expander("📊 Detailed totals vs targets"):
        _render_totals(problem, totals)

    with st.expander("🧮 Plan the rest of the day"):
        st.caption(
            "Given what you've already logged today, the optimizer picks "
            "ingredient amounts for the remaining slots so that the *whole "
            "day* (logged + planned) hits your macro targets. Daily total "
            "caps (e.g. max 60 g whey/day) are credited for what's already "
            "been eaten; per-meal caps still apply to each slot."
        )
        _render_planner(problem, log, ingredient_map)

    if store.is_file_backed():
        st.caption(
            f"Log file: `{Path(logs_dir) / (sel_date.isoformat() + '.json')}` · "
            "edit by hand if you ever need to — it's just JSON."
        )
    else:
        st.caption(f"Saved to {store.location_label()}.")


_PAGE_TRACKER = "🍽️ Tracker"
_PAGE_INGREDIENTS = "🥕 Ingredients"
_PAGE_RECIPES = "🍰 Recipes"
_PAGE_MEALS = "📖 Meals"


def _inject_mobile_css() -> None:
    """Tighten paddings/spacing so the app feels native on a phone screen.

    Streamlit's default desktop chrome wastes a lot of horizontal margin and
    vertical gap; on a narrow viewport that pushes content around. These rules
    are deliberately conservative (paddings + a couple of spacing tweaks) so
    they degrade gracefully if Streamlit's internal markup changes.
    """
    st.markdown(
        """
        <style>
          /* Keep Streamlit's fixed top toolbar above page content, and make
             sure our content starts *below* it (it's ~3.75rem tall) so the top
             navigation isn't clipped. We use !important and cover both the
             legacy `.block-container` class and the newer testid, because
             Streamlit's own stylesheet otherwise overrides our padding. */
          header[data-testid="stHeader"] { z-index: 999; }
          .block-container,
          div[data-testid="stMainBlockContainer"],
          div[data-testid="stAppViewBlockContainer"] {
              padding-top: 4.5rem !important;
              padding-bottom: 4rem !important;
              padding-left: 0.9rem !important;
              padding-right: 0.9rem !important;
          }
          /* Make the top nav radio read like a segmented toolbar. */
          div[role="radiogroup"] {
              gap: 0.35rem;
              flex-wrap: wrap;
              margin-top: 0.25rem;
          }
          /* Buttons inside columns fill their cell for easy thumb taps. */
          .stButton > button { width: 100%; }
          /* Trim the big default gap between stacked blocks a touch. */
          div[data-testid="stVerticalBlock"] { gap: 0.55rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    # Default to the phone-friendly centered layout; a sidebar toggle lets
    # desktop users opt into the wide layout. set_page_config must run first,
    # so we read the persisted preference straight from session_state.
    layout = "wide" if st.session_state.get("wide_layout", False) else "centered"
    st.set_page_config(
        page_title="Foodtimizer",
        page_icon="🍽️",
        layout=layout,
        initial_sidebar_state="collapsed",
    )
    _inject_mobile_css()
    args = _parse_args()

    # When auth is configured, require sign-in before anything else renders.
    auth.require_login()

    problem, config_path, logs_dir = _render_sidebar(args)

    # Top navigation lives in the main column so it's one tap on mobile
    # (the sidebar is collapsed behind the hamburger on small screens).
    page = st.radio(
        "Navigate",
        options=[_PAGE_TRACKER, _PAGE_INGREDIENTS, _PAGE_RECIPES, _PAGE_MEALS],
        key="nav_page",
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == _PAGE_INGREDIENTS:
        _render_ingredients_page(problem, config_path)
    elif page == _PAGE_RECIPES:
        _render_recipes_page(problem, config_path)
    elif page == _PAGE_MEALS:
        _render_meals_page(problem, config_path)
    else:
        _render_tracker_page(problem, logs_dir)


# Streamlit's ``streamlit run`` executes the script with ``__name__ ==
# "__main__"``. Guarding here keeps ``import foodtimizer.streamlit_app``
# side-effect free (for editors, IDEs, doc tools, etc.) without affecting
# the actual app launch.
if __name__ == "__main__":
    main()
