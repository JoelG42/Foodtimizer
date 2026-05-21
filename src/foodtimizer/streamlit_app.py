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
import os
from datetime import date, datetime
from pathlib import Path

import streamlit as st

from foodtimizer.config import load_problem
from foodtimizer.model import Ingredient, MacroTarget, Plan, Problem
from foodtimizer.replan import (
    CUSTOM_MEAL_PREFIX,
    LOGGED_MEAL_LABEL,
    CustomSlot,
    plan_remaining,
)
from foodtimizer.tracker import (
    DayLog,
    compute_totals,
    list_logged_dates,
    load_day_log,
    make_entry,
    save_day_log,
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


def _render_sidebar(args: argparse.Namespace) -> tuple[Problem, str]:
    """Sidebar: config + logs-dir picker, lightweight library stats.

    The YAML is re-read on every Streamlit rerun (no caching), so any UI
    interaction picks up edits you made to ``day.yaml``. The reload button
    below is a no-op that just *forces* a rerun for when you'd rather not
    interact with another widget.
    """
    st.sidebar.header("Settings")
    config_path = st.sidebar.text_input(
        "Config path",
        value=st.session_state.get("config_path", args.config),
        help="YAML file with your ingredients & targets.",
    )
    logs_dir = st.sidebar.text_input(
        "Logs folder",
        value=st.session_state.get("logs_dir", args.logs_dir),
        help="Directory to read/write daily log JSON files.",
    )
    st.session_state["config_path"] = config_path
    st.session_state["logs_dir"] = logs_dir

    try:
        problem = load_problem(config_path)
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        st.sidebar.error(f"Could not load config: {e}")
        st.stop()

    st.sidebar.caption(
        f"{len(problem.ingredients)} ingredients · "
        f"{len(problem.targets)} targets · "
        f"{len(problem.meal_library)} meals in library"
    )

    # Show when this config was last modified so you can see at a glance
    # whether the running UI is reflecting your latest edit.
    try:
        mtime = Path(config_path).stat().st_mtime
        from datetime import datetime as _dt

        st.sidebar.caption(
            f"Config last modified: {_dt.fromtimestamp(mtime).strftime('%H:%M:%S')}"
        )
    except OSError:
        pass

    if st.sidebar.button("🔄 Reload config", help="Re-read the YAML from disk now."):
        st.rerun()

    logged = list_logged_dates(logs_dir)
    if logged:
        st.sidebar.caption(f"{len(logged)} day(s) logged so far")

    return problem, logs_dir


def _render_add_form(problem: Problem, logs_dir: str, log: DayLog) -> None:
    """Render the 'Add an entry' row. On submit, persist + rerun."""
    ing_names = sorted({i.name for i in problem.ingredients})

    with st.form("add_entry", clear_on_submit=True):
        c_name, c_grams, c_slot, c_btn = st.columns([4, 2, 2, 1])
        ingredient = c_name.selectbox(
            "Ingredient",
            options=ing_names,
            index=0 if ing_names else None,
            placeholder="Pick an ingredient",
        )
        grams = c_grams.number_input("Grams", min_value=0.0, step=10.0, value=0.0)
        slot = c_slot.selectbox(
            "Meal (optional)",
            options=("",) + _DEFAULT_SLOTS,
            index=0,
            help="Tag this entry with a meal slot for grouping. Leave blank to skip.",
        )
        submit = c_btn.form_submit_button("➕ Add", use_container_width=True)

    if not submit:
        return
    if not ingredient:
        st.warning("Pick an ingredient first.")
        return
    if grams <= 0:
        st.warning("Grams must be greater than zero.")
        return

    entry = make_entry(
        ingredient=ingredient,
        grams=float(grams),
        slot=(slot or None),
    )
    save_day_log(logs_dir, log.with_added(entry))
    st.toast(f"Added {grams:g} g {ingredient}", icon="✅")
    # Rerun so the new entry shows up everywhere.
    st.rerun()


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


def _render_entries(
    log: DayLog,
    logs_dir: str,
    ingredient_map: dict[str, Ingredient],
    macros: list[str],
) -> None:
    """Render the day's entries with per-row delete buttons.

    Macro columns are taken from ``macros`` (typically every targeted
    macro), so adding ``fat`` / ``fibre`` to the config makes them show
    up here automatically.
    """
    st.subheader(f"Today's log · {len(log.entries)} entr{'y' if len(log.entries) == 1 else 'ies'}")

    if not log.entries:
        st.info("Nothing logged yet for this day. Add your first entry above.")
        return

    # Column weights: [time, ingredient, slot, grams, <one per macro>, delete].
    # The ingredient column stays wide; macros get equal narrow shares.
    base_weights = [2, 4, 2, 2]
    macro_weights = [2] * len(macros)
    weights = base_weights + macro_weights + [1]

    base_labels = ["Time", "Ingredient", "Slot", "Grams"]
    macro_labels = [_macro_header(m) for m in macros]
    labels = base_labels + macro_labels + [""]

    hdr = st.columns(weights)
    for c, label in zip(hdr, labels):
        c.markdown(f"**{label}**")

    for i, entry in enumerate(log.entries):
        cols = st.columns(weights)
        # Time: just hh:mm if we have an ISO timestamp
        if entry.eaten_at:
            try:
                t = datetime.fromisoformat(entry.eaten_at).strftime("%H:%M")
            except ValueError:
                t = entry.eaten_at[:16]
        else:
            t = "—"
        cols[0].write(t)
        cols[1].write(entry.ingredient)
        cols[2].write(entry.slot or "")
        cols[3].write(f"{entry.grams:.0f} g")

        ing = ingredient_map.get(entry.ingredient)
        for j, macro in enumerate(macros):
            cols[4 + j].write(_format_macro_value(entry.grams, ing, macro))

        if cols[-1].button("🗑", key=f"del_{i}", help="Delete entry"):
            new_log = log.with_removed(i)
            save_day_log(logs_dir, new_log)
            st.rerun()


def _render_totals(problem: Problem, totals: dict[str, float]) -> None:
    """Macro totals vs daily targets, with progress bars."""
    st.subheader("Totals vs targets")

    target_macros = [t.name for t in problem.targets]
    other_macros = [m for m in sorted(totals) if m not in target_macros]

    for tgt in problem.targets:
        goal = _target_value(tgt)
        val = totals.get(tgt.name, 0.0)
        lower_only = tgt.value is None and tgt.lower is not None

        cols = st.columns([2, 3, 5])
        cols[0].markdown(f"**{tgt.name}**")
        if goal and goal > 0:
            pct = val / goal
            arrow = "≥" if lower_only else "/"
            colour = _macro_color(pct, lower_only)
            # Streamlit's :color[...] markdown extension; no HTML mixed in.
            cols[1].markdown(
                f":{colour}[**{val:.0f}** {arrow} {goal:.0f} g]  ·  {pct*100:.0f}%"
            )
            cols[2].progress(min(pct, 1.0))
        else:
            cols[1].write(f"{val:.1f}")
            cols[2].write("—")

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

    selected = st.multiselect(
        "Which slots still need planning?",
        options=list(_SLOT_TO_TAG.keys()),
        default=default_slots,
        help="Slots ticked here will be filled by the optimizer.",
    )

    if not selected:
        st.caption("Pick at least one slot to enable planning.")
        return

    saved_plan: dict[str, str] = {}
    custom_plan: dict[str, CustomSlot] = {}

    all_ingredient_names = sorted(ingredient_map)
    all_meal_names = sorted(library_meals_ings)
    tag_options = [""] + sorted(problem.tag_constraints.keys())

    for slot in selected:
        with st.container(border=True):
            st.markdown(f"#### {slot}")
            mode = st.radio(
                f"{slot}-mode",
                options=[_MODE_SAVED, _MODE_CUSTOM],
                key=f"plan_mode_{slot}",
                horizontal=True,
                label_visibility="collapsed",
            )

            if mode == _MODE_SAVED:
                # Saved-meal branch: dropdown filtered by the slot's tag.
                tag = _SLOT_TO_TAG.get(slot)
                candidates = sorted(meals_by_tag.get(tag, [])) if tag else all_meal_names
                if not candidates:
                    st.warning(
                        f"No meals tagged `{tag}` in your library. "
                        "Switch this slot to *Custom ingredients* or add a saved meal."
                    )
                    continue
                default_meal = problem.default_day.get(slot)
                idx = candidates.index(default_meal) if default_meal in candidates else 0
                chosen = st.selectbox(
                    "Meal",
                    options=candidates,
                    index=idx,
                    key=f"plan_meal_{slot}",
                )
                if chosen:
                    saved_plan[slot] = chosen
            else:
                # Custom-ingredients branch: optional base + multiselect + tag.
                base_options = [""] + all_meal_names
                st.selectbox(
                    "Start from an existing meal (optional)",
                    options=base_options,
                    index=0,
                    key=f"plan_base_{slot}",
                    format_func=lambda x: x or "— none —",
                    help=(
                        "Pre-fill the ingredient list with a saved meal's "
                        "ingredients, then add / remove items below."
                    ),
                    on_change=_seed_ingredients_from_base,
                    args=(slot, library_meals_ings),
                )

                picked = st.multiselect(
                    "Ingredients",
                    options=all_ingredient_names,
                    key=f"plan_ingredients_{slot}",
                    help=(
                        "Pick any combination of ingredients. The optimizer "
                        "will choose gram amounts to hit your macros, "
                        "respecting per-ingredient bounds (e.g. step sizes)."
                    ),
                )

                # Tag dropdown. "" means "no tag-level constraints applied",
                # which is the most permissive setting and matches the
                # spirit of an ad-hoc meal.
                default_tag = _SLOT_TO_TAG.get(slot, "")
                idx = tag_options.index(default_tag) if default_tag in tag_options else 0
                chosen_tag = st.selectbox(
                    "Apply tag constraints",
                    options=tag_options,
                    index=idx,
                    key=f"plan_tag_{slot}",
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
                    custom_plan[slot] = CustomSlot(
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
        )
    except Exception as e:  # noqa: BLE001 - surface to UI
        st.error(f"Could not plan: {e}")
        return

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

    cols = st.columns([1, 3, 1, 1])
    if cols[0].button("← prev"):
        st.session_state["sel_date"] = date.fromordinal(
            st.session_state["sel_date"].toordinal() - 1
        )
        st.rerun()
    picked = cols[1].date_input("Date", value=st.session_state["sel_date"], label_visibility="collapsed")
    if picked != st.session_state["sel_date"]:
        st.session_state["sel_date"] = picked
        st.rerun()
    if cols[2].button("next →"):
        st.session_state["sel_date"] = date.fromordinal(
            st.session_state["sel_date"].toordinal() + 1
        )
        st.rerun()
    if cols[3].button("today"):
        st.session_state["sel_date"] = date.today()
        st.rerun()
    return st.session_state["sel_date"]


def main() -> None:
    st.set_page_config(page_title="Foodtimizer Tracker", layout="wide")
    args = _parse_args()

    problem, logs_dir = _render_sidebar(args)
    ingredient_map = {i.name: i for i in problem.ingredients}

    st.title("Foodtimizer · daily tracker")

    sel_date = _render_date_picker()

    log = load_day_log(logs_dir, sel_date)

    # Warn about stale entries referencing ingredients no longer in the config.
    stale = unknown_ingredients(log, ingredient_map)
    if stale:
        st.warning(
            "These entries reference ingredients that aren't in your config "
            "and are excluded from totals: " + ", ".join(stale)
        )

    _render_add_form(problem, logs_dir, log)

    totals = compute_totals(log, ingredient_map)
    _render_totals(problem, totals)

    _render_entries(log, logs_dir, ingredient_map, _displayed_macros(problem))

    with st.expander("🧮 Plan the rest of the day", expanded=False):
        st.caption(
            "Given what you've already logged today, the optimizer picks "
            "ingredient amounts for the remaining slots so that the *whole "
            "day* (logged + planned) hits your macro targets. Daily total "
            "caps (e.g. max 60 g whey/day) are credited for what's already "
            "been eaten; per-meal caps still apply to each slot."
        )
        _render_planner(problem, log, ingredient_map)

    st.caption(
        f"Log file: `{Path(logs_dir) / (sel_date.isoformat() + '.json')}` · "
        "edit by hand if you ever need to — it's just JSON."
    )


# Streamlit's ``streamlit run`` executes the script with ``__name__ ==
# "__main__"``. Guarding here keeps ``import foodtimizer.streamlit_app``
# side-effect free (for editors, IDEs, doc tools, etc.) without affecting
# the actual app launch.
if __name__ == "__main__":
    main()
