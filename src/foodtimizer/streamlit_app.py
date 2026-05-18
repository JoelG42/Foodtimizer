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
from foodtimizer.model import Ingredient, MacroTarget, Problem
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


def _render_entries(log: DayLog, logs_dir: str, ingredient_map: dict[str, Ingredient]) -> None:
    """Render the day's entries with per-row delete buttons."""
    st.subheader(f"Today's log · {len(log.entries)} entr{'y' if len(log.entries) == 1 else 'ies'}")

    if not log.entries:
        st.info("Nothing logged yet for this day. Add your first entry above.")
        return

    # Header row
    hdr = st.columns([2, 4, 2, 2, 2, 1])
    for c, label in zip(hdr, ("Time", "Ingredient", "Slot", "Grams", "kcal", "")):
        c.markdown(f"**{label}**")

    for i, entry in enumerate(log.entries):
        cols = st.columns([2, 4, 2, 2, 2, 1])
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
        if ing is not None:
            kcal = entry.grams * ing.amount_for("kcal") / 100.0
            cols[4].write(f"{kcal:.0f}")
        else:
            cols[4].write("?")

        if cols[5].button("🗑", key=f"del_{i}", help="Delete entry"):
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

    _render_entries(log, logs_dir, ingredient_map)

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
