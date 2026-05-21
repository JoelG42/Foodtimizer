# Foodtimizer

Two tools that share one ingredient library:

- **`foodtimizer-track`** — a Streamlit tracker: type in grams of what you ate,
  see live totals of kcal / protein / carbs / fat / fibre against your daily
  targets. Local, persistent, one JSON file per day.
- **`foodtimizer`** — a linear-programming planner: given your library of
  meals and macro targets, decide the gram amount of every ingredient in
  every meal.

Both read the same `examples/day.yaml` config. Tracker = "what I ate".
Planner = "what I should eat". And they combine: see *Plan the rest of the
day* below.

---

## Tracker quickstart

```powershell
# One-time install (also picks up streamlit):
.\.venv\Scripts\python.exe -m pip install -e ".[app]"

# Launch the UI:
.\.venv\Scripts\foodtimizer-track.exe --config examples\day.yaml --logs-dir logs
```

Streamlit will print a `http://localhost:8501` URL and open your browser. You
get:

- a date picker (← prev / next → / today),
- an *Add an entry* form (ingredient dropdown, grams, optional meal slot),
- the day's log with delete buttons,
- live totals vs each target, with progress bars.

Logs are stored as plain JSON files in `logs/YYYY-MM-DD.json`. You can edit
them by hand, copy them to another machine, or check them into a private
repo for backup. The schema:

```json
{
  "date": "2026-05-18",
  "entries": [
    {"ingredient": "chicken_breast", "grams": 150,
     "slot": "lunch", "note": null, "eaten_at": "2026-05-18T13:30:00"}
  ]
}
```

Environment variables override the defaults:

- `FOODTIMIZER_CONFIG` — path to the YAML config.
- `FOODTIMIZER_LOGS_DIR` — directory for the per-day JSON files.

So a no-args launch in the right shell is just `foodtimizer-track`.

If your PowerShell blocks venv activation, skip activation and call the
executable directly via `.\.venv\Scripts\foodtimizer-track.exe …` as above.

### Plan the rest of the day

Inside the tracker, expand **🧮 Plan the rest of the day**. For each slot
that's still to come you pick one of two modes:

- **Saved meal** — choose a meal from your `meal_library`. Anchors, main
  flags and tag-level constraints all apply, so the result looks
  recipe-shaped.
- **Custom ingredients** — hand-pick any list of ingredients. Optionally
  *start from* an existing saved meal to pre-fill the list, then add or
  remove items (e.g. *chicken_rice* with bell pepper instead of
  broccoli — no YAML edit needed). Custom slots carry no anchors and no
  `main` flags, so the optimizer has maximum freedom to choose grams
  that hit your macros — useful for ad-hoc snacks (a toast + deli
  chicken sandwich) where a fixed recipe doesn't really exist.

You can mix freely: dinner from a saved meal, snack from a custom list,
all in one click.

The whole-day picture is what the optimizer aims for. It reduces each
daily target and each per-ingredient daily total cap by what you've
already eaten, then runs the LP on the remaining slots only:

- If you ate 30 g whey at breakfast and your daily cap is 60 g, the
  remaining slots get at most 30 g more.
- If you already crushed your 135 g protein goal, the optimizer plans a
  light remainder — soft targets stay soft, the LP never goes infeasible.
- Per-meal caps (e.g. *lunch ≤ 750 kcal*) are unchanged: they're per-slot
  rules, not daily ones.

Programmatic equivalent:

```python
from foodtimizer import (
    CustomSlot, DayLog, load_problem, load_day_log, plan_remaining,
)

problem = load_problem("examples/day.yaml")
log = load_day_log("logs", date.today())

plan = plan_remaining(
    problem, log,
    {"lunch": "chicken_rice"},                       # saved meals
    custom_slots={                                   # ad-hoc meals
        "snack": CustomSlot(
            ingredients=("bread", "deli_chicken", "tomato"),
            # tag defaults to the slot-name's tag ("snack" here);
            # pass tag="" to skip tag-level constraints entirely.
        ),
    },
)
```

---

## Planner: how it works

1. You build a **library of meals**. Each meal is a list of ingredients plus
   a **tag** (`breakfast`, `lunch_dinner`, `snack`, …).
2. You define **tag-level constraints** (e.g. *breakfast: kcal_max 400*).
3. You define your **macro targets** (kcal, protein, carbs, fat, …).
4. At run time you **pick one meal per slot** (breakfast / lunch / dinner / snack)
   — either by setting `day:` in the YAML or via CLI flags.
5. Foodtimizer solves an LP that returns the **gram amount** of every
   ingredient in every chosen meal so your macro targets are met as closely as
   possible while respecting all hard constraints.

## Math

For every chosen ``(slot s, ingredient j)`` pair we introduce a continuous
variable ``x[s, j] ≥ 0`` (grams). For every *soft* macro target ``m`` we add
non-negative slack variables ``s_m⁺, s_m⁻`` and minimize

    Σ_m  w_m · (s_m⁺ + s_m⁻)

subject to

- Macro identities:  ``Σ_{s,j} x[s,j] · macro_m(j) / 100 − s_m⁺ + s_m⁻ = T_m``
- Per-slot kcal cap (meal's own ``kcal_max`` if set, else the tag's):
  ``Σ_j x[s,j] · kcal(j) / 100 ≤ kcal_cap(s)``
- Optional macro `lower` / `upper` hard bounds.
- Optional per-ingredient gram bounds (per slot, totals).

Hard targets (`hard: true`) skip the slack and become equality constraints.
Because every soft target has slack, the problem is always feasible — you get
the closest possible plan, never "INFEASIBLE".

Solver: HiGHS via `scipy.optimize.linprog`.

## Install

```bash
pip install -e .
```

## Run

Use the default day plan from the YAML:

```bash
foodtimizer examples/day.yaml
```

List the library:

```bash
foodtimizer examples/day.yaml --list-meals
```

Override slots at the CLI:

```bash
foodtimizer examples/day.yaml --lunch salmon_potato --dinner gnocchi_bolognese
```

Arbitrary slot names (e.g. multiple snacks):

```bash
foodtimizer examples/day.yaml --slot snack_pm=protein_shake --slot snack_late=almonds_apple
```

## Config

See `examples/day.yaml` for a full annotated example. Sketch:

```yaml
ingredients:
  # Macros (per 100 g) + real-life constraints right next to each item.
  # Recognised bound keys: step, serving (= per_meal_min), per_meal_min/max,
  # per_meal_min/max_units, total_min/max[_units].
  banana:         { kcal: 89,  protein: 1.1,  carbs: 23.0, fat: 0.3,  fibre: 2.6, serving: 80 }
  eggs:           { kcal: 155, protein: 13.0, carbs: 1.1,  fat: 11.0, fibre: 0.0,
                    step: 55, per_meal_max_units: 4, total_max_units: 4 }
  chicken_breast: { kcal: 97,  protein: 20.0, carbs: 0.0,  fat: 0.5,  fibre: 0.0,
                    step: 150, per_meal_max_units: 1 }     # 450 g pack = 3 × 150 g
  tomato_sauce:   { kcal: 27,  protein: 1.7,  carbs: 4.1,  fat: 0.5,  fibre: 1.5,
                    serving: 100, per_meal_max: 200 }
  # ...

targets:
  kcal:    { value: 1800, weight: 1.0 }
  protein: { value: 135,  weight: 50.0 }   # high weight => matched tightly
  carbs:   { value: 225,  weight: 5.0 }
  fat:     { value: 60,   weight: 1.0 }
  fibre:   { lower: 25 }

tag_constraints:
  breakfast:    { kcal_max: 400 }
  lunch_dinner: { kcal_max: 800, macro_min: { carbs: 30, protein: 25 } }
  snack:        { kcal_max: 300 }

meal_library:
  # Simple list form: every ingredient cascades from inline bounds + defaults.
  oats_breakfast:
    tag: breakfast
    ingredients: [oats, banana, whey_protein, milk, cacao]
  # Rich form: label structural items as `main` to hit `defaults.main_min`.
  gnocchi_bolognese:
    tag: lunch_dinner
    ingredients:
      gnocchi:      main
      ground_beef:  main
      tomato_sauce: {}
      onion:        {}
      tomato_paste: {}
      garlic:       {}

# Optional. Anything declared here overrides the inline bounds *per field*.
# ingredient_bounds:
#   chicken_breast: { total_max_units: 2 }   # cap daily chicken at 2 portions

day:
  breakfast: oats_breakfast
  lunch:     gnocchi_bolognese
```

### Where do bounds live?

You have three places to influence per-(slot, ingredient) gram amounts; the
optimizer combines them as **max-of-mins, min-of-maxes**:

1. **Inline on the ingredient** (preferred for "intrinsic" facts: step,
   typical serving, daily cap).
2. **`ingredient_bounds:` section** — overrides the inline value field by
   field if you need a one-off tweak.
3. **Per-meal `ingredient_specs`** (the rich `meal_library` form) — only
   tightens, never relaxes. Useful when one specific meal needs a bigger
   chicken portion than the global default.

### Anchors: keep amounts recipe-shaped

Hard bounds keep the plan *legal*, but they don't keep it *typical*. To
make a bolognese look like a bolognese, attach an **anchor** (typical gram
amount) to each ingredient in a meal:

```yaml
defaults:
  anchor_weight: 0.05   # global pull strength

meal_library:
  gnocchi_bolognese:
    tag: lunch_dinner
    ingredients:
      gnocchi:      { anchor: 150, main: true }
      ground_beef:  { anchor: 200, main: true }
      tomato_sauce: 100        # bare number = anchor
      onion:         50
      garlic:         5
```

The optimizer adds an L1 penalty `anchor_weight * |x - anchor|` for each
ingredient with an anchor, so amounts get pulled toward typical values but
can still drift to satisfy macros. Anchors are **never hard constraints** —
use `min` / `max` / `step` for those.

In the plan output every anchored ingredient is annotated with
`[~target g, ±delta]`, so you can see at a glance how far each item was
pushed off-recipe by macro pressure.

### Bootstrapping anchors with an LLM

You don't have to invent typical amounts by hand. The `foodtimizer-anchors`
command asks an LLM for them, given the macros you've already entered:

```bash
pip install "foodtimizer[llm]"
export OPENAI_API_KEY=sk-...

foodtimizer-anchors examples/day.yaml --meal gnocchi_bolognese
# # Suggested anchors (paste into meal_library:):
# gnocchi_bolognese:
#     tag: lunch_dinner   # adjust to your tag
#     ingredients:
#       gnocchi:      { anchor: 160, main: true }
#       ground_beef:  { anchor: 180, main: true }
#       tomato_sauce: 110
#       carrots:       80
#       ...
```

For a meal you haven't added yet, pass the ingredient list:

```bash
foodtimizer-anchors config.yaml --meal new_pasta \
  --ingredients pasta,ground_beef,tomato_sauce,onion,garlic
```

Offline preview (no API key) via a coarse macro-density heuristic:

```bash
foodtimizer-anchors config.yaml --meal chicken_rice --backend heuristic
```

`OPENAI_BASE_URL` is honoured so you can point at any OpenAI-compatible
endpoint (local model via Ollama's OpenAI shim, vLLM, Together, …).


## Tests

```bash
pip install -e ".[dev]"
pytest
```
