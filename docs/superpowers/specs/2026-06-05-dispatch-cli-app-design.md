# Dispatch CLI App — Design

**Date:** 2026-06-05
**Status:** Approved (pending spec review)

## Problem

The dispatch workflow lives in notebooks and two standalone scripts —
`run_dispatch.py` (916 lines) and `get_date_results.py` (589 lines). The two
scripts share ~900 lines of near-identical logic: download XM files, load ~10
CSVs, parse the OFEI text file, build the pyomo set/param dicts, solve, extract
MPO + dispatch, save CSVs. The duplication has already diverged (e.g. the
MILP→LP pricing fix and the dedup landed in different places), and the
evaluation metrics live only in a notebook.

Goal: a comprehensive, headless app driven by a CLI `run` command, with the
shared logic extracted once into the `app` package so it is testable and has a
single source of truth.

## Scope

In scope:
- Extract the shared data-load + case-build logic into the `app` package.
- A Typer CLI exposing a single `run` command (date / range / month / all).
- Auto-evaluation against XM actuals after solving (toggle off with `--no-eval`).
- Migrate the two existing scripts to thin callers of the new pipeline as a
  regression safety net.

Out of scope (deferred):
- `evaluate`, `fetch`, `compare` subcommands (may come later; metrics logic is
  reused by `run` for now).
- Plotting — removed from the run path, stays in notebooks / optional later.
- The bulk-CSV ETL that produces `data/dispo_declarada.csv`,
  `data/ofertas.csv`, etc. (currently `data_fetcher.ipynb`). The app *consumes*
  those CSVs; producing them is a separate concern left as-is for now.

## Architecture

```
app/
  data/
    download.py     # save_file + ensure_data_for_date(date) (from utils/misc.py)
    loaders.py      # read root CSVs: dispo, ofertas, demanda, agc, params, precio_bolsa
    ofei.py         # parse OFEI txt -> PAP prices, MO lines, combined-cycle configs
    actuals.py      # load XM preideal dispatch + price txt (the comparison target)
  pipeline/
    case_builder.py # (date, DispatchConfig) -> (set_data, param_data)  [the dup logic, ONCE]
    results.py      # pull MPO(dual) + dispatch(pout) from solved model; write CSVs
    runner.py       # ensure_data -> build -> solve -> extract -> save [-> evaluate]
  model/            # existing pyomo UnitCommitmentModel — unchanged
  utils/
    metrics.py      # already built — unchanged
  cli.py            # Typer app, `run` command
  __main__.py       # enables `python -m app run ...`
```

### Unit responsibilities

- **`data/download.py`** — `save_file(file_type, date)` (moved from
  `utils/misc.py`, which currently holds it) plus
  `ensure_data_for_date(date)`: download the per-day XM files into `data/{date}/`
  only if the folder is absent (current behavior in `run_dispatch.py:33-38`).
- **`data/loaders.py`** — pure read functions for the root CSVs, each returning
  a typed DataFrame; centralizes the `precio_bolsa * 1e3` and `demanda * 1e-3`
  unit conversions that are currently scattered, so units are applied in exactly
  one place.
- **`data/ofei.py`** — parse the `OFEI{MMDD}.txt` file into PAP startup prices,
  MO lines, and combined-cycle configurations (currently inline in both
  scripts, ~lines 96-330 of `get_date_results.py`).
- **`data/actuals.py`** — load `data/preideal_dispatch/{date}.txt` and
  `data/preideal_price/{date}.txt` (XM's actual predispatch generation and
  marginal price) — the evaluation targets.
- **`pipeline/case_builder.py`** — the single function that turns a date +
  `DispatchConfig` into the `set_data` / `param_data` dicts the model consumes.
  This is where the ~900 duplicated lines collapse to one implementation.
- **`pipeline/results.py`** — given a solved model, extract the MPO from the
  `power_balance` duals and the dispatch from `pout`, and persist the CSVs with
  the existing naming. Handles the `objective.sense.value` sign for the
  maximize (BESS welfare) case.
- **`pipeline/runner.py`** — `run_case(date, config, *, solver, compute_prices,
  evaluate)` orchestrates one (date, type). Returns a result object holding the
  solved model, output paths, and (if evaluated) a metrics dict. A `run_many`
  helper iterates dates × types, isolating per-case failures.
- **`cli.py`** — Typer command wiring args to `runner.run_many`, prints
  progress, writes the run-level summary, sets exit code on failures.

## CLI surface

```
python -m app run [DATES] [OPTIONS]
```

`DATES` positional (one token):
- `2024-04-18` — single date
- `2024-04-18:2024-04-30` — inclusive range
- `2024-04` — whole month
- omitted — all dates found under `data/condicion_inicial/`

Options:
- `--type, -t` — dispatch type; repeatable; `all` = every type; default
  `preideal`. Values: the six `DispatchOptions` members.
- `--solver` — default `cbc` (matches current scripts).
- `--no-eval` — skip the metrics-vs-XM step (default: evaluate when actuals
  present).
- `--no-prices` — skip the fix-integers→LP pricing re-solve (faster; MPO then
  invalid — for debugging only).
- `--skip-existing` / `--force` — reuse vs overwrite existing result CSVs.
- `--skip-dates 2024-10-03,...` — replaces the hardcoded `SKIP_DATES` list.
- `--out` — results directory, default `data/results`.

## Data flow (one case)

1. `ensure_data_for_date(date)` — download per-day XM files if missing.
2. `loaders` + `ofei` + (for ideal types) availability data → raw inputs.
3. `case_builder` → `set_data`, `param_data`.
4. `UnitCommitmentModel.create_model(...)`; `.solve(solver, compute_prices)`
   (the pricing LP re-solve already lives in the model).
5. `results` → write `dispatch_by_gen-{date}-{type}.csv`,
   `marginal_price-{date}-{type}.csv`.
6. If evaluating: load actuals via `actuals`, compute metrics via
   `utils.metrics`, write `metrics-{date}-{type}.csv`.

Run-level: append a row per (date, type) to `data/results/metrics-summary.csv`
with rmse / mae / bias / wape / smape / r2 + commitment f1 + gen-mix error.
This summary replaces the notebook MAPE table.

## Outputs

Per (date, type), in `--out` (default `data/results`):
- `dispatch_by_gen-{date}-{type}.csv` (existing naming)
- `marginal_price-{date}-{type}.csv` (existing naming)
- `metrics-{date}-{type}.csv` (new; only when evaluating)

Run-level:
- `metrics-summary.csv` — one row per (date, type).

## Error handling

- Per-case failures (missing data, solver error) are caught at the `run_many`
  level, logged with the date/type and the exception, and do **not** abort the
  run. This replaces the bare `except Exception: continue` that currently
  swallows failures silently — failures are reported and counted.
- The CLI prints a final summary (N succeeded, M failed, list of failed cases)
  and exits non-zero if any case failed.
- Missing actuals for a date → that case is solved and saved but its metrics row
  is omitted with a warning (not a hard failure).

## Testing

- **`case_builder`** — the regression anchor. Before refactor, capture
  `set_data`/`param_data` from the current scripts for ~2 known dates
  (one preideal, one ideal) as golden fixtures; assert the extracted builder
  reproduces them. This is what makes the extraction safe.
- **`loaders` / `ofei` / `actuals`** — small unit tests on sample files
  (tiny fixtures committed under `tests/fixtures/`).
- **`results`** — given a solved toy model (the 2-gen model already used to
  verify the pricing fix), assert the MPO/dispatch extraction and sign handling.
- **`metrics`** — already covered by the existing test; move it into `tests/`.
- **CLI** — one smoke test (`run` on a single date with a stubbed/cached data
  dir) asserting the expected output files and a summary row appear.

Test runner: `pytest`. Tests run with the project's pyomo venv
(`dam-worker-optimizer` venv has pyomo 6.6.1 + cbc + highs).

## Migration strategy

1. Build `data/*` and `pipeline/case_builder.py`; add golden-fixture tests.
2. Point `get_date_results.py` and the `run_dispatch()` body at the new
   `case_builder` / `results`; confirm byte-identical result CSVs on a sample
   date. (Safety net.)
3. Build `pipeline/runner.py` + `cli.py` + `__main__.py` on top.
4. Reduce the old scripts to thin wrappers (or remove `get_date_results.py`'s
   duplicated body) once the CLI reproduces their output.
5. Notebooks that call `run_dispatch` keep working (it now delegates to the
   pipeline).

## Dependencies

- Add `typer` to `requirement.txt`.
- No other new runtime deps; `pytest` for tests (dev).

## Open questions / deferred

- Whether to eventually fold `data_fetcher.ipynb`'s bulk ETL into the app.
- Whether `evaluate` / `compare` become first-class subcommands later (logic is
  already reusable, so promoting them is cheap).
```
