# despacho-udea — Colombian Electricity Dispatch Model

Optimal-dispatch model for the Colombian electricity system, plus tooling to
download the market data and to evaluate the model against the market operator
(XM). This document is the single entry point for **both humans and AI agents**.
Read it fully before changing anything.

---

## 1. What this project does

Two things:

1. **Download historical data** for the Colombian electricity market from XM.
2. **Solve an optimal dispatch** with [pyomo](https://www.pyomo.org/) and compare
   the result against XM's published dispatch and marginal price.

There are several model variants (see `DispatchOptions` in
[app/model/model.py](app/model/model.py)):

| dispatch_type            | meaning |
|--------------------------|---------|
| `preideal`               | Pre-dispatch: classical economic dispatch against the **demand forecast** (PrId). No ramp/min-up-down commitment constraints. Fast. The primary case. |
| `ideal`                  | Ideal dispatch: adds thermal commitment features (ramps, minimum on-line time, startup/shutdown). Slower, needs more data. |
| `bess_preideal`, `bess_ideal` | Same, plus a Battery Energy Storage System (BESS). |
| `bess_preideal_resource`, `bess_ideal_resource` | BESS modelled as a market resource (social-welfare objective). |

### Key domain concepts

- **MPO (Marginal Price of Operation):** the dual of the `power_balance`
  constraint = the system marginal price. We compare model MPO against XM's MPO.
- **MILP → pricing LP (important):** the model has binary commitment variables
  (`z`), so it is a MILP. The dual of a MILP is **not** a valid marginal price.
  `UnitCommitmentModel.solve(..., compute_prices=True)` (the default) therefore
  runs a second "pricing" solve: it fixes the integer variables to their MILP
  optimum and re-solves the LP, so the `power_balance` duals are valid prices.
  This is the standard ISO pricing run. Do **not** read prices from a raw MILP
  solve.
- **Evaluation metrics:** in [app/utils/metrics.py](app/utils/metrics.py). Use
  RMSE / MAE / bias (COP/kWh) and WAPE/sMAPE. **MAPE is deliberately avoided** —
  the hydro-dominated system drives MPO near zero, which makes MAPE explode.

---

## 2. Repository map

```
app/
  cli.py            # Typer CLI: `python -m app run ...`   <-- start here
  __main__.py       # entrypoint enabling `python -m app`
  dates.py          # date-token parsing (single / range / month / all)
  model/            # the pyomo UnitCommitmentModel + constraints (the math)
  data/
    download.py     # download XM files; ensure_data_for_date()
    ofei.py         # parse the OFEI offer text file
    loaders.py      # read the root CSVs + unit conversions
    actuals.py      # load XM actual predispatch + price (evaluation targets)
    paths.py        # resolve_input(): handles BOTH data layouts (see §4)
  pipeline/
    case_builder.py # (date, config) -> (set_data, param_data, meta)  [core]
    results.py      # extract MPO + dispatch from a solved model; save CSVs
    runner.py       # orchestrate: build -> solve -> save -> evaluate
  utils/
    metrics.py      # evaluation metrics
    misc.py         # back-compat shim -> app.data.download

run_dispatch.py     # LEGACY single-date runner (now delegates to case_builder)
get_date_results.py # LEGACY batch runner (now delegates to runner.run_many)
*.ipynb             # exploratory notebooks (data_fetcher, comparisons, charts)

docs/superpowers/specs/  # design spec for the CLI app
docs/superpowers/plans/  # step-by-step implementation plan
tests/                   # pytest suite (see §7)
data/                    # ALL inputs/outputs; git-ignored (empty in a fresh clone)
```

---

## 3. Setup

Requires Python 3.10+ and the **CBC** solver binary on `PATH`
(`which cbc`; on Debian/Ubuntu: `apt-get install coinor-cbc`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt        # note: file is named requirement.txt (singular)
```

`requirement.txt` is a full freeze and is heavy (it includes Orange3, Jupyter,
etc.). The actual runtime essentials are: `pandas`, `numpy`, `Pyomo`, `thefuzz`,
`typer`, `requests`, `plotly` (plots only), `scikit-learn` (notebooks),
`openpyxl`/`xlrd` (DERs Excel only). `pytest` for tests.

Verify the toolchain:

```bash
python -c "import pyomo.environ as pyo; print('cbc', pyo.SolverFactory('cbc').available())"
# -> cbc True
pytest -q
```

---

## 4. Data layout (read carefully)

`data/` is git-ignored, so a fresh clone has **no data**. You must obtain it.
Two layouts are supported; `app/data/paths.resolve_input()` tries the organized
("historical/offline") location first, then falls back to the live per-date
download folder.

**Inputs that vary by date** (resolved by `paths.py`):

| file kind | historical/offline location | live download location |
|-----------|-----------------------------|------------------------|
| OFEI (offers)            | `data/oferta_inicial/OFEI{MMDD}.txt`        | `data/{YYYY-MM-DD}/OFEI{MMDD}.txt` |
| dCondIniP / dCondIniU    | `data/condicion_inicial/{YYYY-MM-DD}/dCondIni*{MMDD}.txt` | `data/{YYYY-MM-DD}/dCondIni*{MMDD}.txt` |
| PrId (demand forecast)   | `data/predespacho_ideal/PrId{MMDD}_NAL.txt` | `data/{YYYY-MM-DD}/PrId{MMDD}_NAL.txt` |
| iMAR (XM price)          | `data/predespacho_ideal/iMAR{MMDD}_NAL.txt` | `data/{YYYY-MM-DD}/iMAR{MMDD}_NAL.txt` |

**Root CSVs** (loaded by `app/data/loaders.py`, same for every date):

```
data/dispo_declarada.csv      data/ofertas.csv        data/demaCome.csv
data/agc_asignado.csv         data/parametros_plantas.csv
data/precio_bolsa/precio_bolsa_2024.csv
data/DispoCome_resource.csv   # only for `ideal` types
```

**JSON / Excel helpers:**

```
data/ramps.json                  data/preideal_dispatch_map.json
data/error_map.json              data/Supuestos Modelo de despacho.xlsx  # DERs only
```

**Evaluation targets** (XM actuals, loaded by `app/data/actuals.py`):

```
data/preideal_price/{YYYY-MM-DD}.txt      # XM marginal price (24 values)
data/preideal_dispatch/{YYYY-MM-DD}.txt   # XM predispatch generation
```

**Outputs** are written to `data/results/`.

**Date discovery:** the CLI (no date argument) and the legacy batch script find
dates by globbing `data/condicion_inicial/{YYYY-MM-DD}/`.

### Downloading data

`app/data/download.py` fetches the per-date files from the XM portal into
`data/{YYYY-MM-DD}/`. Running the CLI auto-downloads missing per-date files
(`ensure_data_for_date`). The bulk root CSVs come from `data_fetcher.ipynb`
(not yet ported to the app). **Network access to the XM portal is required** for
downloads; in an offline environment, place files manually per the table above.

---

## 5. How to run

### CLI (preferred)

```bash
python -m app run 2024-04-18                 # one date, preideal
python -m app run 2024-04-18 -t ideal        # one type
python -m app run 2024-04-18:2024-04-30 -t all   # range, every type
python -m app run 2024-04                     # whole month
python -m app run                             # all discovered dates
```

Useful options (`python -m app run --help`):

| option | default | meaning |
|--------|---------|---------|
| `-t, --type`     | `preideal` | repeatable; `all` = every type |
| `--solver`       | `cbc`      | pyomo solver name |
| `--eval/--no-eval` | eval     | compute metrics vs XM actuals when present |
| `--prices/--no-prices` | prices | run the fix-integers→LP pricing re-solve (off = MPO invalid, debugging only) |
| `--skip-dates`   | (none)     | comma-separated `YYYY-MM-DD` to skip |
| `--out`          | `data/results` | output dir |
| `--data-dir`     | `data`     | input dir |

The CLI exits non-zero if any case failed and lists the failures. Per-case
failures (missing data, solver error) are isolated and never abort the batch.

### Legacy / programmatic

```python
from datetime import date
from app.model import DispatchConfig
from app.pipeline.runner import run_case

res = run_case(date(2024, 4, 18), DispatchConfig("preideal"), solver="cbc")
print(res.ok, res.metrics)
```

`run_dispatch.run_dispatch(...)` still exists (used by notebooks) and returns
`(mpo_df, model, pmax_new_resources, expansion_sources)`.

---

## 6. Outputs

Per `(date, type)` in `data/results/`:

```
dispatch_by_gen-{date}-{type}.csv     # generation per unit per hour
marginal_price-{date}-{type}.csv      # model MPO per hour
metrics-{date}-{type}.csv             # price metrics vs XM (when evaluated)
```

Run-level: `data/results/metrics-summary.csv` — one row per `(date, type)` with
rmse / mae / bias / wape / smape / r2.

---

## 7. Testing

```bash
pytest -q          # 23 tests
```

Covers: date parsing, OFEI parser, loaders, actuals, path resolver, results
extraction + the pricing fix (a tiny solvable cbc model), metrics, runner
failure-isolation, and the CLI.

---

## 8. Current status & known gaps (DO NOT skip)

- **`app/pipeline/case_builder.py` is NOT validated end-to-end.** It was
  extracted (faithfully, by moving code) from the legacy scripts, but it has
  never run against real data in this repo's CI environment because `data/` is
  empty and XM was unreachable. The unit tests do **not** exercise it. Treat its
  correctness as "code-review only" until proven against data. This is the single
  most likely place for a transcription bug.
- **To validate it:** capture `set_data` / `param_data` from git rev `450eae70`'s
  `run_dispatch` for a known date, pickle as a golden fixture, and assert
  `build_case` reproduces it (see Task 6, Step 3 in the plan). Then run
  `python -m app run <date> -t preideal` and diff the result CSVs against the old
  script's output.
- The two legacy scripts originally read some inputs from **different** locations
  and used **different demand inputs**. The unified `build_case` now uses the
  **PrId forecast** for demand and `resolve_input` for paths. If your historical
  data only has `data/preideal_dispatch/` (realized) and no `PrId`, demand
  resolution will fail with a clear "tried: [...]" error — supply PrId or revisit
  this decision.
- The bulk-CSV ETL (`data_fetcher.ipynb`) is not yet part of the app.
- Plotting was intentionally removed from the run path (kept in notebooks /
  legacy `run_dispatch`).

---

## 9. For AI agents

- **Environment:** there is no project-specific virtualenv checked in. Build one
  per §3, or use any env with `pandas + pyomo + thefuzz + typer + cbc`.
- **Before claiming anything "works":** run `pytest -q` and show the output.
  Green tests prove the edges + the pricing fix; they do **not** prove
  `case_builder` against real data (see §8). Do not round "23 passed" up to "the
  app works."
- **Design docs are authoritative:** read `docs/superpowers/specs/` and
  `docs/superpowers/plans/` before architectural changes.
- **Faithful-extraction rule:** `case_builder.py` is a near-verbatim move of the
  legacy logic (fuzzy name-matching, combined-cycle synthesis, initial
  conditions). Some locals are computed-but-unused — that is preserved on
  purpose. Do not "clean up" the math without a golden-fixture regression test.
- **Pricing:** never read MPO from a raw MILP solve. Use `solve(compute_prices=True)`
  and read `power_balance` duals. Align MPO to XM by **timestamp**, not dict
  order (`runner.py` does this).
- **Commits:** branch off `main`; current work lives on `develop`. End commit
  messages with the `Co-Authored-By` trailer already used in the history.
```
