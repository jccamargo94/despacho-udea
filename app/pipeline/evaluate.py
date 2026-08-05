"""Post-hoc evaluation: re-score a saved run's price CSV against XM actuals
without re-solving the model. Must reproduce the numbers `run --eval`
computes inline, including the datetime sort the inline path applies via
extract_mpo_sorted — the saved CSV is written in dual-iteration order, which
is not guaranteed sorted."""

from datetime import date

import pandas as pd

from app.data.actuals import load_actual_price
from app.schemas import DispatchLevel
from app.storage import get_storage
from app.utils.metrics import price_metrics


def _upsert_metrics_summary(
    storage, dispatch_date: date, level: DispatchLevel, metrics: dict[str, float]
) -> None:
    """Upsert this run's row into metrics-summary.csv, keyed on
    (date, type, scenario). Without this, a `run --no-eval` -> `evaluate`
    -> `compare` flow never sees post-hoc metrics: `run_many` only writes
    metrics-summary.csv once, at the end of a batch, and evaluate_saved_run
    previously only wrote the per-run metrics-{date}-{level}.csv file."""
    row = {"date": str(dispatch_date), "type": level.value, "scenario": "baseline", **metrics}
    summary_path = "metrics-summary.csv"
    if storage.exists(summary_path):
        with storage.open(summary_path) as f:
            summary = pd.read_csv(f)
        stale = (
            (summary["date"] == row["date"])
            & (summary["type"] == row["type"])
            & (summary["scenario"] == row["scenario"])
        )
        summary = pd.concat([summary[~stale], pd.DataFrame([row])], ignore_index=True)
    else:
        summary = pd.DataFrame([row])
    with storage.open(summary_path, "w") as f:
        summary.to_csv(f, index=False)


def evaluate_saved_run(
    dispatch_date: date,
    level: DispatchLevel,
    *,
    out: str = "data/results",
    data_dir: str = "data",
) -> dict[str, float]:
    storage = get_storage(out)
    t = level.value
    price_name = f"marginal_price-{dispatch_date}-{t}.csv"
    if not storage.exists(price_name):
        raise FileNotFoundError(f"no saved price CSV for {dispatch_date} [{t}] in {out}")

    with storage.open(price_name) as f:
        price_df = pd.read_csv(f, parse_dates=["datetime"]).sort_values("datetime")
    model_mpo = price_df["ideal_marginal_price"].to_numpy()

    xm = load_actual_price(dispatch_date, data_dir=data_dir)
    n = min(len(xm), len(model_mpo))
    metrics = price_metrics(xm[:n], model_mpo[:n])

    metrics_name = f"metrics-{dispatch_date}-{t}.csv"
    with storage.open(metrics_name, "w") as f:
        pd.DataFrame([metrics]).to_csv(f, index=False)
    _upsert_metrics_summary(storage, dispatch_date, level, metrics)
    return metrics
