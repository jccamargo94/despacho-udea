"""Orchestrate dispatch runs: ensure data -> build -> solve -> save -> evaluate.

Per-case failures are isolated: one bad date/type does not abort the batch.
"""

from dataclasses import dataclass, field
from datetime import date
import traceback

import pandas as pd

from app.model import UnitCommitmentModel, DispatchConfig
from app.pipeline.case_builder import build_case
from app.pipeline.results import save_results
from app.data.actuals import load_actual_price
from app.utils.metrics import price_metrics


@dataclass
class CaseResult:
    dispatch_date: date
    dispatch_type: str
    ok: bool
    paths: dict = field(default_factory=dict)
    metrics: dict | None = None
    error: str | None = None


def run_case(
    dispatch_date: date,
    config: DispatchConfig,
    *,
    solver: str = "cbc",
    compute_prices: bool = True,
    evaluate: bool = True,
    bess: dict | None = None,
    ders: int | None = None,
    out: str = "data/results",
    data_dir: str = "data",
) -> CaseResult:
    t = config.dispatch_type.value
    try:
        set_data, param_data, _meta = build_case(
            dispatch_date, config, bess=bess, ders=ders, data_dir=data_dir
        )
        model = UnitCommitmentModel(config=config)
        model.create_model(set_data=set_data, param_data=param_data)
        model.solve(solver=solver, compute_prices=compute_prices)
        paths = save_results(model, dispatch_date, config, out=out)

        metrics = None
        if evaluate:
            try:
                xm = load_actual_price(dispatch_date, data_dir=data_dir)
                # Align by timestamp: dual-suffix iteration order is not
                # guaranteed chronological, but the MPO keys are the T index
                # (timestamps) and xm is in hour order.
                model_mpo = [v for _, v in sorted(paths["mpo"].items())]
                n = min(len(xm), len(model_mpo))
                metrics = price_metrics(xm[:n], model_mpo[:n])
                pd.DataFrame([metrics]).to_csv(
                    f"{out}/metrics-{dispatch_date}-{t}.csv", index=False
                )
            except FileNotFoundError:
                print(f"  ! no XM actuals for {dispatch_date}; skipping metrics")

        return CaseResult(dispatch_date, t, True, paths=paths, metrics=metrics)
    except Exception as e:
        traceback.print_exc()
        return CaseResult(dispatch_date, t, False, error=f"{type(e).__name__}: {e}")


def run_many(
    dates: list[date],
    configs: list[DispatchConfig],
    *,
    out: str = "data/results",
    **kw,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for d in dates:
        for cfg in configs:
            print(f"==> {d} [{cfg.dispatch_type.value}]")
            results.append(run_case(d, cfg, out=out, **kw))

    rows = [
        {"date": r.dispatch_date, "type": r.dispatch_type, **r.metrics}
        for r in results
        if r.ok and r.metrics
    ]
    if rows:
        from pathlib import Path

        Path(out).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f"{out}/metrics-summary.csv", index=False)
    return results
