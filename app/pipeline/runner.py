"""Orchestrate dispatch runs: ensure data -> build -> solve -> save -> evaluate.

Per-case failures are isolated: one bad case does not abort the batch.
"""
import traceback

import pandas as pd

from app.model.model import UnitCommitmentModel
from app.schemas import DispatchCase, InputPack, InputSource, RunResult
from app.pipeline.case_builder import build_case
from app.pipeline.results import save_results, extract_mpo
from app.data.actuals import load_actual_price
from app.utils.metrics import price_metrics


def run_case(
    case: DispatchCase,
    *,
    evaluate: bool = True,
    input_source: InputSource = InputSource.historical,
    ders: int | None = None,
    out: str = "data/results",
    data_dir: str = "data",
) -> RunResult:
    t = case.level.value
    try:
        inputs = InputPack(dispatch_date=case.dispatch_date, source=input_source, data_dir=data_dir)
        set_data, param_data, _meta = build_case(case, inputs, ders=ders)
        model = UnitCommitmentModel(case=case)
        model.create_model(set_data=set_data, param_data=param_data)
        model.solve(solver=case.solver, compute_prices=case.compute_prices)
        result = save_results(model, case, out=out)

        if evaluate:
            try:
                xm = load_actual_price(case.dispatch_date, data_dir=data_dir)
                model_mpo = extract_mpo_sorted(model)
                n = min(len(xm), len(model_mpo))
                metrics = price_metrics(xm[:n], model_mpo[:n])
                metrics_path = f"{out}/metrics-{case.dispatch_date}-{t}.csv"
                pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
                result.metrics = metrics
                result.metrics_path = metrics_path
            except FileNotFoundError:
                print(f"  ! no XM actuals for {case.dispatch_date}; skipping metrics")

        return result
    except Exception as e:
        traceback.print_exc()
        return RunResult(case=case, ok=False, error=f"{type(e).__name__}: {e}")


def extract_mpo_sorted(model) -> list[float]:
    mpo = extract_mpo(model)
    return [v for _, v in sorted(mpo.items())]


def run_many(
    cases: list[DispatchCase],
    *,
    out: str = "data/results",
    **kw,
) -> list[RunResult]:
    results: list[RunResult] = []
    for case in cases:
        print(f"==> {case.dispatch_date} [{case.level.value}]")
        results.append(run_case(case, out=out, **kw))

    rows = [
        {
            "date": r.case.dispatch_date,
            "type": r.case.level.value,
            "scenario": (
                r.case.bess_scenario.penetration_level
                if r.case.bess_scenario is not None
                else "baseline"
            ),
            **(r.metrics or {}),
            **(r.bess_summary or {}),
        }
        for r in results
        if r.ok
    ]
    if rows:
        from pathlib import Path

        Path(out).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f"{out}/metrics-summary.csv", index=False)
    return results
