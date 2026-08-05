from datetime import date

import pandas as pd
import pytest

import app.pipeline.runner as runner
from app.pipeline.evaluate import evaluate_saved_run
from app.schemas import DispatchCase, DispatchLevel


def test_evaluate_saved_run_writes_metrics_csv(tmp_path):
    price_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-04-18", periods=24, freq="1h"),
            "ideal_marginal_price": [float(i) for i in range(24)],
        }
    )
    price_df.to_csv(tmp_path / "marginal_price-2024-04-18-preideal.csv", index=False)

    actuals_dir = tmp_path / "preideal_price"
    actuals_dir.mkdir()
    row = "MPO," + ",".join(str(float(i)) for i in range(24))
    (actuals_dir / "2024-04-18.txt").write_text(row + "\n")

    metrics = evaluate_saved_run(
        date(2024, 4, 18), DispatchLevel.preideal, out=str(tmp_path), data_dir=str(tmp_path)
    )
    assert metrics["mae"] == 0.0
    assert (tmp_path / "metrics-2024-04-18-preideal.csv").exists()


def test_evaluate_saved_run_missing_price_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_saved_run(
            date(2024, 4, 18), DispatchLevel.preideal, out=str(tmp_path), data_dir=str(tmp_path)
        )


def test_evaluate_sorts_price_csv_by_datetime(tmp_path):
    """The saved CSV is written in dual-iteration order (not guaranteed
    chronological); alignment against XM actuals is only correct after
    sorting by datetime. Written out-of-order here so a missing sort would
    misalign the comparison and change mae from 0.0 to 80.0."""
    pd.DataFrame(
        [
            {"datetime": pd.Timestamp("2024-04-18 01:00"), "ideal_marginal_price": 90.0},
            {"datetime": pd.Timestamp("2024-04-18 00:00"), "ideal_marginal_price": 10.0},
        ]
    ).to_csv(tmp_path / "marginal_price-2024-04-18-preideal.csv", index=False)

    actuals_dir = tmp_path / "preideal_price"
    actuals_dir.mkdir()
    vals = [10.0, 90.0] + [0.0] * 22
    (actuals_dir / "2024-04-18.txt").write_text("MPO," + ",".join(str(v) for v in vals) + "\n")

    metrics = evaluate_saved_run(
        date(2024, 4, 18), DispatchLevel.preideal, out=str(tmp_path), data_dir=str(tmp_path)
    )
    assert metrics["mae"] == 0.0


def test_evaluate_matches_inline_eval_exactly(monkeypatch, tmp_path):
    """Parity test: the post-hoc `evaluate` path must reproduce the inline
    `run --eval` path's numbers exactly. This does not by itself prove the
    sort-by-datetime step matters — `test_evaluate_sorts_price_csv_by_datetime`,
    elsewhere in this file, is what specifically pins that behavior (it fails
    if the sort is removed; this test would not)."""
    ts = [pd.Timestamp("2024-04-18 00:00"), pd.Timestamp("2024-04-18 01:00")]

    def fake_build(case, inputs, **kw):
        set_data = {
            "G": [],
            "I": ["A", "B"],
            "T": ts,
            "combined_cycle": [],
            "excluded_resource": {},
            "gen_on": [],
            "gen_off": [],
        }
        param_data = {
            "Pmin": {("A", t): 0.0 for t in ts} | {("B", t): 0.0 for t in ts},
            "Pmax": {("A", t): 100.0 for t in ts} | {("B", t): 100.0 for t in ts},
            "max_min_op": 0,
            "ramp_up": {},
            "ramp_down": {},
            "beta": {"A": 10.0, "B": 50.0},
            "cold_start": {},
            "demand": {t: 150.0 for t in ts},
            "TMG": {},
            "Ton": {},
            "z_on_t0_minus_1": {},
        }
        return set_data, param_data, {}

    monkeypatch.setattr(runner, "build_case", fake_build)

    actuals_dir = tmp_path / "preideal_price"
    actuals_dir.mkdir()
    row = "MPO," + ",".join(str(float(i)) for i in range(24))
    (actuals_dir / "2024-04-18.txt").write_text(row + "\n")

    case = DispatchCase(dispatch_date=date(2024, 4, 18), level=DispatchLevel.preideal, solver="cbc")
    inline_result = runner.run_case(case, evaluate=True, out=str(tmp_path), data_dir=str(tmp_path))
    assert inline_result.ok is True
    assert inline_result.metrics is not None

    post_hoc_metrics = evaluate_saved_run(
        date(2024, 4, 18), DispatchLevel.preideal, out=str(tmp_path), data_dir=str(tmp_path)
    )
    for key, value in inline_result.metrics.items():
        assert abs(value - post_hoc_metrics[key]) < 1e-9, (
            f"{key}: inline={value} post_hoc={post_hoc_metrics[key]}"
        )
