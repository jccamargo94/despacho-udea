"""Runner orchestration: a failing case must not abort the batch."""

import app.pipeline.runner as runner
from app.model import DispatchConfig


def _toy_case():
    set_data = {
        "G": [], "I": ["A", "B"], "T": [1], "combined_cycle": [],
        "excluded_resource": {}, "gen_on": [], "gen_off": [],
    }
    param_data = {
        "Pmin": {("A", 1): 0.0, ("B", 1): 0.0},
        "Pmax": {("A", 1): 100.0, ("B", 1): 100.0},
        "max_min_op": 0, "ramp_up": {}, "ramp_down": {},
        "beta": {"A": 10.0, "B": 50.0}, "cold_start": {},
        "demand": {1: 150.0}, "TMG": {}, "Ton": {}, "z_on_t0_minus_1": {},
    }
    return set_data, param_data, {}


def test_failure_isolated(monkeypatch, tmp_path):
    from datetime import date

    good, bad = date(2024, 4, 18), date(2024, 4, 19)

    def fake_build(dispatch_date, config, **kw):
        if dispatch_date == bad:
            raise RuntimeError("boom")
        return _toy_case()

    monkeypatch.setattr(runner, "build_case", fake_build)

    results = runner.run_many(
        [good, bad],
        [DispatchConfig("preideal")],
        solver="cbc",
        evaluate=False,
        out=str(tmp_path),
    )
    assert len(results) == 2
    ok = {r.dispatch_date: r.ok for r in results}
    assert ok[good] is True
    assert ok[bad] is False
    bad_r = next(r for r in results if r.dispatch_date == bad)
    assert "boom" in bad_r.error
