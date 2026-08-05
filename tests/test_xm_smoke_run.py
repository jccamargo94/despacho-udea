"""Layer-3 check: the fixture survives a real cbc solve via run_case, with
no XM actuals file and no network access — the two conditions a Docker
smoke test (Fase 2C) will also run under."""

from datetime import date
from pathlib import Path

import pandas as pd

from app.pipeline.runner import run_case
from app.schemas.case import DispatchCase, DispatchLevel

DD = str(Path(__file__).parent / "fixtures" / "xm_smoke")
FECHA = date(2024, 4, 18)


def test_run_case_solves_with_no_network_and_no_actuals(tmp_path, monkeypatch):
    def _no_network(*a, **kw):
        raise AssertionError(f"unexpected network call: {a} {kw}")

    monkeypatch.setattr("app.data.download.requests.get", _no_network)

    case = DispatchCase(dispatch_date=FECHA, level=DispatchLevel.preideal, solver="cbc")
    out = str(tmp_path / "results")
    result = run_case(case, evaluate=True, out=out, data_dir=DD)

    assert result.ok, result.error
    assert result.error is None
    # no preideal_price/ fixture file exists -> metrics silently skipped, not a failure
    assert result.metrics is None

    price = pd.read_csv(result.price_path)
    assert len(price) == 24
    assert (price["ideal_marginal_price"] > 0).all()
    # TERMO2 (180 COP/kWh bid, more expensive) is the marginal unit once TERMO1
    # (300 MW cap) is exhausted against 350 MW demand -> MPO = TERMO2's beta.
    assert (price["ideal_marginal_price"] == 180000.0).all()

    dispatch = pd.read_csv(result.dispatch_path)
    termo1 = dispatch[dispatch["generador"] == "TERMO1"]["dispatch"]
    termo2 = dispatch[dispatch["generador"] == "TERMO2"]["dispatch"]
    assert (termo1 == 300.0).all()  # at its cap every hour
    assert (termo2 == 50.0).all()  # covers the remaining 350 - 300
