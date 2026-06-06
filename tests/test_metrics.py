import numpy as np
import pandas as pd

from app.utils import metrics as M


def test_price_metrics():
    yt = np.array([100.0, 0.0, 50.0, 200.0])
    yp = np.array([110.0, 0.0, 40.0, 180.0])
    m = M.price_metrics(yt, yp)
    assert abs(m["mae"] - 10.0) < 1e-9
    assert abs(m["bias"] - (-5.0)) < 1e-9  # under-prices on average
    assert abs(m["rmse"] - np.sqrt((100 + 0 + 100 + 400) / 4)) < 1e-9
    assert abs(m["wape"] - (40 / 350)) < 1e-9
    assert not np.isnan(m["smape"])  # survives the 0/0 entry that breaks MAPE


def test_commitment_metrics():
    actual = np.array([10.0, 0.0, 5.0, 0.0, 8.0])
    model = np.array([12.0, 3.0, 0.0, 0.0, 7.0])
    c = M.commitment_metrics(actual, model)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1)
    assert abs(c["precision"] - 2 / 3) < 1e-9
    assert abs(c["recall"] - 2 / 3) < 1e-9
    assert abs(c["accuracy"] - 3 / 5) < 1e-9


def test_generation_mix_error():
    model_d = pd.DataFrame({"generador": ["H1", "H2", "T1"], "dispatch": [100, 50, 80]})
    actual_d = pd.DataFrame({"generador": ["H1", "H2", "T1"], "dispatch": [120, 40, 70]})
    tech = {"H1": "hydro", "H2": "hydro", "T1": "thermal"}
    mix = M.generation_mix_error(model_d, actual_d, tech)
    assert abs(mix.loc["hydro", "model"] - 150) < 1e-9
    assert abs(mix.loc["hydro", "abs_error"] - 10) < 1e-9
    assert abs(mix.loc["thermal", "abs_error"] - 10) < 1e-9


def test_price_duration_curve():
    pdc = M.price_duration_curve([100.0, 0.0, 50.0, 200.0])
    assert list(pdc) == [200.0, 100.0, 50.0, 0.0]
