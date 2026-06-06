"""Results extraction tested against a tiny solvable model.

Cheap gen A (beta=10, Pmax=100), expensive B (beta=50). demand=150 -> A=100,
B=50; marginal unit is B so MPO should equal B's cost (50).
"""

from app.model import UnitCommitmentModel, DispatchConfig
from app.pipeline.results import extract_mpo, extract_dispatch, save_results


def _toy_model():
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
    m = UnitCommitmentModel(config=DispatchConfig(dispatch_type="preideal"))
    m.create_model(set_data=set_data, param_data=param_data)
    m.solve(solver="cbc")
    return m


def test_extract_mpo_is_marginal_cost():
    m = _toy_model()
    mpo = extract_mpo(m)
    assert len(mpo) == 1
    assert abs(list(mpo.values())[0] - 50.0) < 1e-6


def test_extract_dispatch_rows():
    m = _toy_model()
    df = extract_dispatch(m)
    assert set(df.columns) == {"generador", "datetime", "dispatch"}
    by_gen = df.set_index("generador")["dispatch"].to_dict()
    assert abs(by_gen["A"] - 100.0) < 1e-6
    assert abs(by_gen["B"] - 50.0) < 1e-6


def test_save_results_writes_csvs(tmp_path):
    m = _toy_model()
    paths = save_results(m, "2024-04-18", DispatchConfig("preideal"), out=str(tmp_path))
    assert (tmp_path / "dispatch_by_gen-2024-04-18-preideal.csv").exists()
    assert (tmp_path / "marginal_price-2024-04-18-preideal.csv").exists()
    assert abs(list(paths["mpo"].values())[0] - 50.0) < 1e-6
