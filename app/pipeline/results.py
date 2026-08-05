"""Extract and persist dispatch results from a solved model.

The marginal price (MPO) is the dual of the power_balance constraint. The sign
is multiplied by the objective sense so that maximize (BESS welfare) and
minimize (cost) cases both yield a positive price.
"""

import pandas as pd
import pyomo.environ as pyo

from app.schemas import DispatchCase, RunResult
from app.storage import get_storage


def extract_mpo(model) -> dict:
    sense = model._model.objective.sense.value
    return {
        ke.index(): sense * pyo.value(dual_)
        for ke, dual_ in model._model.dual.items()
        if "power_balance" in ke.name
    }


def extract_dispatch(model) -> pd.DataFrame:
    data = {(g, t): pyo.value(v) for (g, t), v in model._model.pout.items()}
    return pd.DataFrame(data=data.values(), index=data.keys(), columns=["dispatch"]).reset_index(
        drop=False, names=["generador", "datetime"]
    )


def extract_bess(model, mpo: dict) -> pd.DataFrame:
    """Per-unit x hour BESS activity, settled at the system marginal price
    (MPO), not at the unit's own bid: the bid is an optimization input, and
    grid_asset units often have no bids at all. MPO is in COP/MWh, so
    energy_MWh x price_COP_per_MWh yields COP directly; no scaling factor
    is needed."""
    charge = {(b, t): pyo.value(v) for (b, t), v in model._model.bess_charge.items()}
    discharge = {(b, t): pyo.value(v) for (b, t), v in model._model.bess_discharge.items()}
    soc = {(b, t): pyo.value(v) for (b, t), v in model._model.soc_bess.items()}

    rows = []
    for key in sorted(charge.keys()):
        b, t = key
        price = mpo[t]
        c, d = charge[key], discharge[key]
        rows.append(
            {
                "unit": b,
                "datetime": t,
                "charge": c,
                "discharge": d,
                "soc": soc[key],
                "revenue": d * price,
                "cost": c * price,
            }
        )
    return pd.DataFrame(rows)


def _bess_summary(bess_df: pd.DataFrame) -> dict[str, float]:
    return {
        "bess_charge_mwh": float(bess_df["charge"].sum()),
        "bess_discharge_mwh": float(bess_df["discharge"].sum()),
        "bess_avg_soc_mwh": float(bess_df["soc"].mean()),
        "bess_net_revenue": float((bess_df["revenue"] - bess_df["cost"]).sum()),
    }


def save_results(model, case: DispatchCase, out: str = "data/results") -> RunResult:
    storage = get_storage(out)
    t = case.level.value

    dispatch = extract_dispatch(model)
    dispatch_name = f"dispatch_by_gen-{case.dispatch_date}-{t}.csv"
    with storage.open(dispatch_name, "w") as f:
        dispatch.to_csv(f, sep=",", index=False)

    mpo = extract_mpo(model)
    price_name = f"marginal_price-{case.dispatch_date}-{t}.csv"
    with storage.open(price_name, "w") as f:
        pd.DataFrame(
            data=mpo.values(), index=mpo.keys(), columns=["ideal_marginal_price"]
        ).reset_index(drop=False, names=["datetime"]).to_csv(f, sep=",", index=False)

    bess_path = None
    bess_summary = None
    if case.bess_scenario is not None:
        bess_df = extract_bess(model, mpo)
        bess_name = f"bess_results-{case.dispatch_date}-{t}.csv"
        with storage.open(bess_name, "w") as f:
            bess_df.to_csv(f, sep=",", index=False)
        bess_path = f"{out}/{bess_name}"
        bess_summary = _bess_summary(bess_df)

    return RunResult(
        case=case,
        ok=True,
        dispatch_path=f"{out}/{dispatch_name}",
        price_path=f"{out}/{price_name}",
        bess_path=bess_path,
        bess_summary=bess_summary,
    )
