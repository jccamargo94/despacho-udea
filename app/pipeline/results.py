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
    return pd.DataFrame(
        data=data.values(), index=data.keys(), columns=["dispatch"]
    ).reset_index(drop=False, names=["generador", "datetime"])


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

    return RunResult(
        case=case,
        ok=True,
        dispatch_path=f"{out}/{dispatch_name}",
        price_path=f"{out}/{price_name}",
    )
