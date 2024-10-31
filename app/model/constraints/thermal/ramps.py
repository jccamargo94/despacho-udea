from typing import Iterator
from dateutil.relativedelta import relativedelta

import pyomo.environ as pyo



def up_ramps_thermal_gen(model, g: str, t: str) -> pyo.Constraint:
    """
    Upward ramping constraint for thermal generator g at time t.
    """
    if t == model.T.first():
        return pyo.Constraint.Skip
    if not model.RU[g]:
        return pyo.Constraint.Skip
    return model.pout[g, t] - model.pout[g, model.T.prev(t)] <= model.RU[g]


def down_ramps_thermal_gen(model, g: str, t: str) -> pyo.Constraint:
    """
    Downward ramping constraint for thermal generator g at time t.
    """
    if t == model.T.first():
        return pyo.Constraint.Skip
    if not model.RD[g]:
        return pyo.Constraint.Skip
    return model.pout[g, model.T.prev(t)] - model.pout[g, t] <= model.RD[g]