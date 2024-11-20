from typing import Iterator

import pyomo.environ as pyo


def power_balance_rule(
    model: pyo.ConcreteModel, t: pyo.Set | Iterator
) -> pyo.Expression:
    """Power balance constraint"""
    return sum(model.pout[i, t] for i in model.I) == model.demand[t]
