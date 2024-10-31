from typing import Iterator

import pyomo.environ as pyo


def power_output_rule(model: pyo.ConcreteModel, i: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """Power output constraint"""
    return model.pout[i, t] <= model.Pmax[i, t]*model.z[i, t]

def power_output_min_rule(model: pyo.ConcreteModel, i: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """Power output min constraint"""
    return model.pout[i, t] >= model.Pmin[i, t]*model.z[i, t]