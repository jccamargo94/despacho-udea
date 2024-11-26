from typing import Iterator
from logging import getLogger
import pyomo.environ as pyo


logger = getLogger("BESS_Constraints")


def compute_soc_bess(model: pyo.ConcreteModel, b, t) -> pyo.Expression:
    """
    Compute the state of charge of battery b at time t
    """
    expr = (model.rt_eff[b] * model.bess_charge[b, t]) - (
        model.bess_discharge[b, t] / model.rt_eff[b]
    )
    if t == model.T.first():
        return model.soc_bess[b, t] == model.bess_soc_0[b] + expr
    return model.soc_bess[b, t] == model.soc_bess[b, model.T.prev(t)] + expr


def min_soc_bess_constraint_rule(model: pyo.ConcreteModel, b, t) -> pyo.Expression:
    """
    Minimum state of charge constraint
    """
    return model.soc_bess[b, t] >= model.bess_min_soc[b]


def max_soc_bess_constraint_rule(model: pyo.ConcreteModel, b, t) -> pyo.Expression:
    """
    Maximum state of charge constraint
    """
    return model.soc_bess[b, t] <= model.bess_max_soc[b]


def bess_max_discharge_constraint_rule(
    model: pyo.ConcreteModel, b, t
) -> pyo.Expression:
    """
    Maximum discharge constraint
    """
    return (
        model.bess_discharge[b, t]
        <= model.bess_max_discharge[b] * model.z_bess_discharge[b, t]
    )


def bess_max_charge_constraint_rule(model: pyo.ConcreteModel, b, t) -> pyo.Expression:
    """
    Maximum charge constraint
    """
    return (
        model.bess_charge[b, t] <= model.bess_max_charge[b] * model.z_bess_charge[b, t]
    )


def bess_avoid_concurrent_charge_discharge_constraint_rule(
    model: pyo.ConcreteModel, b, t
) -> pyo.Expression:
    """
    Avoid concurrent charge and discharge constraint
    """
    return model.z_bess_charge[b, t] + model.z_bess_discharge[b, t] <= 1


def power_balance_with_bess_rule(
    model: pyo.ConcreteModel, t: pyo.Set | Iterator
) -> pyo.Expression:
    """Power balance constraint"""
    expr = 0
    if model._dispatch_type in ["bess_preideal", "bess_ideal"]:
        expr = sum(
            model.bess_discharge[b, t] - model.bess_charge[b, t] for b in model.BESS
        )
    return sum(model.pout[i, t] for i in model.I) + expr == model.demand[t]


def maximize_social_welfare(model: pyo.ConcreteModel) -> pyo.Expression:
    """
    Maximize social welfare
    """
    gen_cost = sum(model.beta[i] * model.pout[i, t] for i in model.I for t in model.T)
    start_up_cost = sum(
        model.cold_start[g] * model.zup[g, t] for g in model.G for t in model.T
    )
    # soc_cost = sum(model.bess_soc_bid[b] * model.soc_bess[b,t] for b in model.BESS for t in model.T)
    bess_discharge_cost = sum(
        model.bess_discharge_bid[b] * model.bess_discharge[b, t]
        for b in model.BESS
        for t in model.T
    )
    bess_charge_cost = sum(
        model.bess_charge_bid[b] * model.bess_charge[b, t]
        for b in model.BESS
        for t in model.T
    )

    return bess_charge_cost - bess_discharge_cost - gen_cost - start_up_cost
    # return - gen_cost - start_up_cost
