from typing import Iterator
from dateutil.relativedelta import relativedelta

import pyomo.environ as pyo


def power_generation_decomposition(model: pyo.ConcreteModel, g: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """
    Power generation decomposition constraint.

    This constraint ensures that the power output of generator g in time period t is decomposed into
    3 components: Effective power output, power output when generator in ramping up and power output when generator
    if ramping down.

    Args:
        model: Pyomo ConcreteModel object.
        g: Set of generators.
        t: Set of time periods.

    Returns:
        Pyomo Expression object.
    """
    return model.pout[g, t] == model.power_ramp_up[g, t] + model.power_ramp_down[g, t] + model.p_out_effective[g, t]


def exclusive_ramp_up_effective_constraint(model: pyo.ConcreteModel, g: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """
    Exclusive ramping up and effective power output constraint.

    This constraints ensures that ramping up generation are not scheduled with effective power output
    at the same time, since they are mutually exclusive by decomposed constraint

    Args:
        model: Pyomo ConcreteModel object.
        g: Set of generators.
        t: Set of time periods.

    Returns:
        Pyomo Expression (Constraint) object.
    """
    return model.power_ramp_up[g,t] <= model.Pmin[g,t]*(1 - model.z[g,t])

def exclusive_ramp_down_effective_constraint(model: pyo.ConcreteModel, g: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """
    Exclusive ramping up and effective power output constraint.

    This constraints ensures that ramping down and effective generation are not scheduled at the same time

    Args:
        model: Pyomo ConcreteModel object.
        g: Set of generators.
        t: Set of time periods.

    Returns:
        Pyomo Expression (Constraint) object.

    """
    return model.power_ramp_down[g,t] <= model.Pmin[g,t]*(1 - model.z[g,t])


def start_up_shut_down_constraints(model: pyo.ConcreteModel, g: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """
    Start up and shut down constraint.

    This constraint decide whether a generator should be started up or shut down in time period t based in the previous condition
    and its scheduled generation condition

    Args:
        model: Pyomo ConcreteModel object.
        g: Set of generators.
        t: Set of time periods.

    Returns:
        Pyomo Expression (Constraint) object.
    """
    if t == model.T.first():
        return pyo.Constraint.Skip
    return model.zup[g,t] - model.zdown[g,t] == model.z[g,t] - model.z[g,t-relativedelta(hours=1)]

def start_up_zero_on_gen(model: pyo.ConcreteModel, g: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """
    This constraints ensure that any scheduled generations scheduled in the first period cannot be started up
    """
    if t == model.T.first():
        return model.zup[g,t] == 0
    return pyo.Constraint.Skip
    
def shutdown_zero_off_gen(model: pyo.ConcreteModel, g: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """
    This constraints ensure that any no scheduled generations scheduled in the first period cannot be shutted_down
    """
    if t == model.T.first():
        return model.zdown[g,t] == 0
    return pyo.Constraint.Skip

    
def start_up_off_gen(model: pyo.ConcreteModel, g: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """
    This constraints decide whether a generator should be started up or not based on the previous condition
    """
    if t == model.T.first():
        return model.zup[g,t] == model.z[g,t]
    return pyo.Constraint.Skip


    
def shut_down_on_gen(model: pyo.ConcreteModel, g: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """
    This constraints decide whether a generator should be starshuted down or not based on the previous condition
    """
    if t == model.T.first():
        return model.zdown[g,t] == 1 - model.z[g,t]
    return pyo.Constraint.Skip
        