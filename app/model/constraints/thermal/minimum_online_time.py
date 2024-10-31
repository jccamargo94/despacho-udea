from typing import Iterator
from logging import getLogger
import pyomo.environ as pyo


logger = getLogger("thermal_constraints")


def compute_Lr(model, g: str) -> int:
    """
    Compute the minimum online time for generator g
    """
    LR_1 = int(len(model.T))
    LR_2 = max(0, int((model.TMG[g] - model.Ton[g]) * model.z_on_t0_minus_1[g]))
    return min(LR_1, LR_2)


def minimum_online_time_rule_for_online_gen_rule(model, g: str) -> pyo.Constraint:
    """
    Minimum online time constraint for generator g if it start up generating and dispatched
    """
    final_time = [dt for i,dt in enumerate(model.T) if i <= model.Lr[g]]
    return sum(1 - model.z[g, t] for t in final_time) == 0



def minimum_online_time_rule_for_offline_gen_rule(model, g: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """
    Minimum online time constraint for generator g if not start generating
    """
    t_num = list(model.T).index(t) + 1
    if model.Lr[g] + 1 <= t_num <= len(model.T) - model.TMG[g]:
        summation_index = [dt for indx,dt in enumerate(model.T) if t_num <= indx <= t_num + model.TMG[g] - 1]
        if summation_index:
            return sum(model.z[g,dt] for dt in summation_index) >= model.TMG[g]*model.zup[g,t]
        return pyo.Constraint.Skip
    return pyo.Constraint.Skip
    

def minimum_online_time_for_offline_gen_rule_last_section(model, g: pyo.Set | Iterator, t: pyo.Set | Iterator) -> pyo.Expression:
    """
    Minimum online time constraint for generator g if not start generating
    """
    t_num = list(model.T).index(t) + 1
    if len(model.T) - model.TMG[g] + 1 <= t_num <= len(model.T):
    # if model.Lr[g] + 1 <= t_num <= len(model.T) - model.TMG[g]:
        summation_index = [dt for indx,dt in enumerate(model.T) if t_num <= len(model.T)]
        if summation_index:
            return sum(model.z[g,dt] - model.zup[g,dt] for dt in summation_index) >= 0
        return pyo.Constraint.Skip
    return pyo.Constraint.Skip