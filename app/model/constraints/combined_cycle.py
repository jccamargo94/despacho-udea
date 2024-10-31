

import pyomo.environ as pyo




def exclude_resource_rule(model, combined_cycle, t):
    """Exclude resource from optimization"""
    return sum(model.z[i, t] for i in model.excluded_resource[combined_cycle]) <= 1