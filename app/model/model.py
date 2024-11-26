from dataclasses import dataclass
from datetime import date
from enum import Enum

import pyomo.environ as pyo

from app.model.constraints import (
    power_output_rule,
    power_output_min_rule,
    power_balance_rule,
    exclude_resource_rule,
)
from app.model.constraints.thermal import (
    compute_Lr,
    minimum_online_time_rule_for_online_gen_rule,
    minimum_online_time_rule_for_offline_gen_rule,
    minimum_online_time_for_offline_gen_rule_last_section,
    power_generation_decomposition,
    exclusive_ramp_up_effective_constraint,
    exclusive_ramp_down_effective_constraint,
    start_up_shut_down_constraints,
    start_up_zero_on_gen,
    shutdown_zero_off_gen,
    start_up_off_gen,
    shut_down_on_gen,
    up_ramps_thermal_gen,
    down_ramps_thermal_gen,
)

from app.model.constraints.bess.soc import (
    compute_soc_bess,
    min_soc_bess_constraint_rule,
    max_soc_bess_constraint_rule,
    bess_max_discharge_constraint_rule,
    bess_max_charge_constraint_rule,
    bess_avoid_concurrent_charge_discharge_constraint_rule,
    power_balance_with_bess_rule,
    maximize_social_welfare,
)


class DispatchOptions(str, Enum):
    preideal = "preideal"
    ideal = "ideal"
    bess_preideal = "bess_preideal"
    bess_ideal = "bess_ideal"


@dataclass
class DispatchConfig:
    dispatch_type: DispatchOptions

    def __init__(self, dispatch_type: str):
        self.dispatch_type = DispatchOptions(dispatch_type)


class UnitCommitmentModel:
    def __init__(self, config: DispatchConfig):
        self._model = pyo.ConcreteModel()
        self._dispatch_config = config
        self._model._dispatch_type = config.dispatch_type.value

    def load_data(self, folder: str, date: date) -> None: ...

    def _create_sets(self, set_data: dict) -> None:
        self._model.G = pyo.Set(
            doc="fuel-fired Generators", initialize=set_data.get("G", [])
        )
        self._model.I = pyo.Set(doc="Generators", initialize=set_data.get("I", []))
        self._model.T = pyo.Set(doc="Time periods", initialize=set_data.get("T", []))
        self._model.combined_cycle = pyo.Set(
            doc="Combined cycle resoruces",
            initialize=set_data.get("combined_cycle", []),
        )
        self._model.excluded_resource = pyo.Set(
            self._model.combined_cycle,
            within=self._model.I,
            doc="Resource to exclude",
            initialize=set_data.get("excluded_resource", []),
        )
        self._model.gen_on = pyo.Set(
            doc="generators that start on at time t",
            within=self._model.I,
            initialize=set_data.get("gen_on", []),
        )
        self._model.gen_off = pyo.Set(
            doc="generators that start off at time t",
            within=self._model.I,
            initialize=set_data.get("gen_off", []),
        )

    def _create_parameters(self, param_data: dict) -> None:
        self._model.Pmin = pyo.Param(
            self._model.I,
            self._model.T,
            domain=pyo.NonNegativeReals,
            doc="Minimum power bid of generator g",
            initialize=param_data["Pmin"],
            default=param_data["max_min_op"],
        )
        self._model.Pmax = pyo.Param(
            self._model.I,
            self._model.T,
            domain=pyo.NonNegativeReals,
            doc="Maximum power bid of generator g",
            initialize=param_data["Pmax"],
            default=0,
        )
        self._model.RU = pyo.Param(
            self._model.G,
            doc="Ramp up limit of generator g",
            domain=pyo.NonNegativeReals,
            initialize=param_data["ramp_up"],
            default=10000,
        )
        self._model.RD = pyo.Param(
            self._model.G,
            doc="Ramp down limit of generator g",
            domain=pyo.NonNegativeReals,
            initialize=param_data["ramp_down"],
            default=10000,
        )
        self._model.beta = pyo.Param(
            self._model.I,
            doc="Cost of generator i",
            initialize=param_data["beta"],
            default=0,
        )
        self._model.cold_start = pyo.Param(
            self._model.G,
            doc="Cold start cost of generator g",
            initialize=param_data["cold_start"],
            default=0,
        )
        self._model.demand = pyo.Param(
            self._model.T,
            doc="Demand in time period t",
            initialize=param_data["demand"],
            default=0,
        )
        self._model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

        # -- Params for other technical issues ---
        self._model.TMG = pyo.Param(
            self._model.G,
            domain=pyo.NonNegativeIntegers,
            doc="minimum up time for generator g",
            initialize=param_data["TMG"],
            default=0,
        )
        self._model.Ton = pyo.Param(
            self._model.G,
            domain=pyo.NonNegativeIntegers,
            doc="online time for generator g until t_0-1",
            initialize=param_data["Ton"],
            default=0,
        )
        self._model.z_on_t0_minus_1 = pyo.Param(
            self._model.G,
            domain=pyo.NonNegativeIntegers,
            doc="on/off status of generator g at t_0-1",
            initialize=param_data["z_on_t0_minus_1"],
            default=0,
        )
        self._model.Lr = pyo.Param(
            self._model.G,
            domain=pyo.NonNegativeIntegers,
            doc="pending time until TMG for generator g",
            initialize=compute_Lr,
            default=0,
        )

    def _create_variables(self) -> None:
        self._model.pout = pyo.Var(
            self._model.I,
            self._model.T,
            domain=pyo.NonNegativeReals,
            doc="Power output of generator g in time period t",
        )
        self._model.z = pyo.Var(
            self._model.I,
            self._model.T,
            domain=pyo.Binary,
            doc="On/off status of generator g in time period t",
        )

        # Vars for fuel-fired generators
        self._model.p_out_effective = pyo.Var(
            self._model.G,
            self._model.T,
            domain=pyo.NonNegativeReals,
            doc="Effective power output of generator g in time period t",
        )
        self._model.power_ramp_up = pyo.Var(
            self._model.G,
            self._model.T,
            domain=pyo.NonNegativeReals,
            doc="Power ramping up of generator g in time period t",
        )
        self._model.power_ramp_down = pyo.Var(
            self._model.G,
            self._model.T,
            domain=pyo.NonNegativeReals,
            doc="Power ramping down of generator g in time period t",
        )
        self._model.zup = pyo.Var(
            self._model.G,
            self._model.T,
            domain=pyo.Binary,
            doc="Start up signal for generator g in time period t",
        )
        self._model.zdown = pyo.Var(
            self._model.G,
            self._model.T,
            domain=pyo.Binary,
            doc="Shutdown signal for generator g in time period t",
        )

    def _create_objective(self) -> None:
        def objective_rule(model):
            return sum(
                model.beta[i] * model.pout[i, t] for i in model.I for t in model.T
            ) + sum(
                model.cold_start[g] * model.zup[g, t] for g in model.G for t in model.T
            )

        self._model.objective = pyo.Objective(
            rule=objective_rule, sense=pyo.minimize, doc="Total cost"
        )

    def _create_constraints(self) -> None:
        self._model.power_balance = pyo.Constraint(
            self._model.T, rule=power_balance_rule, doc=power_balance_rule.__doc__
        )
        self._model.power_output = pyo.Constraint(
            self._model.I,
            self._model.T,
            rule=power_output_rule,
            doc=power_output_rule.__doc__,
        )
        self._model.power_output_min = pyo.Constraint(
            self._model.I,
            self._model.T,
            rule=power_output_min_rule,
            doc=power_output_min_rule.__doc__,
        )
        self._model.exclude_resource = pyo.Constraint(
            self._model.combined_cycle,
            self._model.T,
            rule=exclude_resource_rule,
            doc=exclude_resource_rule.__doc__,
        )

        # Start up and shut down constraints
        self._model.power_generation_decomposition = pyo.Constraint(
            self._model.G,
            self._model.T,
            rule=power_generation_decomposition,
            doc=power_generation_decomposition.__doc__,
        )
        self._model.exclusive_ramp_up_effective_constraint = pyo.Constraint(
            self._model.G,
            self._model.T,
            rule=exclusive_ramp_up_effective_constraint,
            doc=exclusive_ramp_up_effective_constraint.__doc__,
        )
        self._model.exclusive_ramp_down_effective_constraint = pyo.Constraint(
            self._model.G,
            self._model.T,
            rule=exclusive_ramp_down_effective_constraint,
            doc=exclusive_ramp_down_effective_constraint.__doc__,
        )
        self._model.start_up_shut_down_constraints = pyo.Constraint(
            self._model.G,
            self._model.T,
            rule=start_up_shut_down_constraints,
            doc=start_up_shut_down_constraints.__doc__,
        )
        self._model.start_up_zero_on_gen = pyo.Constraint(
            self._model.gen_on,
            self._model.T,
            rule=start_up_zero_on_gen,
            doc=start_up_zero_on_gen.__doc__,
        )
        self._model.shutdown_zero_off_gen = pyo.Constraint(
            self._model.gen_off,
            self._model.T,
            rule=shutdown_zero_off_gen,
            doc=shutdown_zero_off_gen.__doc__,
        )
        self._model.start_up_off_gen = pyo.Constraint(
            self._model.gen_off,
            self._model.T,
            rule=start_up_off_gen,
            doc=start_up_off_gen.__doc__,
        )
        self._model.shut_down_on_gen = pyo.Constraint(
            self._model.gen_on,
            self._model.T,
            rule=shut_down_on_gen,
            doc=shut_down_on_gen.__doc__,
        )

    def create_model(self, set_data: dict, param_data: dict) -> None:
        self._create_sets(set_data=set_data)
        self._create_parameters(param_data=param_data)
        self._create_variables()
        self._create_objective()
        self._create_constraints()
        if self._dispatch_config.dispatch_type == DispatchOptions.ideal:
            self._create_thermal_feature_constraints(
                set_data=set_data, param_data=param_data
            )
        if self._dispatch_config.dispatch_type == DispatchOptions.bess_preideal:
            self._add_bess_operation(set_data=set_data, param_data=param_data)
        if self._dispatch_config.dispatch_type == DispatchOptions.bess_ideal:
            self._create_thermal_feature_constraints(
                set_data=set_data, param_data=param_data
            )
            self._add_bess_operation(set_data=set_data, param_data=param_data)

    def solve(self, solver: str = "appsi_highs", solver_params: dict = {}, **kwargs):
        model_solver = pyo.SolverFactory(solver, **kwargs)
        return model_solver.solve(
            self._model,
            options=solver_params,
            # tee=True
        )

    def _create_thermal_feature_constraints(
        self, set_data: dict, param_data: dict
    ) -> None:
        # Minimum online time constraints
        self._model.minimum_online_time_rule_for_online_gen_rule = pyo.Constraint(
            self._model.gen_on,
            rule=minimum_online_time_rule_for_online_gen_rule,
            doc=minimum_online_time_rule_for_online_gen_rule.__doc__,
        )
        self._model.minimum_online_time_rule_for_offline_gen_rule = pyo.Constraint(
            self._model.G,
            self._model.T,
            rule=minimum_online_time_rule_for_offline_gen_rule,
            doc=minimum_online_time_rule_for_offline_gen_rule.__doc__,
        )
        self._model.minimum_online_time_for_offline_gen_rule_last_section = (
            pyo.Constraint(
                self._model.G,
                self._model.T,
                rule=minimum_online_time_for_offline_gen_rule_last_section,
                doc=minimum_online_time_for_offline_gen_rule_last_section.__doc__,
            )
        )

        # Ramps constraints
        self._model.up_ramps_thermal_gen = pyo.Constraint(
            self._model.G,
            self._model.T,
            rule=up_ramps_thermal_gen,
            doc=up_ramps_thermal_gen.__doc__,
        )

        self._model.down_ramps_thermal_gen = pyo.Constraint(
            self._model.G,
            self._model.T,
            rule=down_ramps_thermal_gen,
            doc=down_ramps_thermal_gen.__doc__,
        )

    def _add_bess_operation(self, set_data: dict, param_data: dict) -> None:
        """
        Create constraints, params and vars for battery energy storage system in dispatch
        """

        # --- Sets ---
        self._model.BESS = pyo.Set(
            doc="Battery Energy Storage System", initialize=set_data.get("BESS", [])
        )

        # --- Params ---
        self._model.bess_soc_0 = pyo.Param(
            self._model.BESS,
            domain=pyo.NonNegativeReals,
            doc="Initial state of charge of BESS",
            initialize=param_data.get("bess_soc_0", {}),
            default=0,
        )
        self._model.bess_charge_bid = pyo.Param(
            self._model.BESS,
            domain=pyo.NonNegativeReals,
            doc="Charge bid of BESS",
            initialize=param_data.get("bess_charge_bid", {}),
            default=1e6,
        )
        self._model.bess_discharge_bid = pyo.Param(
            self._model.BESS,
            domain=pyo.NonNegativeReals,
            doc="Discharge bid of BESS",
            initialize=param_data.get("bess_discharge_bid", {}),
            default=1e6,
        )
        self._model.bess_soc_bid = pyo.Param(
            self._model.BESS,
            domain=pyo.NonNegativeReals,
            doc="Discharge bid of BESS",
            initialize=param_data.get("bess_soc_bid", {}),
            default=1e6,
        )
        self._model.bess_min_soc = pyo.Param(
            self._model.BESS,
            domain=pyo.NonNegativeReals,
            doc="Minimum state of charge of BESS",
            initialize=param_data.get("bess_min_soc", {}),
            default=0,
        )
        self._model.bess_max_soc = pyo.Param(
            self._model.BESS,
            domain=pyo.NonNegativeReals,
            doc="Maximum state of charge of BESS",
            initialize=param_data.get("bess_max_soc", {}),
            default=1e6,
        )
        self._model.rt_eff = pyo.Param(
            self._model.BESS,
            domain=pyo.NonNegativeReals,
            doc="Efficiency of BESS",
            initialize=param_data.get("efficiency", {}),
            default=0.8,
        )
        self._model.bess_max_charge = pyo.Param(
            self._model.BESS,
            domain=pyo.NonNegativeReals,
            doc="Maximum charge of BESS",
            initialize=param_data.get("bess_max_charge", {}),
        )
        self._model.bess_max_discharge = pyo.Param(
            self._model.BESS,
            domain=pyo.NonNegativeReals,
            doc="Maximum discharge of BESS",
            initialize=param_data.get("bess_max_discharge", {}),
        )

        # --- Vars ---
        self._model.bess_charge = pyo.Var(
            self._model.BESS,
            self._model.T,
            domain=pyo.NonNegativeReals,
            doc="Charge of BESS in time period t",
        )
        self._model.bess_discharge = pyo.Var(
            self._model.BESS,
            self._model.T,
            domain=pyo.NonNegativeReals,
            doc="Discharge of BESS in time period t",
        )
        self._model.z_bess_charge = pyo.Var(
            self._model.BESS,
            self._model.T,
            domain=pyo.Binary,
            doc="Charge status of BESS in time period t",
        )
        self._model.z_bess_discharge = pyo.Var(
            self._model.BESS,
            self._model.T,
            domain=pyo.Binary,
            doc="Discharge status of BESS in time period t",
        )
        self._model.soc_bess = pyo.Var(
            self._model.BESS,
            self._model.T,
            domain=pyo.NonNegativeReals,
            doc="State of charge of BESS in time period t",
        )

        # --- Constraints ---
        self._model.compute_soc_bess = pyo.Constraint(
            self._model.BESS,
            self._model.T,
            rule=compute_soc_bess,
            doc=compute_soc_bess.__doc__,
        )

        self._model.min_soc_bess_constraint_rule = pyo.Constraint(
            self._model.BESS,
            self._model.T,
            rule=min_soc_bess_constraint_rule,
            doc=min_soc_bess_constraint_rule.__doc__,
        )

        self._model.max_soc_bess_constraint_rule = pyo.Constraint(
            self._model.BESS,
            self._model.T,
            rule=max_soc_bess_constraint_rule,
            doc=max_soc_bess_constraint_rule.__doc__,
        )

        self._model.bess_max_discharge_constraint_rule = pyo.Constraint(
            self._model.BESS,
            self._model.T,
            rule=bess_max_discharge_constraint_rule,
            doc=bess_max_discharge_constraint_rule.__doc__,
        )

        self._model.bess_max_charge_constraint_rule = pyo.Constraint(
            self._model.BESS,
            self._model.T,
            rule=bess_max_charge_constraint_rule,
            doc=bess_max_charge_constraint_rule.__doc__,
        )

        self._model.bess_avoid_concurrent_charge_discharge_constraint_rule = (
            pyo.Constraint(
                self._model.BESS,
                self._model.T,
                rule=bess_avoid_concurrent_charge_discharge_constraint_rule,
                doc=bess_avoid_concurrent_charge_discharge_constraint_rule.__doc__,
            )
        )

        # --- Modify previous demand constraints ---
        if hasattr(self._model, "power_balance"):
            self._model.del_component(self._model.power_balance)
        self._model.power_balance = pyo.Constraint(
            self._model.T,
            rule=power_balance_with_bess_rule,
            doc=power_balance_with_bess_rule.__doc__,
        )

        # ---objective---
        # deactivate all previous constraints objective
        if hasattr(self._model, "objective"):
            self._model.del_component(self._model.objective)
        self._model.objective = pyo.Objective(
            rule=maximize_social_welfare,
            doc=maximize_social_welfare.__doc__,
            sense=pyo.maximize,
        )
