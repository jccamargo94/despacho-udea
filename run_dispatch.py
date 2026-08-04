"""Legacy single-date dispatch runner.

The data-load + case-build logic now lives in app.pipeline.case_builder; this
module keeps the original solve + plotting + return signature so notebooks that
call run_dispatch() keep working. New code should prefer app.pipeline.runner /
the `python -m app run` CLI.
"""

from datetime import date
import json
from itertools import chain

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from thefuzz import process, fuzz
import plotly.express as px
import plotly.graph_objects as go

from app.model import UnitCommitmentModel
from app.pipeline.case_builder import build_case
from app.data.paths import resolve_input
from app.schemas.case import DispatchCase


def run_dispatch(
    case: DispatchCase,
    show_figs: bool = False,
    BESS: dict | None = None,
    DERS: int | None = None,
):
    from app.schemas.input_pack import InputPack, InputSource

    inputs = InputPack(dispatch_date=case.dispatch_date, source=InputSource.historical, data_dir="data")
    set_data, param_data, meta = build_case(case, inputs, ders=DERS)
    precio_bolsa = meta["precio_bolsa"]
    CC = meta["CC"]
    initial_condition_df = meta["initial_condition_df"]
    major_generators = meta["major_generators"]
    generators = meta["generators"]
    fixed_fuel_fire = meta["fixed_fuel_fire"]
    pmax_new_resources = meta["pmax_new_resources"]
    expansion_sources = meta["expansion_sources"]

    # ## 1.9 Solving model
    model = UnitCommitmentModel(case=case)
    model.create_model(set_data=set_data, param_data=param_data)

    results = model.solve(solver="cbc")

    # # 2. Check Results
    expr = model._model.objective.expr()
    print(f"F.obj: {expr:,.2f}")

    start_up = sum(
        model._model.cold_start[g] * model._model.zup[g, t].value
        for g in model._model.G
        for t in model._model.T
    )
    gen_cost = sum(
        model._model.beta[i] * model._model.pout[i, t].value
        for i in model._model.I
        for t in model._model.T
    )

    mpo_xm = pd.read_csv(resolve_input("iMAR", case.dispatch_date), header=None)
    mpo_xm = mpo_xm.iloc[0, 1:].values

    MPO = {
        ke.index(): model._model.objective.sense.value * pyo.value(dual_)
        for ke, dual_ in model._model.dual.items()
        if "power_balance" in ke.name
    }
    # Save the MPO from model
    mpo_df = pd.DataFrame(
        data=MPO.values(),
        index=pd.Index(MPO.keys(), name="datetime"),
        columns=[f"MPO {case.level.value} Modelo - DERs {DERS}"],
    )
    mpo_df.to_csv(
        f"data/results/MPO_{case.level.value}_{case.dispatch_date}.csv", sep=","
    )

    dispatch = {
        (gen, date_): pyo.value(dispatch)
        for (gen, date_), dispatch in model._model.pout.items()
    }
    dispatch = pd.DataFrame(
        data=dispatch.values(), index=dispatch.keys(), columns=["dispatch"]
    ).reset_index(drop=False, names=["generador", "datetime"])
    dispatch.to_csv(
        f"data/results/dispatch_by_gen-{case.dispatch_date}-{case.level.value}.csv",
        sep=",",
        index=False,
    )

    fixed_fuel_fire = fixed_fuel_fire.rename(columns={"gen": "xm_dispatch"})
    dispatch = dispatch.rename(columns={"dispatch": "udea_dispatch"})
    error_mapper = {
        gen: process.extractOne(
            query=gen.lower(),
            choices=fixed_fuel_fire["generator"].unique(),
            scorer=fuzz.partial_ratio,
            processor=lambda x: x.lower().replace(" ", ""),
        )[0]
        for gen in dispatch["generador"].unique()
    }
    with open("data/error_map.json", "r") as file:
        error_map = json.load(file)
    error_mapper |= error_map

    dispatch["generador_preideal"] = dispatch["generador"].apply(
        lambda x: error_mapper.get(x, x)
    )
    dispatch_merged = dispatch.merge(
        fixed_fuel_fire,
        left_on=["generador_preideal", "datetime"],
        right_on=["generator", "datetime"],
        how="left",
    )

    # --- Mask proelectrica ----
    proelec = dispatch_merged.loc[
        dispatch_merged["generador"].str.lower().str.contains("proelec"), :
    ]
    dispatch_merged = dispatch_merged.drop(index=proelec.index, axis=0)
    fixed_proelect = proelec.groupby("datetime").agg(
        {
            "generador": "first",
            "datetime": "first",
            "udea_dispatch": "sum",
            "generador_preideal": "first",
            "generator": "first",
            "hour": "mean",
            "xm_dispatch": "mean",
        }
    )

    dispatch_merged = pd.concat([dispatch_merged, fixed_proelect], axis=0)
    dispatch_merged["error"] = (
        dispatch_merged["udea_dispatch"] - dispatch_merged["xm_dispatch"]
    )

    available_CC = list(chain(*CC.values()))

    dispatched_cc = initial_condition_df[
        (initial_condition_df["Gpini-1"] > 0)
        & (initial_condition_df["Recurso"].isin(available_CC))
    ].Recurso.values
    delete_cc = set(available_CC) - set(dispatched_cc)
    dispatch_merged = dispatch_merged[~(dispatch_merged["generador"].isin(delete_cc))]
    dispatch_merged["legend_group"] = dispatch_merged["generador"].apply(
        lambda x: "major" if x in major_generators else "minor"
    )
    dispatch_merged = dispatch_merged.sort_values(["generador", "datetime"])

    if show_figs:
        fig = px.line(
            dispatch_merged,
            x="datetime",
            y="error",
            color="generador",
            title=f"Error de despacho por generador en el {case.dispatch_date}",
            hover_data=["xm_dispatch", "udea_dispatch"],
        )
        fig.write_html(
            f"data/results/error_dispatch-{case.dispatch_date}-{case.level.value}.html"
        )
        fig.show()

    if "preideal" in case.level.value:
        MPO_CHART = pd.DataFrame(
            data=mpo_xm, index=precio_bolsa["datetime"], columns=["MPO"]
        )
    else:
        MPO_CHART = (
            precio_bolsa.copy()
            .set_index(["datetime"])
            .rename(columns={"precio_bolsa": "MPO"})
        )

    # Add MPO from XM
    mpo_df[
        f"MPO {str(case.level.value).replace('bess_','')} XM"
    ] = MPO_CHART["MPO"].values
    if show_figs:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(MPO.keys()),
                y=list(MPO.values()),
                mode="lines",
                name=f"MPO {case.level.value} Modelo",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=MPO_CHART.index,
                y=MPO_CHART["MPO"],
                mode="lines",
                name=f"MPO {str(case.level.value).replace('bess_','')} XM",
                line={"dash": "dash"},
            )
        )
        fig.update_layout(
            xaxis_title="Hora",
            yaxis_title="Precio [COP/MWh]",
            width=800,
            height=600,
            xaxis=dict(dtick=3_600_000),
        )
        fig.show()

    if "bess" in case.level.value and show_figs:
        fig = go.Figure()
        for bess_name, bess_params in BESS.items():
            fig.add_traces(
                [
                    go.Bar(
                        x=model._model.T.ordered_data(),
                        y=[
                            pyo.value(val)
                            for _, val in model._model.bess_charge[
                                bess_name, :
                            ].expanded_items()
                        ],
                        name=f"Charging {bess_name}",
                    ),
                    go.Bar(
                        x=model._model.T.ordered_data(),
                        y=[
                            pyo.value(val)
                            for _, val in model._model.bess_discharge[
                                bess_name, :
                            ].expanded_items()
                        ],
                        name=f"Discharging {bess_name}",
                    ),
                    go.Scatter(
                        x=model._model.T.ordered_data(),
                        y=[
                            pyo.value(val)
                            for _, val in model._model.soc_bess[
                                bess_name, :
                            ].expanded_items()
                        ],
                        mode="lines",
                        name=f"SOC {bess_name}",
                        stackgroup="one",
                    ),
                ]
            )

        fig.update_layout(
            {
                "yaxis_title": "Potencia [MW]",
                "xaxis_title": "Fecha-Hora",
                "xaxis": dict(dtick=3_600_000),
            }
        )
        fig.show()
    return mpo_df, model, pmax_new_resources, expansion_sources
