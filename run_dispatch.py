from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
import json
import re
import os

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from thefuzz import process, fuzz
import plotly.express as px
import plotly.graph_objects as go


from itertools import chain
import plotly.express as px

import sys

from app.utils.misc import save_file, PARAMS as files_to_download
from app.model import UnitCommitmentModel, DispatchOptions, DispatchConfig


# config = DispatchConfig(
#     dispatch_type="preideal"

# )
# DISPATCH_DATE = date(2024,4,18)


def run_dispatch(config: DispatchConfig, DISPATCH_DATE: date, show_figs: bool = False, BESS : dict | None = None, DERS: int | None = None):
    CHECK_FOLDER = Path(f"data/{DISPATCH_DATE}")
    if CHECK_FOLDER.is_dir() and CHECK_FOLDER.exists():
        print("... files already downloaded. Skipping download")
    else:
        for file in files_to_download.keys():
            save_file(file_type=file, file_date=DISPATCH_DATE)

    price_pattern = r"P(\d+)"
    dispo_pattern = r"DISCONF(\d+)"

    # # 1. Load data

    # ## 1.1 Load initial data

    if config.dispatch_type == "ideal":
        dispo_come = pd.read_csv(
            "data/DispoCome_resource.csv", parse_dates=["datetime"], engine="pyarrow"
        )
    dispo = pd.read_csv(
        "data/dispo_declarada.csv", parse_dates=["datetime"], engine="pyarrow"
    )
    ofertas = pd.read_csv("data/ofertas.csv", parse_dates=["Date"], engine="pyarrow")
    demanda = pd.read_csv(
        "data/demaCome.csv", parse_dates=["datetime"], engine="pyarrow"
    )
    agc_asignado = pd.read_csv(
        "data/agc_asignado.csv", parse_dates=["datetime"], engine="pyarrow"
    )
    parametros_plantas = pd.read_csv("data/parametros_plantas.csv")

    # Precio bolsa
    precio_bolsa = pd.read_csv(
        "data/precio_bolsa/precio_bolsa_2024.csv",
        parse_dates=["datetime"],
        engine="pyarrow",
    )
    precio_bolsa["precio_bolsa"] = precio_bolsa["precio_bolsa"] * 1e3

    output = []
    MO = []
    CC = {}
    cc_price = {}
    cc_dispo = {}
    prices = {}
    with open(
        f"data/{DISPATCH_DATE}/OFEI{DISPATCH_DATE.month:0>2}{DISPATCH_DATE.day:0>2}.txt",
        "r",
    ) as file:
        for line in file:
            line = line.strip()
            if "PAP" in line:
                output.append(line)
            if "MO" in line:
                mo_line = line.split(",")
                if len(mo_line) > 2 and "MO" in mo_line[1]:
                    MO.append(mo_line)
            if (conf := re.findall(price_pattern, line)) and "CC" in line:
                fline = line.split(",")
                cc_price[f"{fline[0].strip()}_{conf[0]}"] = float(fline[2])
                if CC.get(fline[0].strip()):
                    CC[fline[0].strip()].append(f"{fline[0].strip()}_{conf[0]}")
                else:
                    CC[fline[0].strip()] = [f"{fline[0].strip()}_{conf[0]}"]
            # Disponibilidad CC
            if (conf := re.findall(dispo_pattern, line)) and "CC" in line:
                fline = line.split(",")
                cc_dispo[f"{fline[0].strip()}_{conf[0]}"] = [
                    int(disp) for disp in fline[2:]
                ]

            # Extract prices
            if "P" in line:
                pri = line.split(",")
                if (
                    len(pri) == 3
                    and " P" in pri[1]
                    and "u" not in pri[1].lower()
                    and "a" not in pri[1].lower()
                ):
                    prices[pri[0]] = float(pri[2]) * 1e-3

    precio_arranque = pd.DataFrame(
        [line.split(",") for line in output if "usd" not in line.lower()],
        columns=["resource", "type", "price"],
    )
    precio_arranque["price"] = precio_arranque["price"].astype(float)

    # Minimo operativo
    minimo_operativo = pd.DataFrame(
        MO,
        columns=[
            "resource",
            "type",
        ]
        + list(range(24)),
    )
    minimo_operativo = (
        minimo_operativo.set_index(["resource", "type"]).stack().reset_index()
    )
    minimo_operativo.columns = ["resource", "type", "hour", "minimo_operativo"]
    minimo_operativo["datetime"] = pd.to_datetime(DISPATCH_DATE) + pd.to_timedelta(
        minimo_operativo["hour"], unit="h"
    )
    minimo_operativo["minimo_operativo"] = minimo_operativo["minimo_operativo"].astype(
        float
    )
    minimo_operativo

    # ## 1.2 Filter data by date

    dispo = dispo[
        (dispo.datetime.dt.date == DISPATCH_DATE) & (dispo["resource_name"].notnull())
    ]
    dispo = dispo.drop_duplicates(subset=["resource_name", "datetime"])
    oferta_full = ofertas.copy()
    ofertas = ofertas[ofertas.Date.dt.date == DISPATCH_DATE]
    agc_asignado = agc_asignado[agc_asignado["datetime"].dt.date == DISPATCH_DATE]
    demanda = demanda[demanda["datetime"].dt.date == DISPATCH_DATE]
    precio_bolsa = precio_bolsa[precio_bolsa["datetime"].dt.date == DISPATCH_DATE]

    if config.dispatch_type == "ideal":
        dispo_come = dispo_come[
            (dispo_come.datetime.dt.date == DISPATCH_DATE)
            & (dispo_come["resource_name"].notnull())
        ]
        dispo_come = dispo_come.drop_duplicates(subset=["resource_name", "datetime"])
        for gen in dispo["resource_name"].unique():
            if gen in dispo_come["resource_name"].unique():
                serie = dispo_come[(dispo_come["resource_name"] == gen)]
                serie = (
                    serie.set_index("datetime")
                    .reindex(
                        pd.date_range(
                            start=DISPATCH_DATE,
                            end=DISPATCH_DATE + pd.Timedelta(days=1),
                            freq="1h",
                            inclusive="left",
                        )
                    )
                    .fillna(0)
                )
                dispo.loc[dispo["resource_name"] == gen, "dispo"] = serie[
                    "dispo"
                ].values
            else:
                print(
                    f"no existe el generador {gen} en disponibilidad comercial para el {DISPATCH_DATE}. Se asignará en 0"
                )
                dispo.loc[dispo["resource_name"] == gen, "dispo"] = 0

    # ## 1.3. Extract prices from OFEI

    # ### 1.3.1. Map names

    price_bid_map = {
        gen: process.extractOne(
            query=gen.lower(),
            choices=dispo["resource_name"].unique(),
            scorer=fuzz.token_sort_ratio,
            processor=lambda x: x.lower().replace(" ", ""),
            score_cutoff=70,
        )[0]
        for gen in prices.keys()
    }
    prices = {price_bid_map[gen]: price for gen, price in prices.items()}

    # ### 1.3.2. Transform bids

    ofertas["Value"] = ofertas.apply(
        lambda x: prices.get(x["resource_name"], float(x["Value"])), axis=1
    )
    ofertas

    # ofertas.loc[ofertas["resource_name"].str.contains("TEBSA"),"Value"] = 500.000
    # ofertas[ofertas["resource_name"].str.contains("TEBSA")]

    # import numpy as np
    # dispo.loc[dispo["resource_name"].str.contains("VALLE"),"dispo"] = np.array([239,  1,  1,  1,  1,  1,  1,  1,  1,  1,  239,  239,  239,  239,  239,  239,  239,  239,  239,  239,  239,  239,  239,  239])*1E3
    # ofertas.loc[ofertas["resource_name"].str.contains("VALLE"),"Value"] = 500.000

    # ofertas.loc[ofertas["resource_name"].str.contains("TEBSA"),"Value"] = 1514.537

    # ofertas.head()

    # ## 1.4. Get Initial conditions

    # Load Initial condition by plant and Units
    with open(
        f"data//{DISPATCH_DATE}/dCondIniP{DISPATCH_DATE.month:0>2}{DISPATCH_DATE.day:0>2}.txt",
        "r",
    ) as file:
        data = file.readlines()
        data = [line.strip().split(",") for line in data]
        headers = data.pop(0)
    condicion_inicial_planta = pd.DataFrame(data, columns=headers)

    with open(
        f"data/{DISPATCH_DATE}/dCondIniU{DISPATCH_DATE.month:0>2}{DISPATCH_DATE.day:0>2}.txt",
        "r",
    ) as file:
        data = file.readlines()
        data = [line.strip().split(",") for line in data]
        headers = data.pop(0)

    # Transform dataframe
    condicion_inicial_unidad = pd.DataFrame(data, columns=headers)
    # Generate name mappes
    condicion_inicial_map = {
        gen: process.extractOne(
            query=gen.lower(),
            choices=dispo["resource_name"].unique(),
            scorer=fuzz.token_sort_ratio,
            processor=lambda x: x.lower().replace(" ", ""),
            # score_cutoff=70,
        )[0]
        for gen in condicion_inicial_planta.Recurso.unique()
    }
    # FIX some maps
    condicion_inicial_map |= {
        "FLORES IV": "FLORES 4 CC",
        "TSIERRA": "TERMOSIERRA CC",
        "GUAJIR21": "GUAJIRA 2",
    }
    condicion_inicial_planta["Recurso"] = condicion_inicial_planta["Recurso"].apply(
        lambda x: condicion_inicial_map.get(x, x)
    )

    # ## 1.5 Generating new resources for CC plants

    # ### 1.5.1. New CC resources

    # DROP previous CC
    CC_MAP = {
        gen: process.extractOne(
            query=gen.lower(),
            choices=dispo["resource_name"].unique(),
            scorer=fuzz.partial_token_sort_ratio,
            processor=lambda x: x.lower().replace(" ", ""),
            score_cutoff=70,
        )[0]
        for gen in CC.keys()
    }
    CC_MAP

    dispo = dispo[~dispo["resource_name"].isin(list(CC_MAP.values()))]
    ofertas = ofertas[~ofertas["resource_name"].isin(list(CC_MAP.values()))]

    # INCLUDING CC RESOURCE in DISPO and OFERTAS
    new_cc_resources = pd.DataFrame(cc_dispo).stack().reset_index()
    new_cc_resources.columns = ["hours", "resource_name", "dispo"]
    new_cc_resources["dispo"] = new_cc_resources["dispo"] * 1e3
    new_cc_resources["hours"] = new_cc_resources["hours"].astype(int)
    new_cc_resources["datetime"] = pd.to_datetime(DISPATCH_DATE) + pd.to_timedelta(
        new_cc_resources["hours"], unit="h"
    )
    new_cc_resources["gen_type"] = "TERMICA"
    new_cc_resources["dispatched"] = "DESPACHADO CENTRALMENTE"
    new_cc_resources["company_activity"] = "GENERACIÓN"
    new_cc_resources.pop("hours")

    # OFERTAS

    new_cc_bid = pd.DataFrame(cc_price, index=[1]).stack().reset_index(drop=False)
    new_cc_bid.columns = ["index_", "resource_name", "Value"]
    new_cc_bid["Value"] = new_cc_bid["Value"].apply(lambda x: x * 1e-3)
    # new_cc_bid["datetime"] = pd.to_datetime(DISPATCH_DATE) + pd.to_timedelta(new_cc_bid["hours"], unit="h")
    new_cc_bid["resource_gen_type"] = "TERMICA"
    new_cc_bid["Date"] = DISPATCH_DATE
    # new_cc_bid["dispatched"] = "DESPACHADO CENTRALMENTE"
    # new_cc_bid["company_activity"] = "GENERACIÓN"
    _ = new_cc_bid.pop("index_")

    dispo = pd.concat([dispo, new_cc_resources], axis=0)
    ofertas = pd.concat([ofertas, new_cc_bid], axis=0)

    # ### 1.5.2 Adding units for each CC resource

    CC_MAP_inv = {v: k for k, v in CC_MAP.items()}

    dcondIniPlant = condicion_inicial_planta[
        condicion_inicial_planta.Recurso.isin(CC_MAP.values())
    ]
    dcondIniPlant.loc[:, "Recurso"] = dcondIniPlant["Recurso"].apply(
        lambda x: CC_MAP_inv.get(x, x)
    )
    dcondIniPlant.loc[:, "dispatched_conf"] = dcondIniPlant.loc[:, "Conf_Pini-1"].apply(
        lambda x: int(re.findall(r"\d+", x)[0])
    )
    # dcondIniPlant = dcondIniPlant[dcondIniPlant["dispatched_conf"]>0]
    dcondIniPlant

    initial_condition_df = pd.DataFrame()
    for plant, cc_plants in deepcopy(CC).items():
        filtered_init_condition = dcondIniPlant.query("Recurso == @plant").reset_index()
        dispatched_conf = filtered_init_condition.loc[0, "dispatched_conf"]
        if filtered_init_condition.loc[0, "dispatched_conf"] != 0:
            filtered_init_condition.loc[0, "Recurso"] = f"{plant}_{dispatched_conf}"
            dispatched_config = f"{plant}_{dispatched_conf}"
            cc_plants.pop(cc_plants.index(dispatched_config))
        to_concat = [filtered_init_condition for _ in cc_plants]
        if to_concat:
            filtered_init_condition_ = pd.concat(to_concat)
            filtered_init_condition_["Recurso"] = cc_plants
            filtered_init_condition_["Gpini-1"] = 0
            filtered_init_condition = pd.concat(
                [filtered_init_condition, filtered_init_condition_], ignore_index=True
            )
            filtered_init_condition = filtered_init_condition[
                ~filtered_init_condition["Recurso"].isin([plant])
            ]
        initial_condition_df = pd.concat(
            [initial_condition_df, filtered_init_condition], ignore_index=True
        )

    condicion_inicial_planta_termicas = condicion_inicial_planta[
        ~(condicion_inicial_planta["Tipo"] == "H")
        & ~(condicion_inicial_planta["Recurso"].isin(CC_MAP.values()))
    ]
    initial_condition_df = pd.concat(
        [initial_condition_df, condicion_inicial_planta_termicas], ignore_index=True
    )
    initial_condition_df = initial_condition_df.astype(
        {"T_CONF_Pini-1": int, "Gpini-1": float}
    )

    # ## 1.6. Generating initial set to model
        # Thermal gen
    gen_on = initial_condition_df[initial_condition_df["Gpini-1"] != 0][
        "Recurso"
    ].unique()
    needed_generators = [gen for gen in list(gen_on) if gen not in ofertas.resource_name.unique()]
    for gen in needed_generators:
        gen_oferta = oferta_full.query("resource_name == @gen").head(1).reset_index(drop=True)
        gen_oferta.loc[0,"Date"] = pd.Timestamp(DISPATCH_DATE)
        ofertas = pd.concat([ofertas, gen_oferta], axis=0)

    major_generators = ofertas.resource_name.unique()
    generators = dispo.resource_name.unique()
    timestamps = demanda["datetime"].to_dict().values()
    # fuel_generators = dispo.query('resource_name in @major_generators and gen_type=="TERMICA"').resource_name.unique()
    fuel_generators = dispo[
        (dispo["resource_name"].isin(major_generators))
        & (dispo["gen_type"] == "TERMICA")
    ].resource_name.unique()


    gen_off = list(set(fuel_generators) - set(gen_on))
    # Replicate the first condition for needed generators

    # ## 1.7. Get startup/shutdown costs

    MO_map = {
        gen: results[0]
        for gen in minimo_operativo.resource.unique()
        if (
            results := process.extractOne(
                query=gen.lower(),
                choices=generators,
                # choices=major_generators.tolist(),
                scorer=fuzz.token_sort_ratio,
                processor=lambda x: x.lower().replace(" ", ""),
                score_cutoff=70,
            )
        )
    }
    minimo_operativo["resource"] = minimo_operativo["resource"].apply(
        lambda x: MO_map.get(x, x)
    )
    minimo_operativo

    generators_pap_map = {
        gen: process.extractOne(
            query=gen.lower(),
            choices=precio_arranque.resource.unique(),
            scorer=fuzz.partial_token_sort_ratio,
            processor=lambda x: x.lower().replace(" ", ""),
            score_cutoff=70,
        )[0]
        for gen in fuel_generators
    }

    cold_start = {}
    for gen in fuel_generators:
        gen_name_mapped = generators_pap_map[gen]
        gen_pap = precio_arranque[
            (precio_arranque["resource"] == gen_name_mapped)
            & (precio_arranque.type.str.contains("C"))
        ]["price"].values[0]
        cold_start[gen] = float(gen_pap)

    # Valores en MWh
    Pmax = (
        dispo.query("resource_name in @generators")
        .set_index(["resource_name", "datetime"])
        .sort_index()["dispo"]
        * 1e-3
    )
    Pmin = minimo_operativo.set_index(["resource", "datetime"]).sort_index()[
        "minimo_operativo"
    ]
    beta = (
        ofertas.query("resource_name in @generators")
        .set_index(["resource_name"])
        .sort_index()["Value"]
        * 1e3
    )
    agc_indexed = agc_asignado.set_index(["recurso", "datetime"])["agc"] * 1e-3

    # Pmax.loc[agc_indexed.index] = Pmax.loc[agc_indexed.index] -  agc_indexed

    demand_pronos = pd.read_csv(
        f"data/{DISPATCH_DATE}/PrId{DISPATCH_DATE.month:0>2}{DISPATCH_DATE.day:0>2}_NAL.txt",
        header=None,
        encoding="latin1",
    )
    demand_pronos = demand_pronos.iloc[:, 1:].sum().values

    demand_pronos = dict(zip(demanda["datetime"], demand_pronos))

    Ton = initial_condition_df.set_index(["Recurso"]).query("Recurso in @gen_on")[
        "T_CONF_Pini-1"
    ]
    Ton = Ton[Ton.index.isin(fuel_generators)]

    z_on_t0_minus_1 = {
        gen: 1
        for gen in initial_condition_df[initial_condition_df["Gpini-1"] > 0][
            "Recurso"
        ].unique()
    }

    z_on_t0_minus_1 = {k: v for k, v in z_on_t0_minus_1.items() if k in fuel_generators}

    # ## 1.8 Fix fuel-fire generators to check

    fixed_fuel_fire = pd.read_csv(
        f"data/{DISPATCH_DATE}/PrId{DISPATCH_DATE.month:0>2}{DISPATCH_DATE.day:0>2}_NAL.txt",
        header=None,
        encoding="latin1",
    )
    fixed_fuel_fire.columns = ["generator"] + list(range(24))
    fixed_fuel_fire = fixed_fuel_fire.set_index("generator").stack().reset_index()
    fixed_fuel_fire.columns = ["generator", "hour", "gen"]
    fixed_fuel_fire["datetime"] = pd.to_datetime(DISPATCH_DATE) + pd.to_timedelta(
        fixed_fuel_fire["hour"], unit="h"
    )

    # Fix generation
    fixed_fuel_fired_map = {}
    for gen in fixed_fuel_fire.generator.unique():
        # if not (
        #     str(gen).startswith("AG_") or
        #     str(gen).startswith("M") or
        #     str(gen).startswith("GD") or
        #     str(gen).startswith("AR")
        # ):
        choice = process.extractOne(
            query=gen.lower(),
            choices=generators,
            scorer=fuzz.partial_ratio,
            processor=lambda x: x.lower().replace(" ", ""),
            # score_cutoff=60,
        )
        if choice and choice[0] in generators:
            fixed_fuel_fired_map[gen] = choice[0]
        else:
            ...
            # print(f"{gen} select {choice} but is not a fuel generator")

    fixed_fuel_fire_2 = fixed_fuel_fire.copy()
    with open("data/preideal_dispatch_map.json", "r", encoding="utf-8") as file:
        preideal_dispatch_map = json.load(file)
    fixed_fuel_fire_2["generador_model"] = fixed_fuel_fire_2["generator"].apply(
        lambda x: preideal_dispatch_map.get(x, "")
    )
    fixed_fuel_fire_2 = fixed_fuel_fire_2[
        (fixed_fuel_fire_2["generador_model"].notnull())
        & (fixed_fuel_fire_2["generador_model"] != "")
        & ~(fixed_fuel_fire_2["generador_model"].isin(major_generators))
    ]
    fixed_fuel_fire_2 = fixed_fuel_fire_2.set_index(["generador_model", "datetime"])[
        "gen"
    ]

    Pmax_model = Pmax.apply(lambda x: np.round(x, 0)).to_dict()

    if "preideal" in config.dispatch_type:
        Pmax_model.update(fixed_fuel_fire_2[fixed_fuel_fire_2.index.get_level_values(0).isin(generators)].to_dict())

    # --- RAMPS ---
    with open("data/ramps.json", "r") as file:
        ramps = json.load(file)

    DEMANDA = (
        demand_pronos
        if "preideal" in config.dispatch_type
        else (demanda.set_index("datetime")["dema"] * 1e-3).astype(int)
    )
    MAX_MIN_OP = 1 if "preideal" in config.dispatch_type else 0
    TMG = (
        parametros_plantas[parametros_plantas["generador"].isin(fuel_generators)]
        .set_index("generador")["TMG"]
        .astype(int)
    )

    ramps = {k: v for k, v in ramps.items() if k in fuel_generators}


    # Add ders
    pmax_new_resources = pd.DataFrame()
    expansion_sources = list()
    if DERS:
        DERS = str(DERS)
        new_resources_df = pd.read_excel("data/Supuestos Modelo de despacho.xlsx", sheet_name="series")
        expansion_sources = [col for col in new_resources_df.columns if DERS in col]
        pmax_new_resources = new_resources_df[expansion_sources]
        pmax_new_resources.index = pd.Index(pd.to_datetime(DISPATCH_DATE) + pd.to_timedelta(new_resources_df.hours, unit="h"), name="datetime")
        pmax_new_resources = pmax_new_resources.stack().reset_index()
        pmax_new_resources.columns = ["datetime", "resource_name", "dispo"]
        Pmax_model.update(pmax_new_resources.set_index(["resource_name","datetime",]).to_dict()["dispo"])
        generators = generators.tolist() + expansion_sources


    set_data = {
        "G": fuel_generators,
        "T": timestamps,
        "I": generators,
        "combined_cycle": list(CC.keys()),
        "excluded_resource": CC,
        "gen_on": gen_on,
        "gen_off": gen_off,
    }

    param_data = {
        "Pmax": Pmax_model,
        # "Pmin" : Pmin,
        "Pmin": {},
        "beta": beta,
        "cold_start": cold_start,
        "demand": DEMANDA,
        "Ton": Ton,
        "z_on_t0_minus_1": z_on_t0_minus_1,
        "TMG": TMG,
        "ramp_up": ramps,
        "ramp_down": ramps,
        "max_min_op": MAX_MIN_OP,
    }



    if config.dispatch_type in [
        DispatchOptions.bess_ideal,
        DispatchOptions.bess_preideal,
        DispatchOptions.bess_preideal_resource,
        DispatchOptions.bess_ideal_resource,
    ]:
        set_data.update(**{"BESS": list(BESS.keys())})
        BESS_PARAMS_NAMES = [
            "bess_soc_0",
            "bess_charge_bid",
            "bess_discharge_bid",
            "bess_soc_bid",
            "bess_min_soc",
            "bess_max_soc",
            "efficiency",
            "bess_max_charge",
            "bess_max_discharge",
        ]
        bess_params_model = dict(
            zip(BESS_PARAMS_NAMES, [{} for _ in BESS_PARAMS_NAMES])
        )
        for bess_name, bess_params in BESS.items():
            bess_params_model["bess_soc_0"].update(
                **{bess_name: bess_params["initial_soc"] * bess_params["MWh_nom"]}
            )
            bess_params_model["bess_charge_bid"].update(
                **{bess_name: bess_params["charge_bid"]}
            )
            bess_params_model["bess_discharge_bid"].update(
                **{bess_name: bess_params["discharge_bid"]}
            )
            bess_params_model["bess_min_soc"].update(
                **{bess_name: bess_params["min_soc"] * bess_params["MWh_nom"]}
            )
            bess_params_model["bess_max_soc"].update(
                **{bess_name: bess_params["max_soc"] * bess_params["MWh_nom"]}
            )
            bess_params_model["efficiency"].update(
                **{bess_name: bess_params["efficiency"]}
            )
            bess_params_model["bess_max_charge"].update(
                **{bess_name: bess_params["MWh_nom"] / bess_params["hours_to_deplete"]}
            )
            bess_params_model["bess_max_discharge"].update(
                **{bess_name: bess_params["MWh_nom"] / bess_params["hours_to_deplete"]}
            )

        param_data.update(**bess_params_model)

    # ## 1.9 Solving model

    model = UnitCommitmentModel(config=config)
    model.create_model(set_data=set_data, param_data=param_data)

    # model._model.pout["ALBAN",[pd.Timestamp("2024-04-25 18:00:00")]].fix(388)
    # model._model.pout["ALBAN",[pd.Timestamp("2024-04-25 19:00:00")]].fix(388)
    # model._model.pout["ALBAN",[pd.Timestamp("2024-04-25 20:00:00")]].fix(353)

    # model._model.pout["SOGAMOSO",[pd.Timestamp("2024-04-25 18:00:00")]].fix(3)
    # model._model.pout["SOGAMOSO",[pd.Timestamp("2024-04-25 19:00:00")]].fix(89)

    # results = model.solve(solver="cplex", executable="solver/cplex")

    # model._model.z.fix()

    # results = model.solve(solver="cplex", executable="solver/cplex")

    results = model.solve(solver="cbc")
    # print("...Fixing Var for second solve")
    # for component in model._model.component_data_objects(pyo.Var, active=True):
    #     if not component.is_continuous():
    #         # print (f"fixing {component}") 
    #         component.fix()
    # results = model.solve(solver="cplex")

    # # ===== WARNING FIXING VARIABLES =====
    # for gen, model_gen_name in fix_fuel_fired_gen_.items():
    #     # Filter data
    #     serie = fixed_fuel_fire[fixed_fuel_fire["generator"]==gen]
    #     serie["generator"] = model_gen_name
    #     for k,v in serie.set_index(["generator", "datetime"])["gen"].to_dict().items():
    #         model._model.pout[k].fix(v)

    # for t in model._model.T:
    #     model._model.pout["TERMONORTE",t].fix(0)

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

    # print(f"f.o.{start_up + gen_cost:,.2f}")

    mpo_xm = pd.read_csv(
        f"data/{DISPATCH_DATE}/iMAR{DISPATCH_DATE.month:0>2}{DISPATCH_DATE.day:0>2}_NAL.txt",
        header=None,
    )
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
        columns=[f"MPO {config.dispatch_type.value} Modelo - DERs {DERS}"],
    )
    mpo_df.to_csv(f"data/results/MPO_{config.dispatch_type.value}_{DISPATCH_DATE}.csv", sep=",")



    dispatch = {
        (gen, date_): pyo.value(dispatch)
        for (gen, date_), dispatch in model._model.pout.items()
    }
    dispatch = pd.DataFrame(
        data=dispatch.values(), index=dispatch.keys(), columns=["dispatch"]
    ).reset_index(drop=False, names=["generador", "datetime"])
    dispatch.to_csv(
        f"data/results/dispatch_by_gen-{DISPATCH_DATE}-{config.dispatch_type.value}.csv",
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
            # score_cutoff=60,
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
            # legendgroup="legend_group",
            title=f"Error de despacho por generador en el {DISPATCH_DATE}",
            hover_data=["xm_dispatch", "udea_dispatch"],
        )

        fig.write_html(
            f"data/results/error_dispatch-{DISPATCH_DATE}-{config.dispatch_type.value}.html"
        )
        fig.show()

    if "preideal" in config.dispatch_type:
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
        f"MPO {str(config.dispatch_type.value).replace('bess_','')} XM"
    ] = MPO_CHART["MPO"].values
    if show_figs:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(MPO.keys()),
                y=list(MPO.values()),
                mode="lines",
                name=f"MPO {config.dispatch_type.value} Modelo",
                
            )
        )
        fig.add_trace(
            go.Scatter(
                x=MPO_CHART.index,
                y=MPO_CHART["MPO"],
                mode="lines",
                name=f"MPO {str(config.dispatch_type.value).replace('bess_','')} XM",
                line={"dash": "dash"},
            )
        )
        

        fig.update_layout(
            # title=f"Precio Bolsa {DISPATCH_DATE}",
            xaxis_title="Hora",
            yaxis_title="Precio [COP/MWh]",
            width=800,
            height=600,
            xaxis=dict(
                dtick=3_600_000,
            ),
        )
        fig.show()

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)

    # pd.DataFrame(data=MPO, index=[f"MPO-{config.dispatch_type.value}"]).T.plot(kind="line",ax=ax)

    # if config.dispatch_type == "ideal":
    #     precio_bolsa.plot(kind="line", x="datetime", y="precio_bolsa", ax=ax, linestyle='-.')
    # else:
    #     pd.DataFrame(data=mpo_xm, index=timestamps, columns=["MPO_XM"]).plot(kind="line", ax=ax, linestyle='--')

    # plt.show()
    # # pd.DataFrame(data=mpo_xm, index=timestamps, columns=["MPO_XM"]).plot(kind="line", ax=ax, linestyle='--')

    if "bess" in config.dispatch_type.value and show_figs:
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
                        # mode="lines",
                        name=f"Charging {bess_name}",
                        # stackgroup="one",
                    ),
                    go.Bar(
                        x=model._model.T.ordered_data(),
                        y=[
                            pyo.value(val)
                            for _, val in model._model.bess_discharge[
                                bess_name, :
                            ].expanded_items()
                        ],
                        # mode="lines",
                        name=f"Discharging {bess_name}",
                        # stackgroup="one",
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
                "xaxis": dict(
                    # tickformat="%-d-%-H",
                    dtick=3_600_000,
                ),
            }
        )
        fig.show()
    return mpo_df, model, pmax_new_resources, expansion_sources
