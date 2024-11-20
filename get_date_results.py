from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
import itertools
import json
import re
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
from thefuzz import process, fuzz

import sys

from app.model import UnitCommitmentModel, DispatchOptions, DispatchConfig


price_pattern = r"P(\d+)"
dispo_pattern = r"DISCONF(\d+)"


# --- Get dates to run all dates ---
PATH = Path(os.path.join("./", "data", "condicion_inicial"))
dates_ = []
for folder in PATH.glob("*"):
    if folder.is_dir():
        date_name_list = folder.stem.split("-")
        date_ = date(*[int(d) for d in date_name_list])
        dates_.append(date_)

config = DispatchConfig(dispatch_type="preideal")

SKIP_DATES = [
    # Fechas malas
    date(2024, 10, 3),
    date(2024, 10, 9),
    date(2024, 10, 30),
    date(2024, 10, 17),
    date(2024, 2, 15),
    # Ya ejecutadas
    date(2024, 3, 23),
    date(2024, 3, 29),
    date(2024, 4, 22),
    date(2024, 4, 25),
    date(2024, 4, 14),
    date(2024, 4, 18),
    date(2024, 4, 19),
    date(2024, 5, 25),
    date(2024, 6, 9),
    date(2024, 7, 2),
    date(2024, 8, 10),
    date(2024, 8, 29),
]
for DISPATCH_DATE, DISPATCH_TYPE in itertools.product(
    dates_, DispatchOptions._member_names_
):
    if DISPATCH_DATE in SKIP_DATES:
        print(f"Skipping date {DISPATCH_DATE}")
        continue
    print(f"Running a case for {DISPATCH_DATE} with {DISPATCH_TYPE} dispatch type")
    config = DispatchConfig(dispatch_type=DISPATCH_TYPE)

    print("...Loading some data")
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
    print("...Computing data from OFEI")
    with open(
        f"data/oferta_inicial/OFEI{DISPATCH_DATE.month:0>2}{DISPATCH_DATE.day:0>2}.txt",
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
    print("...Computing minimum allowable power output")
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

    # --- Get availability ---
    dispo = dispo[
        (dispo.datetime.dt.date == DISPATCH_DATE) & (dispo["resource_name"].notnull())
    ]
    dispo = dispo.drop_duplicates(subset=["resource_name", "datetime"])

    ofertas = ofertas[ofertas.Date.dt.date == DISPATCH_DATE]
    agc_asignado = agc_asignado[agc_asignado["datetime"].dt.date == DISPATCH_DATE]
    demanda = demanda[demanda["datetime"].dt.date == DISPATCH_DATE]
    precio_bolsa = precio_bolsa[precio_bolsa["datetime"].dt.date == DISPATCH_DATE]

    # --- Read comercial availability ---

    if config.dispatch_type == "ideal":
        print("...Ideal dispatch selected: Loading commercial availability")
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

    # --- Map names ---
    print("...Mapping generators names to match")
    price_bid_map = {
        gen: process.extractOne(
            query=gen.lower(),
            choices=dispo["resource_name"].unique(),
            scorer=fuzz.token_sort_ratio,
            processor=lambda x: x.lower().replace(" ", ""),
            score_cutoff=70,
        )[0]
        for gen in prices
    }
    prices = {price_bid_map[gen]: price for gen, price in prices.items()}
    # Transform bids
    ofertas["Value"] = ofertas.apply(
        lambda x: prices.get(x["resource_name"], float(x["Value"])), axis=1
    )

    # Load Initial condition by plant and Units
    print("...Loading initial condition for resources")
    with open(
        f"data/condicion_inicial/{DISPATCH_DATE}/dCondIniP{DISPATCH_DATE.month:0>2}{DISPATCH_DATE.day:0>2}.txt",
        "r",
    ) as file:
        data = file.readlines()
        data = [line.strip().split(",") for line in data]
        headers = data.pop(0)
    condicion_inicial_planta = pd.DataFrame(data, columns=headers)
    print("...Loading initial condition by units")
    with open(
        f"data/condicion_inicial/{DISPATCH_DATE}/dCondIniU{DISPATCH_DATE.month:0>2}{DISPATCH_DATE.day:0>2}.txt",
        "r",
    ) as file:
        data = file.readlines()
        data = [line.strip().split(",") for line in data]
        headers = data.pop(0)

    # Transform dataframe
    print("...Mapping resources and units names")
    condicion_inicial_unidad = pd.DataFrame(data, columns=headers)
    condicion_inicial_map = {
        gen: process.extractOne(
            query=gen.lower(),
            choices=dispo["resource_name"].unique(),
            scorer=fuzz.token_sort_ratio,
            processor=lambda x: x.lower().replace(" ", ""),
        )[0]
        for gen in condicion_inicial_planta.Recurso.unique()
    } | {
        "FLORES IV": "FLORES 4 CC",
        "TSIERRA": "TERMOSIERRA CC",
        "GUAJIR21": "GUAJIRA 2",
    }
    condicion_inicial_planta["Recurso"] = condicion_inicial_planta["Recurso"].apply(
        lambda x: condicion_inicial_map.get(x, x)
    )

    # CC resources
    # DROP previous CC
    print("...Mapping resources names for combined heat-and-power")
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
    dispo = dispo[~dispo["resource_name"].isin(list(CC_MAP.values()))]
    ofertas = ofertas[~ofertas["resource_name"].isin(list(CC_MAP.values()))]
    # INCLUDING CC RESOURCE in DISPO and OFERTAS
    print("...Generating new resources for CC plant configurations")
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
    print("...Getting bids for CC plant configurations")
    new_cc_bid = pd.DataFrame(cc_price, index=[1]).stack().reset_index(drop=False)
    new_cc_bid.columns = ["index_", "resource_name", "Value"]
    new_cc_bid["Value"] = new_cc_bid["Value"].apply(lambda x: x * 1e-3)
    # new_cc_bid["datetime"] = pd.to_datetime(DISPATCH_DATE) + pd.to_timedelta(new_cc_bid["hours"], unit="h")
    new_cc_bid["resource_gen_type"] = "TERMICA"
    new_cc_bid["Date"] = DISPATCH_DATE
    # new_cc_bid["dispatched"] = "DESPACHADO CENTRALMENTE"
    # new_cc_bid["company_activity"] = "GENERACIÓN"
    _ = new_cc_bid.pop("index_")

    # Concatenate new disponibility and bids
    print("...Using new availability and bids for CC plant configurations")
    dispo = pd.concat([dispo, new_cc_resources], axis=0)
    ofertas = pd.concat([ofertas, new_cc_bid], axis=0)

    # --- Get inverse CC mappers---
    CC_MAP_inv = {v: k for k, v in CC_MAP.items()}

    # --- Get initial condition for CC ---
    print("...Get initial condition for CC plant configurations")
    dcondIniPlant = condicion_inicial_planta[
        condicion_inicial_planta.Recurso.isin(CC_MAP.values())
    ]
    dcondIniPlant.loc[:, "Recurso"] = dcondIniPlant["Recurso"].apply(
        lambda x: CC_MAP_inv.get(x, x)
    )
    dcondIniPlant.loc[:, "dispatched_conf"] = dcondIniPlant.loc[:, "Conf_Pini-1"].apply(
        lambda x: int(re.findall(r"\d+", x)[0])
    )
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

    # Generating pyomo sets
    print("...Generating values to pyomo model")
    major_generators = ofertas.resource_name.unique()
    generators = dispo.resource_name.unique()
    timestamps = demanda["datetime"].to_dict().values()
    # fuel_generators = dispo.query('resource_name in @major_generators and gen_type=="TERMICA"').resource_name.unique()
    fuel_generators = dispo[
        (dispo["resource_name"].isin(major_generators))
        & (dispo["gen_type"] == "TERMICA")
    ].resource_name.unique()

    # Thermal gen
    gen_on = initial_condition_df[initial_condition_df["Gpini-1"] != 0][
        "Recurso"
    ].unique()
    gen_off = list(set(fuel_generators) - set(gen_on))

    # --- Minium generation ----
    MO_map = {
        gen: results[0]
        for gen in minimo_operativo.resource.unique()
        if (
            results := process.extractOne(
                query=gen.lower(),
                choices=generators,
                scorer=fuzz.token_sort_ratio,
                processor=lambda x: x.lower().replace(" ", ""),
                score_cutoff=70,
            )
        )
    }
    minimo_operativo["resource"] = minimo_operativo["resource"].apply(
        lambda x: MO_map.get(x, x)
    )
    # --- PAP ----
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

    # --- Params for pyomo model ---
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

    # --- Load demand forecasting ----
    demand_pronos = pd.read_csv(
        f"data/preideal_dispatch/{DISPATCH_DATE}.txt", header=None, encoding="latin1"
    )
    demand_pronos = demand_pronos.iloc[:, 1:].sum().values
    demand_pronos = dict(zip(demanda["datetime"], demand_pronos))
    # On generation at t0-1
    Ton = initial_condition_df.set_index(["Recurso"]).query("Recurso in @gen_on")[
        "T_CONF_Pini-1"
    ]
    # Dispatched generators at t0-1
    z_on_t0_minus_1 = {
        gen: 1
        for gen in initial_condition_df[initial_condition_df["Gpini-1"] > 0][
            "Recurso"
        ].unique()
    }
    # Fix Fuel-fired generators to check
    fixed_fuel_fire = pd.read_csv(
        f"data/preideal_dispatch/{DISPATCH_DATE}.txt", header=None, encoding="latin1"
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
        if not (
            str(gen).startswith("AG_")
            or str(gen).startswith("M")
            or str(gen).startswith("GD")
            or str(gen).startswith("AR")
        ):
            choice = process.extractOne(
                query=gen.lower(),
                choices=generators,
                scorer=fuzz.partial_token_sort_ratio,
                processor=lambda x: x.lower().replace(" ", ""),
                # score_cutoff=60,
            )[0]
            if choice in fuel_generators:
                fixed_fuel_fired_map[gen] = choice
            else:
                ...
                # print(f"{gen} select {choice} but is not a fuel generator")

    # --- Load Ramps ---
    print("...Loading resource ramps")
    with open("data/ramps.json", "r") as file:
        ramps = json.load(file)

    # --- Load data based in dispatch type ---
    DEMANDA = (
        demand_pronos
        if config.dispatch_type == "preideal"
        else (demanda.set_index("datetime")["dema"] * 1e-3).astype(int)
    )
    MAX_MIN_OP = 1 if config.dispatch_type == "preideal" else 0
    TMG = (
        parametros_plantas[parametros_plantas["generador"].isin(fuel_generators)]
        .set_index("generador")["TMG"]
        .astype(int)
    )

    # --- Generating pyomo data ----
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
        "Pmax": Pmax.apply(lambda x: np.round(x, 0)),
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

    # --- Create model ---
    print("...Creating pyomo model")
    model = UnitCommitmentModel(config=config)
    model.create_model(set_data=set_data, param_data=param_data)
    print("...Solving Unit Commitment")
    results = model.solve(solver="cbc")
    expr = model._model.objective.expr()
    print(f"F.obj: {expr:,.2f}")

    # --- Load values ----
    mpo_xm = pd.read_csv(f"data/preideal_price/{DISPATCH_DATE}.txt", header=None)
    mpo_xm = mpo_xm.iloc[0, 1:].values
    MPO = {
        ke.index(): pyo.value(dual_)
        for ke, dual_ in model._model.dual.items()
        if "power_balance" in ke.name
    }

    # --- Save results ---
    print("...Saving results")
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
    dispatch.to_csv(
        f"data/results/dispatch_by_gen-{DISPATCH_DATE}-{config.dispatch_type.value}.csv",
        sep=",",
        index=False,
    )

    # --- Save prices ---

    price_model = pd.DataFrame(
        data=MPO.values(), index=MPO.keys(), columns=["ideal_marginal_price"]
    ).reset_index(drop=False, names=["datetime"])
    price_model.to_csv(
        f"data/results/marginal_price-{DISPATCH_DATE}-{config.dispatch_type.value}.csv",
        sep=",",
        index=False,
    )

    # print results
    print("...Saving images")
    fig, ax = plt.subplots(1, 1)

    if config.dispatch_type == "ideal":
        precio_bolsa.plot(
            kind="line", x="datetime", y="precio_bolsa", ax=ax, linestyle="-"
        )
    else:
        pd.DataFrame(data=mpo_xm, index=timestamps, columns=["MPO_XM"]).plot(
            kind="line", ax=ax, linestyle="-"
        )
    pd.DataFrame(data=MPO, index=[f"MPO-{config.dispatch_type.value}"]).T.plot(
        kind="line", ax=ax, linestyle="--"
    )
    fig.savefig(
        f"data/results/images/comparison_{config.dispatch_type.value}_{DISPATCH_DATE}.png"
    )
