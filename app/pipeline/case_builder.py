"""Build the pyomo set/param dictionaries for one (date, dispatch type).

This is the single source of truth for the data-to-model translation that was
previously duplicated across run_dispatch.py and get_date_results.py. It loads
the XM inputs, parses OFEI, synthesizes combined-cycle resources, resolves
initial conditions, fuzzy-maps names, and assembles `set_data` / `param_data`.

It does NOT build or solve the model; callers do that with the returned dicts.
`meta` carries the non-model artifacts the legacy plotting/return path needs.
"""

from copy import deepcopy
from datetime import date
import json
import re

import numpy as np
import pandas as pd
from thefuzz import process, fuzz

try:
    from app.model import DispatchConfig, DispatchOptions
except ImportError:  # pragma: no cover - app.model dropped these in Task 5;
    # Task 7 migrates build_case's signature/body to app.schemas.case (DispatchCase).
    # Stubbed here only so this module (and the new bess_scenario_to_params below)
    # can still be imported/unit-tested in the meantime.
    DispatchConfig = DispatchOptions = None
from app.data.download import ensure_data_for_date
from app.data import loaders
from app.data.ofei import parse_ofei
from app.data.paths import resolve_input
from app.schemas.bess import BessScenario


def bess_scenario_to_params(scenario: BessScenario) -> tuple[list[str], dict]:
    """Map a BessScenario's units to the pyomo-level set/param dicts consumed
    by UnitCommitmentModel._add_bess_operation. Mirrors the historical bess
    dict shape 1:1 (initial_soc/min_soc/max_soc are fractions of mwh_nom;
    max_charge/max_discharge = mwh_nom / hours_to_deplete)."""
    names = [u.name for u in scenario.units]
    params: dict[str, dict] = {
        "bess_soc_0": {}, "bess_charge_bid": {}, "bess_discharge_bid": {},
        "bess_min_soc": {}, "bess_max_soc": {}, "efficiency": {},
        "bess_max_charge": {}, "bess_max_discharge": {},
    }
    for u in scenario.units:
        params["bess_soc_0"][u.name] = u.initial_soc * u.mwh_nom
        params["bess_min_soc"][u.name] = u.min_soc * u.mwh_nom
        params["bess_max_soc"][u.name] = u.max_soc * u.mwh_nom
        params["efficiency"][u.name] = u.efficiency
        params["bess_max_charge"][u.name] = u.mwh_nom / u.hours_to_deplete
        params["bess_max_discharge"][u.name] = u.mwh_nom / u.hours_to_deplete
        if u.charge_bid is not None:
            params["bess_charge_bid"][u.name] = u.charge_bid
        if u.discharge_bid is not None:
            params["bess_discharge_bid"][u.name] = u.discharge_bid
    return names, params


def build_case(
    dispatch_date: date,
    config: DispatchConfig,
    *,
    bess: dict | None = None,
    ders: int | None = None,
    data_dir: str = "data",
) -> tuple[dict, dict, dict]:
    """Return (set_data, param_data, meta) for `UnitCommitmentModel`.

    meta keys: timestamps, precio_bolsa, CC, initial_condition_df,
    major_generators, generators, fixed_fuel_fire, pmax_new_resources,
    expansion_sources.
    """
    DISPATCH_DATE = dispatch_date
    BESS = bess
    DERS = ders
    dd = data_dir

    ensure_data_for_date(DISPATCH_DATE, data_dir=dd)

    # --- Load root CSVs ---
    if config.dispatch_type == "ideal":
        dispo_come = loaders.load_dispo_come(dd)
    dispo = loaders.load_dispo(dd)
    ofertas = loaders.load_ofertas(dd)
    demanda = loaders.load_demanda(dd)
    agc_asignado = loaders.load_agc(dd)
    parametros_plantas = loaders.load_parametros_plantas(dd)
    precio_bolsa = loaders.load_precio_bolsa(dd)

    # --- Parse OFEI ---
    ofei_path = resolve_input("OFEI", DISPATCH_DATE, dd)
    ofei = parse_ofei(ofei_path, DISPATCH_DATE)
    precio_arranque = ofei.precio_arranque
    minimo_operativo = ofei.minimo_operativo
    CC = ofei.cc
    cc_price = ofei.cc_price
    cc_dispo = ofei.cc_dispo
    prices = ofei.prices

    # --- Filter data by date ---
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
                dispo.loc[dispo["resource_name"] == gen, "dispo"] = serie["dispo"].values
            else:
                print(
                    f"no existe el generador {gen} en disponibilidad comercial para el {DISPATCH_DATE}. Se asignará en 0"
                )
                dispo.loc[dispo["resource_name"] == gen, "dispo"] = 0

    # --- Map names / extract prices from OFEI ---
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

    ofertas["Value"] = ofertas.apply(
        lambda x: prices.get(x["resource_name"], float(x["Value"])), axis=1
    )

    # --- Initial conditions ---
    with open(resolve_input("dCondIniP", DISPATCH_DATE, dd), "r") as file:
        data = file.readlines()
        data = [line.strip().split(",") for line in data]
        headers = data.pop(0)
    condicion_inicial_planta = pd.DataFrame(data, columns=headers)

    with open(resolve_input("dCondIniU", DISPATCH_DATE, dd), "r") as file:
        data = file.readlines()
        data = [line.strip().split(",") for line in data]
        headers = data.pop(0)
    condicion_inicial_unidad = pd.DataFrame(data, columns=headers)

    condicion_inicial_map = {
        gen: process.extractOne(
            query=gen.lower(),
            choices=dispo["resource_name"].unique(),
            scorer=fuzz.token_sort_ratio,
            processor=lambda x: x.lower().replace(" ", ""),
        )[0]
        for gen in condicion_inicial_planta.Recurso.unique()
    }
    condicion_inicial_map |= {
        "FLORES IV": "FLORES 4 CC",
        "TSIERRA": "TERMOSIERRA CC",
        "GUAJIR21": "GUAJIRA 2",
    }
    condicion_inicial_planta["Recurso"] = condicion_inicial_planta["Recurso"].apply(
        lambda x: condicion_inicial_map.get(x, x)
    )

    # --- New CC resources ---
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

    new_cc_bid = pd.DataFrame(cc_price, index=[1]).stack().reset_index(drop=False)
    new_cc_bid.columns = ["index_", "resource_name", "Value"]
    new_cc_bid["Value"] = new_cc_bid["Value"].apply(lambda x: x * 1e-3)
    new_cc_bid["resource_gen_type"] = "TERMICA"
    new_cc_bid["Date"] = DISPATCH_DATE
    _ = new_cc_bid.pop("index_")

    dispo = pd.concat([dispo, new_cc_resources], axis=0)
    ofertas = pd.concat([ofertas, new_cc_bid], axis=0)

    # --- Adding units for each CC resource ---
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

    # --- Initial set to model ---
    gen_on = initial_condition_df[initial_condition_df["Gpini-1"] != 0][
        "Recurso"
    ].unique()
    needed_generators = [
        gen for gen in list(gen_on) if gen not in ofertas.resource_name.unique()
    ]
    for gen in needed_generators:
        gen_oferta = (
            oferta_full.query("resource_name == @gen").head(1).reset_index(drop=True)
        )
        gen_oferta.loc[0, "Date"] = pd.Timestamp(DISPATCH_DATE)
        ofertas = pd.concat([ofertas, gen_oferta], axis=0)

    major_generators = ofertas.resource_name.unique()
    generators = dispo.resource_name.unique()
    timestamps = demanda["datetime"].to_dict().values()
    fuel_generators = dispo[
        (dispo["resource_name"].isin(major_generators))
        & (dispo["gen_type"] == "TERMICA")
    ].resource_name.unique()

    gen_off = list(set(fuel_generators) - set(gen_on))

    # --- Startup/shutdown costs ---
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

    prid_path = resolve_input("PrId", DISPATCH_DATE, dd)
    demand_pronos = pd.read_csv(prid_path, header=None, encoding="latin1")
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

    # --- Fix fuel-fired generators ---
    fixed_fuel_fire = pd.read_csv(prid_path, header=None, encoding="latin1")
    fixed_fuel_fire.columns = ["generator"] + list(range(24))
    fixed_fuel_fire = fixed_fuel_fire.set_index("generator").stack().reset_index()
    fixed_fuel_fire.columns = ["generator", "hour", "gen"]
    fixed_fuel_fire["datetime"] = pd.to_datetime(DISPATCH_DATE) + pd.to_timedelta(
        fixed_fuel_fire["hour"], unit="h"
    )

    fixed_fuel_fired_map = {}
    for gen in fixed_fuel_fire.generator.unique():
        choice = process.extractOne(
            query=gen.lower(),
            choices=generators,
            scorer=fuzz.partial_ratio,
            processor=lambda x: x.lower().replace(" ", ""),
        )
        if choice and choice[0] in generators:
            fixed_fuel_fired_map[gen] = choice[0]
        else:
            ...

    fixed_fuel_fire_2 = fixed_fuel_fire.copy()
    with open(f"{dd}/preideal_dispatch_map.json", "r", encoding="utf-8") as file:
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
        Pmax_model.update(
            fixed_fuel_fire_2[
                fixed_fuel_fire_2.index.get_level_values(0).isin(generators)
            ].to_dict()
        )

    # --- RAMPS ---
    with open(f"{dd}/ramps.json", "r") as file:
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

    # --- DERs ---
    pmax_new_resources = pd.DataFrame()
    expansion_sources = list()
    if DERS:
        DERS = str(DERS)
        new_resources_df = pd.read_excel(
            f"{dd}/Supuestos Modelo de despacho.xlsx", sheet_name="series"
        )
        expansion_sources = [col for col in new_resources_df.columns if DERS in col]
        pmax_new_resources = new_resources_df[expansion_sources]
        pmax_new_resources.index = pd.Index(
            pd.to_datetime(DISPATCH_DATE)
            + pd.to_timedelta(new_resources_df.hours, unit="h"),
            name="datetime",
        )
        pmax_new_resources = pmax_new_resources.stack().reset_index()
        pmax_new_resources.columns = ["datetime", "resource_name", "dispo"]
        Pmax_model.update(
            pmax_new_resources.set_index(["resource_name", "datetime"]).to_dict()[
                "dispo"
            ]
        )
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
        bess_params_model = dict(zip(BESS_PARAMS_NAMES, [{} for _ in BESS_PARAMS_NAMES]))
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

    meta = {
        "timestamps": timestamps,
        "precio_bolsa": precio_bolsa,
        "demanda": demanda,
        "CC": CC,
        "initial_condition_df": initial_condition_df,
        "major_generators": major_generators,
        "generators": generators,
        "fixed_fuel_fire": fixed_fuel_fire,
        "pmax_new_resources": pmax_new_resources,
        "expansion_sources": expansion_sources,
    }

    return set_data, param_data, meta
