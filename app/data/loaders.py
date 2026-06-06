"""Readers for the root-level XM CSVs.

Unit conversions that were previously scattered across the scripts are applied
here, in exactly one place (e.g. precio_bolsa is scaled to COP/MWh).
"""

import pandas as pd


def load_dispo(data_dir: str = "data") -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/dispo_declarada.csv", parse_dates=["datetime"])


def load_ofertas(data_dir: str = "data") -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/ofertas.csv", parse_dates=["Date"])


def load_demanda(data_dir: str = "data") -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/demaCome.csv", parse_dates=["datetime"])


def load_agc(data_dir: str = "data") -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/agc_asignado.csv", parse_dates=["datetime"])


def load_parametros_plantas(data_dir: str = "data") -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/parametros_plantas.csv")


def load_precio_bolsa(data_dir: str = "data") -> pd.DataFrame:
    df = pd.read_csv(
        f"{data_dir}/precio_bolsa/precio_bolsa_2024.csv", parse_dates=["datetime"]
    )
    df["precio_bolsa"] = df["precio_bolsa"] * 1e3
    return df


def load_dispo_come(data_dir: str = "data") -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/DispoCome_resource.csv", parse_dates=["datetime"])
