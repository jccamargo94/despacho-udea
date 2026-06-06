"""Loaders for XM's actual predispatch results (the evaluation targets)."""

from datetime import date

import numpy as np
import pandas as pd


def load_actual_price(dispatch_date: date, data_dir: str = "data") -> np.ndarray:
    """XM marginal price (MPO) for the date as a 24-length float array."""
    df = pd.read_csv(f"{data_dir}/preideal_price/{dispatch_date}.txt", header=None)
    return df.iloc[0, 1:].astype(float).values


def load_actual_dispatch(dispatch_date: date, data_dir: str = "data") -> pd.DataFrame:
    """XM predispatch generation matrix for the date (raw, latin1-encoded)."""
    return pd.read_csv(
        f"{data_dir}/preideal_dispatch/{dispatch_date}.txt",
        header=None,
        encoding="latin1",
    )
