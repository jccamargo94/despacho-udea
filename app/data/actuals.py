"""Loaders for XM's actual predispatch results (the evaluation targets)."""

from datetime import date

import numpy as np
import pandas as pd

from app.storage import get_storage


def load_actual_price(dispatch_date: date, data_dir: str = "data") -> np.ndarray:
    """XM marginal price (MPO) for the date as a 24-length float array."""
    storage = get_storage(data_dir)
    with storage.open(f"preideal_price/{dispatch_date}.txt", "rb") as f:
        df = pd.read_csv(f, header=None)
    return df.iloc[0, 1:].astype(float).values


def load_actual_dispatch(dispatch_date: date, data_dir: str = "data") -> pd.DataFrame:
    """XM predispatch generation matrix for the date (raw, latin1-encoded)."""
    storage = get_storage(data_dir)
    with storage.open(f"preideal_dispatch/{dispatch_date}.txt", "rb") as f:
        return pd.read_csv(f, header=None, encoding="latin1")
