"""Evaluation metrics for dispatch / marginal-price model validation.

Two families:

* Price metrics -- compare model marginal price (MPO) against the market
  operator's (XM) price. MAPE is intentionally avoided as the headline metric:
  the Colombian system is hydro-dominated, so MPO is frequently near zero and
  MAPE explodes / is asymmetric. Lead with RMSE, MAE (absolute units), bias
  and WAPE instead.

* Dispatch metrics -- compare generation results unit by unit: commitment
  agreement (on/off classification) and generation-by-technology error. These
  validate the *structure* of the dispatch, not only the resulting price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _align(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays and drop positions where either side is NaN."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


# --------------------------------------------------------------------------- #
# scalar price metrics
# --------------------------------------------------------------------------- #
def mae(y_true, y_pred) -> float:
    """Mean absolute error (same units as the price)."""
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (penalises large misses, e.g. price spikes)."""
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean (signed) error = mean(pred - true).

    Positive -> model systematically over-prices, negative -> under-prices.
    """
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error = sum|e| / sum|true|.

    Robust replacement for MAPE: a single small denominator can't blow it up
    because it normalises by the *total* actual energy/price.
    """
    y_true, y_pred = _align(y_true, y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_pred - y_true)) / denom)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in [0, 2]; safe when individual values are near zero."""
    y_true, y_pred = _align(y_true, y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2 * np.abs(y_pred - y_true)[mask] / denom[mask]))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination. Reported with a caveat: on autocorrelated
    price series R2 looks deceptively high; prefer RMSE/MAE/bias for ranking."""
    y_true, y_pred = _align(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def price_metrics(y_true, y_pred) -> dict[str, float]:
    """All price metrics at once. RMSE/MAE/bias are the ones to lead with."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def metrics_by_group(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    by,
) -> pd.DataFrame:
    """Price metrics computed per group (e.g. per day, or per hour-of-day).

    ``by`` is anything ``DataFrame.groupby`` accepts (column name, list, or a
    grouper such as ``df.index.hour``). One row of metrics per group.
    """
    return df.groupby(by).apply(lambda g: pd.Series(price_metrics(g[y_true_col], g[y_pred_col])))


# --------------------------------------------------------------------------- #
# dispatch / commitment metrics
# --------------------------------------------------------------------------- #
def commitment_metrics(
    actual_gen,
    model_gen,
    threshold: float = 1e-3,
) -> dict[str, float]:
    """On/off agreement between actual and model generation.

    A unit-period is "committed" when its generation exceeds ``threshold``.
    Returns accuracy + precision/recall/F1 of the model's committed set against
    the actual committed set (positive class = committed).
    """
    actual_gen, model_gen = _align(actual_gen, model_gen)
    actual_on = actual_gen > threshold
    model_on = model_gen > threshold

    tp = int(np.sum(actual_on & model_on))
    fp = int(np.sum(~actual_on & model_on))
    fn = int(np.sum(actual_on & ~model_on))
    tn = int(np.sum(~actual_on & ~model_on))

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and not (np.isnan(precision) or np.isnan(recall))
        else float("nan")
    )
    accuracy = (tp + tn) / total if total else float("nan")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def generation_by_tech(
    dispatch_df: pd.DataFrame,
    tech_map: dict[str, str],
    gen_col: str = "generador",
    value_col: str = "dispatch",
    default_tech: str = "other",
) -> pd.Series:
    """Aggregate generation by technology using a {generator: technology} map."""
    tech = dispatch_df[gen_col].map(tech_map).fillna(default_tech)
    return dispatch_df.groupby(tech)[value_col].sum()


def generation_mix_error(
    model_dispatch: pd.DataFrame,
    actual_dispatch: pd.DataFrame,
    tech_map: dict[str, str],
    gen_col: str = "generador",
    value_col: str = "dispatch",
) -> pd.DataFrame:
    """Generation-by-technology comparison: model vs actual, with abs error.

    Captures whether the model gets the energy *mix* (hydro vs thermal ...)
    right even when the marginal price differs.
    """
    model_mix = generation_by_tech(model_dispatch, tech_map, gen_col, value_col)
    actual_mix = generation_by_tech(actual_dispatch, tech_map, gen_col, value_col)
    out = pd.DataFrame({"model": model_mix, "actual": actual_mix}).fillna(0.0)
    out["abs_error"] = (out["model"] - out["actual"]).abs()
    out["pct_error"] = np.where(out["actual"] != 0, out["abs_error"] / out["actual"], np.nan)
    return out


def price_duration_curve(series) -> np.ndarray:
    """Values sorted descending -- plot model vs actual curves on the same axes
    to compare price/load structure independent of timing."""
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    return np.sort(arr)[::-1]
