"""Feature engineering for stock time series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_feature_frame(df: pd.DataFrame, target: str = "log_return") -> pd.DataFrame:
    """
    Build model-ready features from OHLCV data.

    Returns a DataFrame with target column and past-covariate columns.
    """
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["return_1d"] = out["close"].pct_change()
    out["ma5"] = out["close"].rolling(5).mean()
    out["ma20"] = out["close"].rolling(20).mean()
    out["ma60"] = out["close"].rolling(60).mean()
    out["volatility_20"] = out["log_return"].rolling(20).std()
    out["rsi_14"] = _rsi(out["close"], 14)
    out["volume_ma5"] = out["volume"].rolling(5).mean()
    out["volume_ratio"] = out["volume"] / out["volume_ma5"]
    out["hl_spread"] = (out["high"] - out["low"]) / out["close"]
    out["oc_spread"] = (out["close"] - out["open"]) / out["open"]

    if target not in out.columns:
        raise ValueError(f"Unknown target: {target}")

    feature_cols = [
        "volume",
        "ma5",
        "ma20",
        "ma60",
        "volatility_20",
        "rsi_14",
        "volume_ratio",
        "hl_spread",
        "oc_spread",
    ]

    out = out.dropna(subset=[target] + feature_cols)
    out = out.reset_index(drop=True)
    out.attrs["target_col"] = target
    out.attrs["past_cov_cols"] = feature_cols
    return out
