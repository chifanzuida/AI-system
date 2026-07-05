"""Load real OHLCV market data from APIs or local CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and types."""
    rename_map = {
        "datetime": "date",
        "time": "date",
        "timestamp": "date",
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df.columns = [str(c).lower() for c in df.columns]

    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "date"})
        else:
            raise ValueError("DataFrame must contain a date column or DatetimeIndex.")

    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df = df.reset_index(drop=True)
    return df[list(REQUIRED_COLUMNS)]


def _load_yfinance(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"No data returned for symbol '{symbol}' from yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    return _normalize_ohlcv(data)


def _load_akshare(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError(
            "A-share data requires akshare. Install with: pip install akshare"
        ) from exc

    start_fmt = pd.Timestamp(start).strftime("%Y%m%d")
    end_fmt = pd.Timestamp(end or pd.Timestamp.today()).strftime("%Y%m%d")
    code = symbol.replace(".SH", "").replace(".SZ", "").replace("sh", "").replace("sz", "")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_fmt,
        end_date=end_fmt,
        adjust="qfq",
    )
    if df.empty:
        raise ValueError(f"No data returned for A-share symbol '{symbol}'.")

    return _normalize_ohlcv(df)


def _load_csv(path: str | Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_ohlcv(df)

    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]

    return df.reset_index(drop=True)


def load_ohlcv(
    symbol: str,
    market: str = "us",
    start: str = "2020-01-01",
    end: str | None = None,
    csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data.

    Parameters
    ----------
    symbol
        Ticker, e.g. ``AAPL`` (US) or ``600519`` (A-share).
    market
        ``us`` (yfinance), ``cn`` (akshare), or ``csv`` (local file).
    start, end
        Date range (``YYYY-MM-DD``).
    csv_path
        Required when ``market='csv'``. Expected columns:
        date, open, high, low, close, volume.
    """
    market = market.lower()
    if market == "us":
        df = _load_yfinance(symbol, start, end)
    elif market == "cn":
        df = _load_akshare(symbol, start, end)
    elif market == "csv":
        if not csv_path:
            raise ValueError("csv_path is required when market='csv'.")
        df = _load_csv(csv_path, start=start, end=end)
    else:
        raise ValueError(f"Unsupported market: {market}. Use us, cn, or csv.")

    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]

    if len(df) < 60:
        raise ValueError(
            f"Insufficient data ({len(df)} rows). Need at least 60 trading days."
        )
    return df.reset_index(drop=True)


def save_ohlcv_cache(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
