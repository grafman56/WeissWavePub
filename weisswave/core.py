"""Basic indicator building blocks matching Pine v4 semantics."""

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    """Pine ema(): alpha = 2/(length+1), recursive from the first value."""
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Pine rma() / Wilder's smoothing: alpha = 1/length."""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = rma(delta.clip(lower=0), length)
    loss = rma((-delta).clip(lower=0), length)
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def hlc3(df: pd.DataFrame) -> pd.Series:
    return (df["High"] + df["Low"] + df["Close"]) / 3


def rising(series: pd.Series, length: int) -> pd.Series:
    """Pine v4 rising(): current value greater than ANY of the previous
    `length` values."""
    return series > series.shift(1).rolling(length).max()


def falling(series: pd.Series, length: int) -> pd.Series:
    """Pine v4 falling(): current value lower than ANY of the previous
    `length` values."""
    return series < series.shift(1).rolling(length).min()


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """Pine crossover(a, b): a crosses above b on this bar."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """Pine crossunder(a, b): a crosses below b on this bar."""
    return (a < b) & (a.shift(1) >= b.shift(1))


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Accept yahooquery lowercase or already-capitalized OHLCV columns."""
    rename = {c: c.capitalize() for c in df.columns
              if c in ("open", "high", "low", "close", "volume")}
    df = df.rename(columns=rename)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {sorted(missing)}")
    return df
