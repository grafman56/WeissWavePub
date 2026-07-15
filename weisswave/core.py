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
    """Pine v4 rising(): the series has been MONOTONICALLY rising for `length`
    bars -- source > source[1] AND source[1] > source[2] AND ... `length` times.

    NOT "greater than the max of the last `length` bars". That was the previous
    implementation and it is a BREAKOUT test, not a monotonicity test: a single
    strong up-bar after a dip passes the max test and fails Pine's.

    This sits under weis_wave's `isTrending`, which decides when the wave FLIPS,
    which decides whether volumeup/volumedn accumulate or reset -- so the wrong
    definition silently rewrote every volume-derived signal in the stack.
    Caught on BTC 2025-03-23: close 86,082 after 83,841 after 84,089. The max
    test says rising (86,082 > 84,089) and flips the wave; Pine says NOT rising
    (83,841 < 84,089 breaks the chain) and holds it. Paul's chart shows
    volumedn at 6.1x the bar volume there -- accumulating. Pine wins.
    """
    up = series.diff() > 0
    return up.rolling(length, min_periods=length).min().fillna(0).astype(bool)


def falling(series: pd.Series, length: int) -> pd.Series:
    """Pine v4 falling(): MONOTONICALLY falling for `length` bars. See rising()."""
    dn = series.diff() < 0
    return dn.rolling(length, min_periods=length).min().fillna(0).astype(bool)


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
