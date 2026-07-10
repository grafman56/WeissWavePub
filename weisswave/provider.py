"""Market data provider abstraction.

All fetching goes through a provider object with one method:

    history(symbols, start, interval) -> normalized DataFrame | None

The normalized schema (what the rest of the system depends on):
    columns: symbol, interval, ts, open, high, low, close, volume, adjclose
    ts: trading-date midnight for daily+ bars, naive UTC for intraday bars
    at most one row per (symbol, interval, ts)

To switch to Polygon, Alpaca, or any other source later: implement a class
with the same `history()` contract and return it from get_provider().
Nothing else in the codebase changes.
"""

from datetime import date, datetime

import pandas as pd


def normalize_history(raw: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Provider-agnostic timestamp/column normalization (see module doc).

    Timestamp rules (the crux of dedup for time-sensitive data):
      daily+   : ts = midnight of the trading date. Completed bars arrive as
                 dates; the live session bar arrives as a datetime in
                 exchange-local time — both map to the same trading date,
                 i.e. the same primary key.
      intraday : ts = bar start converted to UTC (tz-aware input), or
                 localized from America/New_York if the source returns naive
                 exchange-local times (fine for an all-US universe).
    """
    df = raw.reset_index()
    df = df.rename(columns={"date": "ts", "index": "ts"})
    daily = interval.endswith(("d", "wk", "mo"))

    def _norm(v):
        if isinstance(v, datetime):          # includes pd.Timestamp
            t = pd.Timestamp(v)
            if daily:
                return pd.Timestamp(t.date())          # date in its own tz
            if t.tzinfo is None:
                t = t.tz_localize("America/New_York")
            return t.tz_convert("UTC").tz_localize(None)
        if isinstance(v, date):
            return pd.Timestamp(v)
        return pd.NaT

    df["ts"] = df["ts"].map(_norm)
    df["interval"] = interval
    if "adjclose" not in df.columns:
        df["adjclose"] = df.get("close")
    keep = ["symbol", "interval", "ts", "open", "high", "low",
            "close", "volume", "adjclose"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["ts", "close"])
    # a batch can still contain the same key twice (rare source glitch):
    # keep the last occurrence rather than letting the upsert order decide
    return df.drop_duplicates(subset=["symbol", "interval", "ts"], keep="last")


class YahooProvider:
    """yahooquery-backed provider. Intraday history limits (Yahoo):
    1m ~7 days, 5m/15m/30m ~60 days, 1h ~730 days, 1d effectively full."""

    name = "yahoo"

    def history(self, symbols: list, start: datetime,
                interval: str) -> pd.DataFrame | None:
        from yahooquery import Ticker
        t = Ticker(symbols, asynchronous=True, max_workers=8)
        raw = t.history(start=start.strftime("%Y-%m-%d"), interval=interval)
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return None
        return normalize_history(raw, interval)


def get_provider() -> YahooProvider:
    """Single switch point for the data source."""
    return YahooProvider()
