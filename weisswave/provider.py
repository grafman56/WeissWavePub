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

import os
import subprocess
import time
from datetime import date, datetime, timedelta, timezone

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


def _alpaca_creds() -> tuple:
    """Key id + secret from env, else from Vault via WSL vault-get.sh.
    Values live only in this process's memory — never logged or written."""
    key = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if key and sec:
        return key, sec

    def _vault(field):
        r = subprocess.run(
            ["wsl", "/usr/local/bin/vault-get.sh", "lab/alpaca", field],
            capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"vault-get.sh lab/alpaca {field} failed "
                               f"(is Vault unsealed?)")
        return r.stdout.strip()

    return _vault("paper_key_id"), _vault("paper_secret")


class AlpacaProvider:
    """Alpaca Market Data v2 (SIP feed, split/dividend adjusted).

    Free tier serves deep history — intraday bars back to ~2016 — but not
    the most recent 15 minutes; `history()` therefore ends requests 16
    minutes ago. Intraday bars are filtered to the regular session
    (09:30-16:00 ET) to match the Yahoo-era bar conventions."""

    name = "alpaca"
    BASE = "https://data.alpaca.markets/v2/stocks/bars"
    TIMEFRAMES = {"1m": "1Min", "5m": "5Min", "15m": "15Min",
                  "1h": "1Hour", "1d": "1Day"}
    CHUNK = 50                    # symbols per request
    THROTTLE = 0.31               # seconds between requests (<200/min)

    def __init__(self):
        import requests
        self._key, self._secret = _alpaca_creds()
        self._session = requests.Session()      # connection reuse
        self._session.headers.update({
            "APCA-API-KEY-ID": self._key,
            "APCA-API-SECRET-KEY": self._secret})

    def _get(self, params: dict) -> dict:
        for attempt in range(5):
            r = self._session.get(self.BASE, params=params, timeout=60)
            if r.status_code == 429:            # rate limited — back off
                wait = int(r.headers.get("Retry-After", 20 * (attempt + 1)))
                print(f"  [rate-limited, waiting {wait}s]", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Alpaca: still rate-limited after 5 attempts")

    def history(self, symbols: list, start: datetime,
                interval: str) -> pd.DataFrame | None:
        tf = self.TIMEFRAMES.get(interval)
        if tf is None:
            return None
        if isinstance(symbols, str):
            symbols = [symbols]
        end = datetime.now(timezone.utc) - timedelta(minutes=16)
        rows = []
        for i in range(0, len(symbols), self.CHUNK):
            chunk = symbols[i:i + self.CHUNK]
            # share classes: Yahoo/DB use BRK-B, Alpaca wants BRK.B
            to_alpaca = {s: s.replace("-", ".") for s in chunk}
            from_alpaca = {v: k for k, v in to_alpaca.items()}
            params = {
                "symbols": ",".join(to_alpaca.values()), "timeframe": tf,
                "start": pd.Timestamp(start, tz="UTC").isoformat(),
                "end": end.isoformat(), "limit": 10000,
                "adjustment": "all", "feed": "sip", "sort": "asc",
            }
            while True:
                data = self._get(params)
                for sym, bars in (data.get("bars") or {}).items():
                    sym = from_alpaca.get(sym, sym.replace(".", "-"))
                    for b in bars:
                        rows.append((sym, b["t"], b["o"], b["h"], b["l"],
                                     b["c"], b["v"]))
                token = data.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
                time.sleep(self.THROTTLE)
            time.sleep(self.THROTTLE)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high",
                                         "low", "close", "volume"])
        ts = pd.to_datetime(df["ts"], utc=True)
        if interval.endswith("d"):
            df["ts"] = ts.dt.tz_convert("America/New_York").dt.normalize() \
                         .dt.tz_localize(None)
        else:
            ny = ts.dt.tz_convert("America/New_York")
            session = ((ny.dt.hour * 60 + ny.dt.minute >= 570)
                       & (ny.dt.hour < 16))          # 09:30 <= t < 16:00 ET
            df = df[session.to_numpy()]
            ts = ts[session.to_numpy()]
            df["ts"] = ts.dt.tz_localize(None)       # naive UTC, DB convention
        df["interval"] = interval
        df["adjclose"] = df["close"]
        keep = ["symbol", "interval", "ts", "open", "high", "low",
                "close", "volume", "adjclose"]
        return df[keep].drop_duplicates(subset=["symbol", "interval", "ts"],
                                        keep="last")


def get_provider():
    """Single switch point for the data source. Yahoo remains the default
    for incremental fetching; set WEISSWAVE_PROVIDER=alpaca to switch."""
    if os.environ.get("WEISSWAVE_PROVIDER", "yahoo").lower() == "alpaca":
        return AlpacaProvider()
    return YahooProvider()
