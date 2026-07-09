#!/usr/bin/env python3
"""
Incremental S&P 500 OHLCV fetcher -> DuckDB (market.duckdb).

Duplicate-safe by construction:
  * the DB has PRIMARY KEY (symbol, interval, ts); every write is an
    INSERT OR REPLACE, so overlapping fetches overwrite instead of append
  * daily timestamps are normalized to the trading DATE, so the live
    in-progress bar Yahoo returns during market hours (which carries a
    full datetime) collapses onto the same key as the final bar fetched
    after the close — the fresh row replaces the stale partial one
  * every incremental run deliberately re-fetches a small overlap window
    for exactly that reason: the most recent bars are the least trustworthy

Usage:
    python fetch_data.py                 # full S&P 500, incremental
    python fetch_data.py AAPL MSFT       # subset
    python fetch_data.py --full          # ignore stored data, refetch lookbacks
    python fetch_data.py --report        # show DB coverage and exit
"""

import io
import sys
import time
from datetime import datetime, timedelta, date

import pandas as pd
import requests
from yahooquery import Ticker

from weisswave.db import (connect, coverage_report, first_timestamps,
                          last_timestamps, upsert_prices)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHUNK_SIZE = 150          # symbols per batched yahooquery request

# interval -> (full-history lookback, incremental overlap re-fetched each run)
# Daily: effectively "everything Yahoo has" for most large caps.
# Hourly: Yahoo only serves ~730 days of 1h bars; stay under that.
INTERVALS = {
    "1d": (timedelta(days=365 * 25), timedelta(days=5)),
    "1h": (timedelta(days=365), timedelta(days=2)),
}

# If a symbol's stored history starts this much later than the configured
# lookback, refetch its whole lookback (deep backfill) instead of just the
# incremental tail.
BACKFILL_SLACK = timedelta(days=45)


def get_sp500_tickers() -> list:
    """Current S&P 500 symbols from Wikipedia (browser UA to avoid 403s)."""
    resp = requests.get(SP500_URL, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"})
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text), header=0)[0]
    return sorted(df["Symbol"].str.replace(".", "-", regex=False).tolist())


def normalize_history(raw: pd.DataFrame, interval: str) -> pd.DataFrame:
    """yahooquery history -> rows ready for db.upsert_prices.

    Timestamp rules (the crux of dedup for time-sensitive data):
      daily+   : ts = midnight of the trading date. Completed bars arrive as
                 dates; the live session bar arrives as a datetime in
                 exchange-local time — both map to the same trading date,
                 i.e. the same primary key.
      intraday : ts = bar start converted to UTC (tz-aware input), or
                 localized from America/New_York if Yahoo returns naive
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
    # a chunk can still contain the same key twice (rare Yahoo glitch):
    # keep the last occurrence rather than letting the upsert order decide
    return df.drop_duplicates(subset=["symbol", "interval", "ts"], keep="last")


def fetch_chunk(symbols: list, start: datetime, interval: str) -> pd.DataFrame | None:
    """One batched yahooquery call; returns None when nothing came back."""
    t = Ticker(symbols, asynchronous=True, max_workers=8)
    raw = t.history(start=start.strftime("%Y-%m-%d"), interval=interval)
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return None
    return normalize_history(raw, interval)


def run(symbols: list, full: bool = False):
    con = connect()
    t0 = time.time()
    for interval, (lookback, overlap) in INTERVALS.items():
        now = datetime.now()
        default_start = now - lookback
        known = {} if full else last_timestamps(con, interval)
        firsts = {} if full else first_timestamps(con, interval)

        # Three groups: unknown symbols and symbols whose stored history is
        # much shallower than the configured lookback get the full lookback
        # (deep backfill); the rest only need the incremental tail since
        # their last stored bar (minus the safety overlap).
        backfill_syms = [s for s in symbols
                         if s not in known
                         or firsts.get(s, now) > default_start + BACKFILL_SLACK]
        old_syms = [s for s in symbols if s not in backfill_syms]
        batches = []
        if backfill_syms:
            batches.append((backfill_syms, default_start, "backfill"))
        if old_syms:
            oldest_last = min(known[s] for s in old_syms)
            start = max(min(oldest_last - overlap, now - overlap), default_start)
            batches.append((old_syms, start, "incremental"))

        total, failed = 0, []
        for syms, start, label in batches:
            for i in range(0, len(syms), CHUNK_SIZE):
                chunk = syms[i:i + CHUNK_SIZE]
                try:
                    rows = fetch_chunk(chunk, start, interval)
                except Exception as e:
                    failed.extend(chunk)
                    print(f"  [{interval}] chunk {i // CHUNK_SIZE + 1} FAILED: {e}")
                    continue
                if rows is None:
                    failed.extend(chunk)
                    continue
                n = upsert_prices(con, rows)
                got = rows["symbol"].nunique()
                failed.extend(sorted(set(chunk) - set(rows["symbol"])))
                total += n
                print(f"  [{interval}] {label} chunk {i // CHUNK_SIZE + 1}: "
                      f"{got}/{len(chunk)} symbols, {n} rows upserted")
        print(f"[{interval}] done - {total} rows upserted"
              + (f", no data for: {sorted(set(failed))}" if failed else ""))

    print(f"\nElapsed {time.time() - t0:.0f}s")
    print("\nDatabase coverage:")
    print(coverage_report(con).to_string(index=False))
    con.close()


def main():
    args = sys.argv[1:]
    if "--report" in args:
        con = connect(read_only=True)
        print(coverage_report(con).to_string(index=False))
        con.close()
        return
    full = "--full" in args
    symbols = [a.upper() for a in args if not a.startswith("--")]
    if not symbols:
        print("Fetching S&P 500 constituent list...")
        symbols = get_sp500_tickers()
    print(f"{len(symbols)} symbols, mode={'full' if full else 'incremental'}")
    run(symbols, full=full)


if __name__ == "__main__":
    main()
