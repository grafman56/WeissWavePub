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
    python fetch_data.py                    # full S&P 500, incremental
    python fetch_data.py AAPL MSFT          # subset
    python fetch_data.py --full             # ignore stored data, refetch lookbacks
    python fetch_data.py --report           # show DB coverage and exit
    python fetch_data.py --intervals=1d,1h  # only these intervals
"""

import io
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from weisswave.db import (connect, coverage_report, first_timestamps,
                          last_timestamps, upsert_prices)
from weisswave.provider import get_provider

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHUNK_SIZE = 150          # symbols per batched provider request

# interval -> (full-history lookback, incremental overlap re-fetched each run)
# Daily: effectively "everything Yahoo has" for most large caps.
# Intraday lookbacks sit just inside Yahoo's serving limits (1h ~730d,
# 15m/5m ~60d, 1m ~7d). The fine intervals exist for fill verification:
# replaying coarse-timeframe trades against what actually happened intrabar.
INTERVALS = {
    "1d": (timedelta(days=365 * 25), timedelta(days=5)),
    "1h": (timedelta(days=365), timedelta(days=2)),
    "15m": (timedelta(days=55), timedelta(days=2)),
    "5m": (timedelta(days=55), timedelta(days=1)),
    "1m": (timedelta(days=6), timedelta(days=1)),
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


def run(symbols: list, full: bool = False, intervals=None):
    provider = get_provider()
    con = connect()
    t0 = time.time()
    for interval, (lookback, overlap) in INTERVALS.items():
        if intervals and interval not in intervals:
            continue
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

        def fetch_split(chunk, start, label):
            """Fetch a chunk; on provider error, bisect until the poison
            symbol(s) are isolated instead of losing the whole chunk."""
            nonlocal total
            try:
                rows = provider.history(chunk, start, interval)
            except Exception as e:
                if len(chunk) <= 4:
                    failed.extend(chunk)
                    print(f"  [{interval}] {label} {chunk} FAILED: {e}")
                    return
                mid = len(chunk) // 2
                fetch_split(chunk[:mid], start, label)
                fetch_split(chunk[mid:], start, label)
                return
            if rows is None:
                failed.extend(chunk)
                return
            n = upsert_prices(con, rows)
            failed.extend(sorted(set(chunk) - set(rows["symbol"])))
            total += n
            print(f"  [{interval}] {label}: {rows['symbol'].nunique()}"
                  f"/{len(chunk)} symbols, {n} rows upserted")

        for syms, start, label in batches:
            for i in range(0, len(syms), CHUNK_SIZE):
                fetch_split(syms[i:i + CHUNK_SIZE], start, label)
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
    intervals = None
    for a in args:
        if a.startswith("--intervals="):
            intervals = [x.strip() for x in a.split("=", 1)[1].split(",")]
    symbols = [a.upper() for a in args if not a.startswith("--")]
    if not symbols:
        print("Fetching S&P 500 constituent list...")
        symbols = get_sp500_tickers()
    print(f"{len(symbols)} symbols, mode={'full' if full else 'incremental'}"
          + (f", intervals={intervals}" if intervals else ""))
    run(symbols, full=full, intervals=intervals)


if __name__ == "__main__":
    main()
