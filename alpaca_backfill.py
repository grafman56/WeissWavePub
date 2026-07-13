#!/usr/bin/env python3
"""Deep intraday backfill from Alpaca Market Data (SIP, adjusted).

Phases (run separately; each is resumable and idempotent):
    python alpaca_backfill.py 15m          # 15m bars since 2018-01-01
    python alpaca_backfill.py 5m           # 5m bars for the last year
    python alpaca_backfill.py derive-1h    # rebuild 1h from 15m bars
    python alpaca_backfill.py derive-4h    # build 4h from 15m bars

Why derive 1h instead of fetching it: Alpaca hourly bars are clock-aligned
(09:00, 10:00, ...) and the 09:00 bar includes premarket trades, while this
project's hourly convention (from Yahoo) is session-aligned (09:30, 10:30,
..., 15:30 half-bar) with regular-session volume only. Resampling our own
15m bars keeps one grid and clean Weis Wave volumes.

Resume logic: symbols whose earliest stored bar is already at/near the
requested start are skipped, so a run killed mid-way continues where it
left off. Requires exclusive DB access (stop the dashboard first).
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

from weisswave.db import (connect, first_timestamps, list_symbols,
                          load_prices, upsert_prices)
from weisswave.provider import AlpacaProvider

DEEP_START = datetime(2018, 1, 2)
CHUNK = 25              # symbols per request-chunk (persists per chunk)
WORKERS = 1             # 3 workers 429-stormed the 200 req/min free tier


def backfill(interval: str, start: datetime):
    prov = AlpacaProvider()
    con = connect()                       # read-write: needs exclusive access
    symbols = list_symbols(con, "1d")
    firsts = first_timestamps(con, interval)
    resume_edge = start + timedelta(days=14)
    todo = [s for s in symbols
            if firsts.get(s) is None or firsts[s] > resume_edge]
    print(f"[{interval}] {len(todo)}/{len(symbols)} symbols need backfill "
          f"from {start.date()}", flush=True)
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    total = 0
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(prov.history, c, start, interval): c
                   for c in chunks}
        for fut in as_completed(futures):   # main thread = the only writer
            done += len(futures[fut])
            try:
                df = fut.result()
            except Exception as e:
                print(f"  CHUNK FAILED ({futures[fut][0]}..): {e}",
                      flush=True)
                continue
            if df is not None and len(df):
                total += upsert_prices(con, df)
            print(f"  {done}/{len(todo)} symbols, {total} rows upserted",
                  flush=True)
    con.close()
    print(f"[{interval}] done - {total} rows", flush=True)


def derive_interval(out_interval: str, rule: str, offset: str,
                    start: datetime):
    """Rebuild a session-aligned derived interval from stored 15m bars.
    1h: 60min bins offset 30min  -> 09:30, 10:30, ..., 15:30 ET
    4h: 240min bins offset 90min -> 09:30 (morning) and 13:30 (afternoon)"""
    con = connect()
    symbols = list_symbols(con, "15m")
    print(f"[derive-{out_interval}] resampling 15m for {len(symbols)} "
          f"symbols", flush=True)
    total = 0
    for n, s in enumerate(symbols, 1):
        df = load_prices(con, s, "15m", start=start)
        if len(df) < 4:
            continue
        ny = df.tz_localize("UTC").tz_convert("America/New_York")
        agg = ny.resample(rule, offset=offset).agg(
            {"Open": "first", "High": "max", "Low": "min",
             "Close": "last", "Volume": "sum", "AdjClose": "last"}).dropna(
            subset=["Open", "Close"])
        agg.index = agg.index.tz_convert("UTC").tz_localize(None)
        out = agg.rename(columns=str.lower).reset_index() \
                 .rename(columns={"ts": "ts", "index": "ts"})
        out["symbol"] = s
        out["interval"] = out_interval
        total += upsert_prices(con, out)
        if n % 50 == 0:
            print(f"  {n}/{len(symbols)} symbols, {total} rows", flush=True)
    con.close()
    print(f"[derive-{out_interval}] done - {total} rows")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    if phase == "15m":
        backfill("15m", DEEP_START)
    elif phase == "5m":
        backfill("5m", datetime.now() - timedelta(days=365))
    elif phase == "derive-1h":
        derive_interval("1h", "60min", "30min", DEEP_START)
    elif phase == "derive-4h":
        derive_interval("4h", "240min", "90min", DEEP_START)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
