#!/usr/bin/env python3
"""Backfill crypto OHLCV from Binance's public data archive
(data.binance.vision) into the same market.duckdb `prices` table as the stocks.

No account, no API key. Binance's LIVE api is geo-blocked from the US (HTTP
451), but the archive CDN is not -- so we pull the pre-packaged monthly kline
ZIPs (full history, real volume) rather than hitting the API. Crypto and stocks
coexist in one table; crypto symbols are stored as e.g. BTC-USD, so
load_universe just sees more symbols.

    python binance_fetch.py                         # defaults: 15m+1d, from 2020
    python binance_fetch.py --pairs=BTC,ETH,SOL --intervals=15m --from=2021

ASCII output. Months that don't exist yet (before a coin listed, or the current
incomplete month) are skipped."""

import io
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from test_strategy import arg
from weisswave.db import connect, upsert_prices

BASE = "https://data.binance.vision/data/spot/monthly/klines"
# BTC/ETH back to 2017, SOL/high-beta alts listed 2020-2023 (early months just
# 404 and are skipped).
DEFAULT_PAIRS = ["BTC", "ETH", "SOL", "AVAX", "LINK", "DOGE", "ADA", "MATIC",
                 "DOT", "ATOM", "NEAR", "APT", "INJ"]
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "qv", "trades", "tbb", "tbq", "ignore"]


def months(start_ym):
    now = pd.Timestamp.now()
    m = pd.Timestamp(start_ym + "-01")
    out = []
    while m < pd.Timestamp(now.year, now.month, 1):   # skip incomplete month
        out.append(m.strftime("%Y-%m"))
        m += pd.DateOffset(months=1)
    return out


def fetch_month(pair, interval, ym):
    """Download one monthly kline ZIP -> DataFrame in PRICE_COLS, or None."""
    url = f"{BASE}/{pair}USDT/{interval}/{pair}USDT-{interval}-{ym}.zip"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None                       # month not published (pre-listing/now)
    z = zipfile.ZipFile(io.BytesIO(data))
    raw = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None,
                      names=KLINE_COLS)
    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
    raw = raw.dropna(subset=["open_time"])               # drop any header row
    if raw.empty:
        return None
    ot = raw["open_time"].astype("int64")
    # Binance switched kline times from ms to MICROseconds in 2025 archives;
    # detect per-file by magnitude (ms ~1.7e12, us ~1.7e15 for these years).
    unit = "us" if ot.iloc[0] >= 10 ** 14 else "ms"
    out = pd.DataFrame({
        "symbol": f"{pair}-USD", "interval": interval,
        "ts": pd.to_datetime(ot, unit=unit),
        "open": raw["open"], "high": raw["high"], "low": raw["low"],
        "close": raw["close"], "volume": raw["volume"]})
    out["adjclose"] = out["close"]                       # no splits/divs in crypto
    return out


def main():
    args = sys.argv[1:]
    p = arg(args, "pairs", None)
    pairs = p.split(",") if p else DEFAULT_PAIRS
    intervals = arg(args, "intervals", "15m,1d").split(",")
    start = arg(args, "from", "2020")
    start = f"{start}-01" if len(start) == 4 else start
    jobs = [(pr, iv, ym) for pr in pairs for iv in intervals
            for ym in months(start)]
    print(f"binance archive: {len(pairs)} pairs x {len(intervals)} intervals "
          f"x months from {start} -> {len(jobs)} files to try", flush=True)
    con = connect()
    total = done = got = 0
    with ThreadPoolExecutor(max_workers=8) as pool:            # parallel download
        futs = {pool.submit(fetch_month, *j): j for j in jobs}
        for f in as_completed(futs):                           # upsert in main thd
            df = f.result()
            done += 1
            if df is not None and not df.empty:
                total += upsert_prices(con, df)
                got += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} tried, {got} files had data, "
                      f"{total} rows", flush=True)
    con.close()
    print(f"done: {total} rows from {got} files across {len(pairs)} pairs "
          f"{[f'{p}-USD' for p in pairs]}", flush=True)


if __name__ == "__main__":
    main()
