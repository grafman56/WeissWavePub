#!/usr/bin/env python3
"""Nightly setup scanner: evaluate saved strategy configs on the latest bar.

Reads strategy definitions from strategies.json (gitignored — see
strategies.example.json for the format), builds signals for every symbol
in the DB, and reports which symbols fired each strategy's entry rule on
the most recent bar(s). Designed to run post-close from Task Scheduler /
cron; writes a CSV report per run under scans/.

Usage:
    python scan_today.py                    # scan latest bar, 1d interval
    python scan_today.py --fetch            # incremental 1d fetch first
    python scan_today.py --lookback=3       # also report fires 1-2 bars ago
    python scan_today.py --config=my.json   # alternate strategy file

Exit code: 0 on any successful scan (hits are reported via stdout and the
CSV), 99 on config/data errors.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd

from weisswave.db import DB_PATH, connect, list_symbols, load_prices
from weisswave.signals import build_signals, combine_signals

SCAN_DIR = "scans"
STALE_DAYS = 5


def load_strategies(path: str) -> list:
    if not os.path.exists(path):
        print(f"No {path} — copy strategies.example.json to {path} and "
              f"define your strategies.")
        sys.exit(99)
    with open(path, encoding="utf-8") as f:
        strategies = json.load(f)
    for s in strategies:
        s.setdefault("min_count", 1)
        s.setdefault("window", 5)
        s.setdefault("filter", None)
        s.setdefault("stop_pct", 0.08)
    return strategies


def main():
    args = sys.argv[1:]
    lookback = int(next((a.split("=", 1)[1] for a in args
                         if a.startswith("--lookback=")), "1"))
    cfg_path = next((a.split("=", 1)[1] for a in args
                     if a.startswith("--config=")), "strategies.json")
    interval = next((a.split("=", 1)[1] for a in args
                     if a.startswith("--interval=")), "1d")

    strategies = load_strategies(cfg_path)

    if "--fetch" in args:
        print("Incremental fetch (1d)...")
        r = subprocess.run([sys.executable, "fetch_data.py",
                            f"--intervals={interval}"])
        if r.returncode:
            print("fetch_data.py failed; scanning existing data.")

    con = connect(read_only=True)
    symbols = list_symbols(con, interval)
    if not symbols:
        print(f"No {interval} data in {DB_PATH}.")
        sys.exit(99)

    hits = []
    latest_seen = None
    n_scanned = 0
    for sym in symbols:
        df = load_prices(con, sym, interval)
        if len(df) < 260:
            continue
        try:
            sig = build_signals(df)
        except Exception:
            continue
        n_scanned += 1
        last_ts = sig.index[-1]
        if latest_seen is None or last_ts > latest_seen:
            latest_seen = last_ts
        for strat in strategies:
            entry = combine_signals(sig, strat["entry_cols"],
                                    strat["min_count"], strat["window"])
            if strat["filter"] and strat["filter"] in sig.columns:
                entry = entry & sig[strat["filter"]].fillna(False)
            tail = entry.iloc[-lookback:]
            if not tail.any():
                continue
            bars_ago = len(tail) - 1 - int(tail.to_numpy().nonzero()[0][-1])
            close = float(sig["Close"].iloc[-1])
            upto = len(sig) - bars_ago      # components as of the signal bar
            comp = {c: _bars_since(sig[c].iloc[:upto], strat["window"])
                    for c in strat["entry_cols"] if c in sig.columns}
            hits.append({
                "strategy": strat["name"], "symbol": sym,
                "signal_date": str(sig.index[-1 - bars_ago].date()),
                "bars_ago": bars_ago, "close": round(close, 2),
                "stop": round(close * (1 - strat["stop_pct"]), 2),
                "components": "; ".join(f"{k}@-{v}" if v is not None else
                                        f"{k}@none" for k, v in comp.items()),
            })
    con.close()

    if latest_seen is None:
        print("No usable symbols.")
        sys.exit(99)
    age = datetime.now() - latest_seen.to_pydatetime()
    print(f"Scanned {n_scanned} symbols ({interval}); latest bar "
          f"{latest_seen.date()}.")
    if age > timedelta(days=STALE_DAYS):
        print(f"WARNING: data is {age.days} days old — run with --fetch or "
              f"check fetch_data.py.")

    if not hits:
        print("No setups today.")
        sys.exit(0)

    out = pd.DataFrame(hits).sort_values(["strategy", "bars_ago", "symbol"])
    print(f"\n{len(out)} setup(s):\n")
    print(out.to_string(index=False))

    os.makedirs(SCAN_DIR, exist_ok=True)
    path = os.path.join(SCAN_DIR,
                        f"scan_{latest_seen.date()}_{interval}.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")


def _bars_since(col: pd.Series, window: int):
    """Bars since this component last fired, if within window (else None)."""
    tail = col.iloc[-(window + 1):].to_numpy()
    idx = tail.nonzero()[0]
    return int(len(tail) - 1 - idx[-1]) if len(idx) else None


if __name__ == "__main__":
    main()
