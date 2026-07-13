#!/usr/bin/env python3
"""Firing-count check for the Pine barcolor combos on lower timeframes —
where they're designed to fire (daily is trend-only). Sample of symbols,
per interval. ASCII output.

    python combo_fire_check.py 1h
    python combo_fire_check.py 5m
"""

import sys

import pandas as pd

from weisswave.combined import BULL_COLUMNS, BEAR_COLUMNS
from weisswave.db import connect, list_symbols, load_prices
from weisswave.signals import build_signals

interval = sys.argv[1] if len(sys.argv) > 1 else "1h"
SAMPLE = 30

cbar = [c for c in BULL_COLUMNS + BEAR_COLUMNS if c.startswith("cbar")]
con = connect(read_only=True)
syms = list_symbols(con, interval)[:SAMPLE]
totals = {c: 0 for c in cbar}
nbars = 0
nsym = 0
for s in syms:
    df = load_prices(con, s, interval)
    if len(df) < 500:
        continue
    try:
        sig = build_signals(df)
    except Exception:
        continue
    nsym += 1
    nbars += len(sig)
    for c in cbar:
        totals[c] += int(sig[c].sum())
con.close()

# bars per year for this interval (24/7 crypto no; equities ~252 days)
bars_per_day = {"1d": 1, "4h": 2, "1h": 7, "15m": 26, "5m": 78}.get(interval, 7)
sym_years = nbars / (252 * bars_per_day) if nbars else 1
print(f"{nsym} symbols, {nbars} {interval} bars (~{sym_years:.0f} symbol-years)\n")
print(f"{'combo':34s} {'fires':>7s} {'/sym-yr':>8s}")
for c in sorted(cbar, key=lambda x: -totals[x]):
    print(f"{c:34s} {totals[c]:7d} {totals[c]/sym_years:8.1f}")
