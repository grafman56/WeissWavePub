#!/usr/bin/env python3
"""Discover 5m setups INSIDE a big-timeframe uptrend. The higher timeframes
(daily + 4h) select which bars are tradable; the finder then searches for
what actually has edge as a 5m entry on those bars. Signals are pre-gated
(ANDed with the trend) so every discovered entry is in-trend by construction.

    python finder_5m_gated.py            # default 150 symbols
    python finder_5m_gated.py 60 --gate=rci_bull@1d
    python finder_5m_gated.py --gate=none

WHICH SCREEN WINS IS A TEST, NOT A CONSTANT. The gates were hardcoded here with
no override, so trying any other screen meant editing the file. The default
keeps the old behaviour. See finder_gated.py, which does the same for any
interval.

ASCII output."""

import sys
import time

import pandas as pd

from test_strategy import apply_gates, arg, load_universe, parse_gates
from weisswave.optimize import find_strategies
from weisswave.signals import SIGNAL_COLUMNS_BULL

INTERVAL = "5m"
_args = sys.argv[1:]
_pos = [a for a in _args if not a.startswith("--")]   # flags are not positional
GATES = parse_gates(arg(_args, "gate", "minervini@1d,above_50ma@4h"))
N_SYMBOLS = int(_pos[0]) if _pos else 150

t0 = time.time()
frames = load_universe(INTERVAL, None)
frames = dict(list(frames.items())[:N_SYMBOLS])
gated = apply_gates(frames, GATES, None)
print(f"{len(gated)} symbols gated by {GATES}, {time.time()-t0:.0f}s",
      flush=True)

# bake the trend gate into every entry signal so the finder can't un-gate it
for s, sig in gated.items():
    g = sig["xtf_gate"].to_numpy(bool)
    for c in SIGNAL_COLUMNS_BULL:
        if c in sig.columns:
            sig[c] = sig[c].to_numpy(bool) & g


def cb(frac, text):
    print(f"  stage2 {frac:.0%} {text}", flush=True)


# 5m-appropriate: short confluence window, ~1hr/half-day horizons,
# half-day / full-day holds, tighter stops
s1, s2 = find_strategies(gated, entry_signals=SIGNAL_COLUMNS_BULL,
                         filter_cols=[], window=3, horizons=(12, 39),
                         train_frac=0.7, max_size=2, min_events=150,
                         top_k=10, exit_names=("none", "wt_cross_down"),
                         stops=(0.03, 0.05), holds=(39, 78),
                         progress_cb=cb)

pd.set_option("display.width", 240)
print(f"\nTotal {time.time()-t0:.0f}s")
print("\n== STAGE 1: top 20 gated-5m combos by train excess ==")
print(s1.head(20).to_string())
print("\n== STAGE 2: full sim, train vs test (honesty check) ==")
cols = [c for c in ["entry", "exit", "max_hold", "train_n", "train_avg",
                    "train_xs", "test_n", "test_avg", "test_xs", "test_pf"]
        if c in s2.columns]
print(s2[cols].to_string())
