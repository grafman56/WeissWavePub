#!/usr/bin/env python3
"""Discover intraday setups inside a stacked higher-TF uptrend. Higher
timeframes (daily + 4h) select tradable bars; the finder searches for what
has edge as an entry on the chosen low timeframe. Signals are pre-gated so
every discovered entry is in-trend by construction.

    python finder_gated.py 15m [n_symbols] [max_size]
    python finder_gated.py 15m --gate=rci_bull@1d
    python finder_gated.py 5m  --gate=minervini@1d,above_50ma@4h,in_up_wave@1h
    python finder_gated.py 15m --gate=none        (ungated, for comparison)

WHICH SCREEN WINS IS A TEST, NOT A CONSTANT. The screen is just a trend gauge:
higher timeframes pick the tradable stocks, the finder searches the low TF for
the entry. Which gauge is best is empirical. This was
`GATES = [("minervini","1d"), ("above_50ma","4h")]` hardcoded with no override,
so testing any other screen meant editing the file. The default below keeps the
old behaviour; --gate makes it a question.

Note: finder output is GROSS (no cost). Cost-test survivors with
test_strategy.py --cost-bps=10. ASCII output."""

import sys
import time

import pandas as pd

from test_strategy import apply_gates, arg, load_universe, parse_gates
from weisswave.optimize import find_strategies
from weisswave.signals import SIGNAL_COLUMNS_BULL

_args = sys.argv[1:]
_pos = [a for a in _args if not a.startswith("--")]   # flags are not positional
INTERVAL = _pos[0] if _pos else "15m"
N_SYMBOLS = int(_pos[1]) if len(_pos) > 1 else 200
MAX_SIZE = int(_pos[2]) if len(_pos) > 2 else 2
GATES = parse_gates(arg(_args, "gate", "minervini@1d,above_50ma@4h"))

# per-interval horizons (bars) and holds (bars): ~half-day / 1-day, 1-day / 3-day
PARAMS = {"5m":  {"horizons": (39, 78),  "holds": (39, 78)},
          "15m": {"horizons": (13, 26),  "holds": (26, 78)},
          "1h":  {"horizons": (7, 21),   "holds": (20, 33)},
          "4h":  {"horizons": (6, 12),   "holds": (10, 20)}}
p = PARAMS.get(INTERVAL, PARAMS["15m"])

t0 = time.time()
frames = load_universe(INTERVAL, None)
frames = dict(list(frames.items())[:N_SYMBOLS])
gated = apply_gates(frames, GATES, None)
print(f"{len(gated)} symbols, {INTERVAL}, gated {GATES}, max_size={MAX_SIZE}, "
      f"{time.time()-t0:.0f}s", flush=True)

for s, sig in gated.items():
    g = sig["xtf_gate"].to_numpy(bool)
    for c in SIGNAL_COLUMNS_BULL:
        if c in sig.columns:
            sig[c] = sig[c].to_numpy(bool) & g


def cb(frac, text):
    print(f"  stage2 {frac:.0%} {text}", flush=True)


s1, s2 = find_strategies(gated, entry_signals=SIGNAL_COLUMNS_BULL,
                         filter_cols=[], window=3, horizons=p["horizons"],
                         train_frac=0.7, max_size=MAX_SIZE, min_events=150,
                         top_k=12, exit_names=("none", "wt_cross_down"),
                         stops=(0.03, 0.05), holds=p["holds"], progress_cb=cb)

pd.set_option("display.width", 240)
print(f"\nTotal {time.time()-t0:.0f}s")
print("\n== STAGE 2 sorted by TEST excess (out-of-sample honesty) ==")
cols = [c for c in ["entry", "exit", "max_hold", "stop_value", "train_n",
                    "train_xs", "test_n", "test_xs", "test_pf"] if c in s2.columns]
top = s2.sort_values("test_xs", ascending=False)[cols].head(20)
print(top.to_string())
