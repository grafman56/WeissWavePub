#!/usr/bin/env python3
"""Edge test for the Pine barcolor combos on lower timeframes. Pooled
event study: forward returns after each combo vs the all-bars baseline.
Optional daily-trend gate (--gate=sma50_over_200) restricts to bars inside a
prior-session daily uptrend. ASCII output.

    python combo_event_study.py 1h
    python combo_event_study.py 1h --gate=sma50_over_200
"""

import sys

import numpy as np
import pandas as pd

from weisswave.combined import BULL_COLUMNS, BEAR_COLUMNS
from weisswave.db import connect, list_symbols, load_prices
from weisswave.signals import build_signals
from weisswave.study import forward_returns

interval = next((a for a in sys.argv[1:] if not a.startswith("-")), "1h")
gate_col = next((a.split("=", 1)[1] for a in sys.argv[1:]
                 if a.startswith("--gate=")), None)
HORIZONS = (7, 21, 33)          # 1h: ~1 day, 3 days, ~1 week
combos = [c for c in BULL_COLUMNS + BEAR_COLUMNS if c.startswith("cbar")]

con = connect(read_only=True)
syms = list_symbols(con, interval)
daily_gate = {}
if gate_col:
    for s in list_symbols(con, "1d"):
        d = load_prices(con, s, "1d")
        if len(d) < 260:
            continue
        try:
            ds = build_signals(d)
        except Exception:
            continue
        g = ds[gate_col].astype(bool).shift(1).fillna(False)   # no lookahead
        g.index = pd.DatetimeIndex(g.index).normalize()
        daily_gate[s] = g[~g.index.duplicated(keep="last")]

frames = []
for s in syms:
    df = load_prices(con, s, interval)
    if len(df) < 500:
        continue
    try:
        sig = build_signals(df)
    except Exception:
        continue
    fwd = forward_returns(sig, HORIZONS)
    keep = sig[[c for c in combos if c in sig.columns]].astype(bool).join(fwd)
    if gate_col:
        g = daily_gate.get(s)
        if g is None:
            continue
        mask = g.reindex(sig.index.normalize()).fillna(False).to_numpy()
        keep = keep[mask]
    frames.append(keep)
con.close()

pooled = pd.concat(frames)
base = {h: pooled[f"fwd_{h}"].mean() for h in HORIZONS}
gtxt = f" gated by {gate_col}@1d" if gate_col else ""
print(f"{len(frames)} symbols, {len(pooled)} {interval} bars{gtxt}, "
      f"baseline fwd33 = {base[33]*100:.3f}%\n")

rows = []
for c in combos:
    if c not in pooled.columns:
        continue
    r = pooled.loc[pooled[c], "fwd_33"].dropna()
    if len(r) < 30:
        continue
    sd = r.std()
    rows.append({"combo": c, "n": len(r),
                 "edge33": round((r.mean() - base[33]) * 100, 3),
                 "win33": round((r > 0).mean() * 100, 1),
                 "t33": round((r.mean() - base[33]) / (sd / np.sqrt(len(r)))
                              if sd > 0 else np.nan, 2)})
out = pd.DataFrame(rows).sort_values("edge33", ascending=False)
print(out.to_string(index=False))
