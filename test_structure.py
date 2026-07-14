#!/usr/bin/env python3
"""Trust tests for weisswave.structure (Fibonacci / swing-structure levels).

The load-bearing property is NO LOOKAHEAD: the stop/target level at bar t must
depend only on bars <= t. We prove it directly — recompute on the prefix
data[:t+1] and require the last value to equal the full-series value at t.
Plus a hand-built series with a known pivot low and pivot high pins the fib
math. Run: python test_structure.py (exit 0 = all pass)."""

import sys

import numpy as np
import pandas as pd

from weisswave.structure import (confirmed_pivots, structure_levels,
                                 trend_points)

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# 1. known pivots -> known fib levels -----------------------------------------
# 5-bar fractal (lbL=lbR=2). One pivot low (8 @ idx2, confirmed t=4) and one
# pivot high (20 @ idx6, confirmed t=8).
low = pd.Series([12, 11, 8, 11, 12, 13, 14, 15, 16, 15, 14, 13, 12], dtype=float)
high = pd.Series([12, 13, 14, 15, 16, 18, 20, 18, 16, 15, 14, 13, 12], dtype=float)
ph, pl = confirmed_pivots(high, low, 2, 2)

check("pivot low confirmed at t=4 with value 8",
      approx(pl.iloc[4], 8.0) and pd.isna(pl.iloc[3]),
      f"pl[3]={pl.iloc[3]} pl[4]={pl.iloc[4]}")
check("pivot high confirmed at t=8 with value 20",
      approx(ph.iloc[8], 20.0) and pd.isna(ph.iloc[7]),
      f"ph[7]={ph.iloc[7]} ph[8]={ph.iloc[8]}")

stop, target, pvl = structure_levels(high, low, 2, 2, stop_ratio=0.786,
                                     buf=0.005)
# H=20, L=8, span=12 -> level = 20 - 0.786*12 = 10.568; *(1-.005) = 10.51516
exp_stop = (20 - 0.786 * 12) * (1 - 0.005)
check("fib stop matches H - ratio*(H-L), buffered",
      approx(stop.iloc[8], exp_stop), f"stop[8]={stop.iloc[8]} exp={exp_stop}")
check("fib target = prior pivot high H", approx(target.iloc[8], 20.0),
      f"target[8]={target.iloc[8]}")
check("no up-leg yet -> stop/target NaN (fallback)",
      pd.isna(stop.iloc[5]) and pd.isna(target.iloc[5]))
check("pivot_low ladder exposed for trailing", approx(pvl.iloc[8], 8.0))
check("different stop_ratio moves the stop",
      not approx(structure_levels(high, low, 2, 2, stop_ratio=0.618)[0]
                 .iloc[8], stop.iloc[8]))


# 2. NO LOOKAHEAD: prefix recompute must equal full-series value --------------
rng = np.random.default_rng(7)
steps = rng.normal(0, 1, 400).cumsum() + 100
h = pd.Series(steps + rng.random(400) * 2, dtype=float)
lo = pd.Series(steps - rng.random(400) * 2, dtype=float)
full_stop, full_tgt, full_pvl = structure_levels(h, lo, 5, 5)

mismatch = 0
checked = 0
for t in range(50, 400, 17):
    ps, pt, pv = structure_levels(h.iloc[:t + 1], lo.iloc[:t + 1], 5, 5)
    for full, pre in ((full_stop, ps), (full_tgt, pt), (full_pvl, pv)):
        fv, pvv = full.iloc[t], pre.iloc[t]
        checked += 1
        if pd.isna(fv) and pd.isna(pvv):
            continue
        if pd.isna(fv) or pd.isna(pvv) or not approx(fv, pvv, 1e-9):
            mismatch += 1

check(f"prefix recompute == full at bar t ({checked} points, no lookahead)",
      mismatch == 0, f"{mismatch} mismatches")


# 3. trend_points: three anchors of a fib trend move (leg-low, high, pullback)
# 5-bar fractal. pivot low 8 @idx2, pivot high 30 @idx8, pivot low 20 @idx14.
tp_low = pd.Series([12, 11, 8, 11, 13, 16, 19, 22, 25, 24, 23, 22, 21, 20.5,
                    20, 21, 22, 23, 24], dtype=float)
tp_high = pd.Series([13, 12, 9, 13, 15, 18, 21, 24, 30, 27, 25, 24, 23, 22,
                     21, 23, 24, 25, 26], dtype=float)
P1, P2, P3 = trend_points(tp_high, tp_low, 2, 2)

check("point1 = leg-start swing low (8)", approx(P1.iloc[16], 8.0),
      f"P1[16]={P1.iloc[16]}")
check("point2 = swing high (30)", approx(P2.iloc[16], 30.0),
      f"P2[16]={P2.iloc[16]}")
check("point3 = pullback low after the high (20)", approx(P3.iloc[16], 20.0),
      f"P3[16]={P3.iloc[16]}")
check("point3 is NaN before the pullback low confirms (idx12)",
      pd.isna(P3.iloc[12]) and approx(P1.iloc[12], 8.0)
      and approx(P2.iloc[12], 30.0))

# no-lookahead for the 3 anchors: prefix recompute must match
q1, q2, q3 = trend_points(h, lo, 5, 5)
mism = 0
tot = 0
for t in range(60, 400, 23):
    a1, a2, a3 = trend_points(h.iloc[:t + 1], lo.iloc[:t + 1], 5, 5)
    for full, pre in ((q1, a1), (q2, a2), (q3, a3)):
        tot += 1
        fv, pv = full.iloc[t], pre.iloc[t]
        if pd.isna(fv) and pd.isna(pv):
            continue
        if pd.isna(fv) or pd.isna(pv) or not approx(fv, pv, 1e-9):
            mism += 1
check(f"trend_points prefix recompute == full ({tot} points, no lookahead)",
      mism == 0, f"{mism} mismatches")


if __name__ == "__main__":
    print("\n" + ("ALL STRUCTURE TRUST TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
