#!/usr/bin/env python3
"""Trust tests for the backtest engine (weisswave.study._simulate and
backtest_long). Hand-built scenarios with KNOWN correct answers, verifying
the properties a backtester must never get wrong:

  - next-open execution (act on the bar AFTER the signal, never the same bar)
  - stop fills (at the stop, or the open when it gaps through)
  - target fills (at the target, or the open when it gaps through)
  - stop-checked-before-target when a bar spans both (conservative)
  - signal and time exits at the next open
  - NO LOOKAHEAD (a trade cannot depend on bars after it exits)
  - no overlapping positions
  - correct return arithmetic

Run:  python test_engine.py   (exit 0 = all pass; prints PASS/FAIL per test)
Designed to be run by CI or an orchestrator agent to validate any change.
"""

import sys

import numpy as np
import pandas as pd

from weisswave.study import _simulate, backtest_long

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def sim(o, lo, hi, c, ent, ext=None, stop=None, max_bars=None, tp=None):
    n = len(o)
    ext = ext or [False] * n
    return _simulate(np.array(o, float), np.array(lo, float),
                     np.array(hi, float), np.array(c, float),
                     np.array(ent, bool), np.array(ext, bool),
                     stop, max_bars, tp)


# 1. next-open entry + time exit -----------------------------------------------
# signal on bar 0 -> enter open[1]; max_bars=2 -> exit two bars later at open
t = sim(o=[10, 11, 12, 13, 14, 15], lo=[10, 11, 12, 13, 14, 15],
        hi=[10, 11, 12, 13, 14, 15], c=[10, 11, 12, 13, 14, 15],
        ent=[True, False, False, False, False, False], max_bars=2)
check("next-open entry price", t[0][2] == 11, f"entry_px={t[0][2]}")
check("time exit price/reason", t[0][3] == 14 and t[0][4] == "time",
      f"exit_px={t[0][3]} reason={t[0][4]}")

# 2. stop fill at stop price ---------------------------------------------------
t = sim(o=[10, 11, 11, 11], lo=[10, 11, 11, 9], hi=[10, 11, 11, 11],
        c=[10, 11, 11, 11], ent=[True, False, False, False], stop=0.10)
# entry 11, stop_px=9.9; bar3 low 9 <= 9.9, open 11 > 9.9 -> fill at 9.9
check("stop fills at stop price", abs(t[0][3] - 9.9) < 1e-9 and t[0][4] == "stop",
      f"exit_px={t[0][3]}")

# 3. stop gaps through -> fill at open ----------------------------------------
t = sim(o=[10, 11, 11, 9], lo=[10, 11, 11, 8], hi=[10, 11, 11, 9],
        c=[10, 11, 11, 9], ent=[True, False, False, False], stop=0.10)
# stop_px=9.9 but bar3 opens 9 (< stop) -> fill at open 9, not 9.9
check("stop gap-through fills at open", t[0][3] == 9 and t[0][4] == "stop",
      f"exit_px={t[0][3]}")

# 4. target fill --------------------------------------------------------------
t = sim(o=[10, 11, 11, 11], lo=[10, 11, 11, 11], hi=[10, 11, 11, 13],
        c=[10, 11, 11, 11], ent=[True, False, False, False], tp=0.10)
# entry 11, tp_px=12.1; bar3 high 13 >= 12.1, open 11 < 12.1 -> fill at 12.1
check("target fills at target price",
      abs(t[0][3] - 12.1) < 1e-9 and t[0][4] == "target", f"exit_px={t[0][3]}")

# 5. stop checked before target when a bar spans both -------------------------
t = sim(o=[10, 11, 11, 11], lo=[10, 11, 11, 9], hi=[10, 11, 11, 13],
        c=[10, 11, 11, 11], ent=[True, False, False, False], stop=0.10, tp=0.10)
check("stop wins when bar spans stop+target", t[0][4] == "stop",
      f"reason={t[0][4]}")

# 6. signal exit at next open -------------------------------------------------
t = sim(o=[10, 11, 12, 13, 14], lo=[10, 11, 12, 13, 14],
        hi=[10, 11, 12, 13, 14], c=[10, 11, 12, 13, 14],
        ent=[True, False, False, False, False],
        ext=[False, False, True, False, False])
# exit signal on bar 2 close -> sell open[3]=13
check("signal exit at next open", t[0][3] == 13 and t[0][4] == "signal",
      f"exit_px={t[0][3]} reason={t[0][4]}")

# 7. NO LOOKAHEAD: changing bars AFTER the exit must not change the trade ------
base = dict(o=[10, 11, 11, 9, 20, 20], lo=[10, 11, 11, 8, 20, 20],
            hi=[10, 11, 11, 9, 20, 20], c=[10, 11, 11, 9, 20, 20],
            ent=[True, False, False, False, False, False], stop=0.10)
t1 = sim(**base)
altered = dict(base); altered["o"] = [10, 11, 11, 9, 999, 999]
altered["hi"] = [10, 11, 11, 9, 999, 999]; altered["lo"] = [10, 11, 11, 8, 999, 999]
t2 = sim(**altered)
check("no lookahead (post-exit bars ignored)", t1[0] == t2[0],
      f"{t1[0]} vs {t2[0]}")

# 8. no overlapping positions: a signal during an open trade is ignored --------
t = sim(o=[10, 11, 12, 13, 14, 15], lo=[10, 11, 12, 13, 14, 15],
        hi=[10, 11, 12, 13, 14, 15], c=[10, 11, 12, 13, 14, 15],
        ent=[True, True, False, False, False, False], max_bars=2)
# first trade holds bars 1..4; the signal on bar 1 must not open a 2nd trade
starts = [tr[0] for tr in t]
check("no overlapping positions", len(t) == 1 or starts[1] > t[0][1],
      f"trade starts={starts}")

# 9. return arithmetic via backtest_long --------------------------------------
idx = pd.date_range("2026-01-01", periods=5, freq="D")
df = pd.DataFrame({"Open": [10, 10, 10, 10, 12], "High": [10, 10, 10, 10, 12],
                   "Low": [10, 10, 10, 10, 12], "Close": [10, 10, 10, 10, 12]},
                  index=idx)
entry = pd.Series([True, False, False, False, False], index=idx)
exit_ = pd.Series([False, False, False, True, False], index=idx)
res = backtest_long(df, entry, exit_)
# enter open[1]=10, exit signal bar3 -> open[4]=12 -> ret +20%
check("return arithmetic", res.n_trades == 1
      and abs(res.trades["ret"].iloc[0] - 0.20) < 1e-9,
      f"ret={res.trades['ret'].iloc[0] if res.n_trades else 'n/a'}")


if __name__ == "__main__":
    print(f"\n{'='*50}\n{len(FAILS)} failures"
          if FAILS else "\nALL ENGINE TRUST TESTS PASSED")
    if FAILS:
        print("FAILED:", ", ".join(FAILS))
    sys.exit(1 if FAILS else 0)
