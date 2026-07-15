#!/usr/bin/env python3
"""Raw indicator smoke: every bull indicator alone, and every PAIR of them.

No gate, no filter, no market screen. Just the indicator, a stop, and a target,
so the number is the indicator's own and nothing else's.

    python combo_smoke.py                        # TSLA 1d, last 12mo
    python combo_smoke.py --symbols=TSLA,NVDA --months=24 --interval=1d
    python combo_smoke.py --stop=0.05 --target=0.2 --window=10

WHY ONE PROCESS AND NOT 406 CLI CALLS: the slow part is building the signal
frame; the backtest itself is milliseconds. Build once, evaluate 406 times --
the same trick sweep.py uses. 406 subprocesses would be ~20 minutes of paying
for the same frame over and over.

PAIRS ARE CONFLUENCE, NOT UNION. A pair runs at min_count=2, so BOTH indicators
must fire within `window` bars. min_count=1 would just be "either one", which is
a different (and much looser) question -- and would make every pair score
between its two singles by construction, telling you nothing.

READ THE n COLUMN BEFORE READING ANYTHING ELSE. One symbol over one year of 1d
is ~250 bars. Most pairs will not fire at all, and a pair that fires twice is
not evidence of anything. This exists to exercise the machinery and to show
WHICH combinations are even reachable -- it is not a search for edge, and the
ranking is not meaningful at these counts. Widen the universe before believing
any row.
"""

import sys
from itertools import combinations

import numpy as np
import pandas as pd

from test_strategy import arg
from weisswave.db import connect, load_prices
from weisswave.optimize import evaluate_config
from weisswave.signals import SIGNAL_COLUMNS_BULL, build_signals

# test_strategy's defaults, so a row here is comparable to a row there. They are
# DEFAULTS, not constants: every one is a flag. This file shipped them as module
# literals for about an hour, which is the exact thing RNG_LOOK=20 and
# SIGNAL_NORM=3.0 are on the backlog for. A constant that encodes a tradeoff is
# a decision nobody can revisit.
DEF_STOP = 0.10
DEF_TARGET = 0.10
DEF_WINDOW = 5
HOLD = None          # no clock, and NOT a flag: hold=0 is permanent (see CLAUDE.md)


def stats(tr):
    if not len(tr):
        return dict(n=0, win=np.nan, avg=np.nan, pf=np.nan)
    r = tr["ret"]
    w, l = r[r > 0], r[r <= 0]
    pf = (w.sum() / abs(l.sum())) if len(l) and l.sum() else np.inf
    return dict(n=len(r), win=100 * (r > 0).mean(), avg=100 * r.mean(), pf=pf)


def main():
    args = sys.argv[1:]
    syms = arg(args, "symbols", "TSLA").split(",")
    interval = arg(args, "interval", "1d")
    months = int(arg(args, "months", "12"))
    STOP = float(arg(args, "stop", str(DEF_STOP)))
    TARGET = float(arg(args, "target", str(DEF_TARGET)))
    WINDOW = int(arg(args, "window", str(DEF_WINDOW)))
    cutoff = (pd.Timestamp.now() - pd.DateOffset(months=months)
              if months else None)

    con = connect(read_only=True)
    frames = {}
    for s in syms:
        df = load_prices(con, s, interval)
        if df is None or not len(df):
            print(f"no {interval} data for {s}")
            return 2
        # build on FULL history (indicators need warm-up), then cut to the
        # window. Cutting first would leave the first ~200 bars of a 1d run
        # computing SMA200 against nothing.
        sig = build_signals(df)
        frames[s] = sig[sig.index >= cutoff] if cutoff is not None else sig

    bars = sum(len(f) for f in frames.values())
    cols = [c for c in SIGNAL_COLUMNS_BULL if all(c in f.columns
                                                  for f in frames.values())]
    missing = [c for c in SIGNAL_COLUMNS_BULL if c not in cols]
    print(f"universe: {','.join(syms)}  {interval}  last {months}mo  "
          f"{bars} bars")
    print(f"indicators: {len(cols)}"
          + (f"  (SKIPPED, not in frame: {', '.join(missing)})" if missing
             else ""))
    print(f"stop={STOP:.0%} target={TARGET:.0%} window={WINDOW} hold=0  "
          f"gate=none filter=none\n")

    rows = []
    for c in cols:
        tr = evaluate_config(frames, [c], 1, WINDOW, None, [], STOP, HOLD,
                             take_profit=TARGET)
        rows.append(dict(kind="single", combo=c, fires=int(
            sum(f[c].fillna(False).sum() for f in frames.values())), **stats(tr)))
    for a, b in combinations(cols, 2):
        tr = evaluate_config(frames, [a, b], 2, WINDOW, None, [], STOP, HOLD,
                             take_profit=TARGET)
        rows.append(dict(kind="pair", combo=f"{a} + {b}", fires=np.nan,
                         **stats(tr)))

    df = pd.DataFrame(rows)
    df.to_csv("combo_smoke.csv", index=False)

    live = df[df.n > 0]
    print(f"{len(df)} configs: {len(df[df.kind=='single'])} singles, "
          f"{len(df[df.kind=='pair'])} pairs")
    print(f"{len(live)} produced ANY trade; {len(df)-len(live)} never fired\n")

    print("SINGLES (fires = bars the indicator was true; n = trades taken)")
    s = df[df.kind == "single"].sort_values("n", ascending=False)
    print(f"  {'indicator':24s} {'fires':>5s} {'n':>4s} {'win%':>6s} "
          f"{'avg%':>7s} {'PF':>6s}")
    for _, r in s.iterrows():
        print(f"  {r.combo:24s} {int(r.fires):5d} {int(r.n):4d} "
              f"{r.win:6.1f} {r.avg:+7.2f} "
              + (f"{r.pf:6.2f}" if np.isfinite(r.pf) else "   inf"))

    p = df[(df.kind == "pair") & (df.n > 0)].sort_values("n", ascending=False)
    print(f"\nPAIRS THAT FIRED AT ALL: {len(p)} of "
          f"{len(df[df.kind=='pair'])}")
    print(f"  {'combination':52s} {'n':>4s} {'win%':>6s} {'avg%':>7s}")
    for _, r in p.head(25).iterrows():
        print(f"  {r.combo:52s} {int(r.n):4d} {r.win:6.1f} {r.avg:+7.2f}")

    print(f"\nwrote combo_smoke.csv")
    print(f"n is TINY at one symbol / one year. This shows WHICH combinations "
          f"are reachable,\nnot which ones work. Do not rank on these counts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
