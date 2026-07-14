#!/usr/bin/env python3
"""Fast exit-parameter sweep — the payoff of the numba engine. Builds the
signal grid ONCE (the slow part), then runs every combination of exit params
through the numba sim (each = milliseconds). Turns a multi-minute sweep into
one grid-build plus near-instant configs.

Sweep axes accept comma lists; the cartesian product is run:
    --stop-mode=swing,atr   --atr-mult=2,3   --swing-buf=0.005,0.01
    --trail-activate=0,0.04,0.06   --trail-dist=0.03,0.06,0.10
    --target=0,0.06   --max-positions=3,5

Fixed (rebuild the grid to change): strategies, --gate, --market, --interval,
--months, --window. Example:
    python sweep.py --months=12 --trail-activate=0.04,0.06 \
        --trail-dist=0.03,0.06,0.10 --stop-mode=swing,atr --cost-bps=10

Strategies from bot_strategies.json (or --strategies=names). ASCII output.
"""

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from portfolio_multi import BOT_FILE, STOP_MODES, prepare_grid_cached
from test_strategy import arg, fail
from weisswave import portsim


def listarg(args, name, default, cast):
    v = arg(args, name, None)
    return [cast(x) for x in v.split(",")] if v not in (None, "") else default


def main():
    args = sys.argv[1:]
    interval = arg(args, "interval", "15m")
    gate = arg(args, "gate", "minervini@1d,above_50ma@4h")
    market = arg(args, "market", "none")
    months = int(arg(args, "months", "0"))
    cost_side = float(arg(args, "cost-bps", "0")) / 10000.0 / 2
    capital = float(arg(args, "capital", "100000"))
    swing_look = int(arg(args, "swing-look", "20"))
    atr_len = int(arg(args, "atr-len", "14"))
    top = int(arg(args, "top", "25"))

    with open(arg(args, "file", BOT_FILE), encoding="utf-8") as f:
        strategies = json.load(f)
    only = arg(args, "strategies", None)
    if only:
        want = set(only.split(","))
        strategies = [s for s in strategies if s["name"] in want]
    if not strategies:
        fail("no strategies")

    ax = {
        "stop": listarg(args, "stop-mode", ["swing"], str),
        "atrm": listarg(args, "atr-mult", [2.5], float),
        "swb": listarg(args, "swing-buf", [0.005], float),
        "ta": listarg(args, "trail-activate", [0.0], float),
        "td": listarg(args, "trail-dist", [0.03], float),
        "tgt": listarg(args, "target", [0.0], float),
        "mp": listarg(args, "max-positions", [5], int),
    }

    t0 = time.time()
    (A, V, ENT, SIDX, EXT, syms, grid, st_stop, st_hold, st_tgt), cached = \
        prepare_grid_cached(strategies, interval, gate, market, months,
                            atr_len=atr_len, swing_look=swing_look)
    build_s = time.time() - t0
    years = max((pd.Timestamp(grid[-1]) - pd.Timestamp(grid[0])).days / 365.25,
                1e-9)

    combos = list(itertools.product(ax["stop"], ax["atrm"], ax["swb"],
                                    ax["ta"], ax["td"], ax["tgt"], ax["mp"]))
    rows = []
    t1 = time.time()
    for sm, am, sb, ta, td, tg, mp in combos:
        tgt_arr = np.full_like(st_tgt, tg) if tg > 0 else st_tgt
        res = portsim.simulate(
            A["O"], A["H"], A["L"], A["C"], V, ENT, A["SCORE"], SIDX, EXT,
            A["ATR"], A["SW"], st_stop, st_hold, tgt_arr,
            stop_mode=STOP_MODES.get(sm, 0), atr_mult=am, swing_buf=sb,
            trail_act=ta, trail_dist=td, cost_side=cost_side, max_pos=mp,
            init_cash=capital)
        eq = pd.Series(res["equity"])
        r = res["ret"]
        n = len(r)
        rows.append({
            "stop": sm, "atrm": am, "swb": sb, "trail": f"{ta:.0%}/{td:.0%}"
            if ta > 0 else "-", "tgt": f"{tg:.0%}" if tg > 0 else "-", "mp": mp,
            "CAGR%": round(((eq.iloc[-1] / capital) ** (1 / years) - 1) * 100, 1),
            "maxDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
            "inv%": round(res["invested"].mean() * 100, 0),
            "n": n, "win%": round((r > 0).mean() * 100, 1) if n else 0,
            "avg%": round(r.mean() * 100, 2) if n else 0,
        })
    sweep_s = time.time() - t1

    df = pd.DataFrame(rows).sort_values("CAGR%", ascending=False)
    # drop constant columns for readability
    for c in ["stop", "atrm", "swb", "trail", "tgt", "mp"]:
        if df[c].nunique() == 1:
            df = df.drop(columns=c)
    pd.set_option("display.width", 200)
    print(f"grid: {len(syms)} syms x {len(grid)} bars, {interval}, gate={gate}, "
          f"market={market}" + (f", last {months}mo" if months else ""))
    print(f"grid {'loaded from cache' if cached else 'built'} in {build_s:.1f}s; "
          f"swept {len(combos)} configs in "
          f"{sweep_s:.1f}s ({sweep_s / len(combos) * 1000:.0f} ms/config "
          f"incl first-call compile)\n")
    print(df.head(top).to_string(index=False))


if __name__ == "__main__":
    main()
