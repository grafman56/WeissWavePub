#!/usr/bin/env python3
"""Fast exit-parameter sweep — the payoff of the numba engine. Builds the
signal grid ONCE (the slow part), then runs every combination of exit params
through the numba sim (each = milliseconds). Turns a multi-minute sweep into
one grid-build plus near-instant configs.

Sweep axes accept comma lists; the cartesian product is run:
    --stop-mode=swing,atr,fib   --atr-mult=2,3   --swing-buf=0.005,0.01
    --fib-stop=0.618,0.786   --fib-buf=0.005   --fib-target=0,1
    --fib-entry=off,zone,bounce,bounce-trend  (pullback-into-zone entry;
        bounce needs an up-close off the band, bounce-trend needs only trend)
    --trail-mode=pct,structure,fib   --fib-ext=1.0,1.618  (fib-ladder trail)
    --trail-activate=0,0.04,0.06
    --conf-entry  --w-signal=0,1,2  --w-trend=0,1  --w-fib_prox=0,1,2
        --conf-threshold=0.5,1,1.5   (weighted-confluence entry; each factor's
        --w-<name> is a sweep axis, 0 = mute it)
    --trail-dist=0.03,0.06,0.10   --target=0,0.06   --max-positions=3,5

Fixed (rebuild the grid to change): strategies, --gate, --market, --interval,
--months, and the fib pivot window --fib-left/--fib-right. Everything else is
applied at sim time, so it sweeps cheaply. Example:
    python sweep.py --months=12 --stop-mode=fib --fib-stop=0.618,0.786 \
        --trail-mode=pct,structure --fib-target=0,1 --cost-bps=10

Strategies from bot_strategies.json (or --strategies=names). ASCII output.
"""

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from portfolio_multi import (BOT_FILE, FACTOR_NAMES, FIB_ENTRY_MODES,
                             STOP_MODES, TRAIL_MODES, prepare_grid_cached)
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
    # fib pivot window is a BUILD param (baked into the grid); the ratio/buf/
    # target/trail-mode/zone are swept at sim time below.
    fib = {"left": int(arg(args, "fib-left", "10")),
           "right": int(arg(args, "fib-right", "10"))}
    fib_zone_lo = float(arg(args, "fib-zone-lo", "0.5"))
    fib_zone_hi = float(arg(args, "fib-zone-hi", "0.786"))
    fib_bounce_look = int(arg(args, "fib-bounce-look", "3"))
    fib_ext = [float(x) for x in arg(args, "fib-ext", "1.0,1.272,1.618,2.0")
               .split(",")]
    # confluence entry: when on, sweep the per-factor weights + threshold
    conf_entry = 1 if ("--conf-entry" in args or arg(args, "conf-entry", "0")
                       not in ("0", "no", "false", "none", "")) else 0
    conf_size = 1 if ("--conf-size" in args or arg(args, "conf-size", "0")
                      not in ("0", "no", "false", "none", "")) else 0
    if conf_entry:
        w_lists = [listarg(args, f"w-{n}", [1.0], float) for n in FACTOR_NAMES]
        thr_list = listarg(args, "conf-threshold", [1.0], float)
    else:
        w_lists = [[1.0] for _ in FACTOR_NAMES]   # muted: no wasted combos
        thr_list = [1.0]

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
        "fr": listarg(args, "fib-stop", [0.786], float),   # fib retr. ratio
        "fb": listarg(args, "fib-buf", [0.005], float),
        "tm": listarg(args, "trail-mode", ["pct"], str),
        "ftg": listarg(args, "fib-target", [0], int),      # 0/1 use fib tp
        "fe": listarg(args, "fib-entry", ["off"], str),    # off/zone/bounce/..
        "ta": listarg(args, "trail-activate", [0.0], float),
        "td": listarg(args, "trail-dist", [0.03], float),
        "tgt": listarg(args, "target", [0.0], float),
        "mp": listarg(args, "max-positions", [5], int),
    }

    t0 = time.time()
    (A, V, ENT, SIDX, EXT, syms, grid, st_stop, st_hold, st_tgt), cached = \
        prepare_grid_cached(strategies, interval, gate, market, months,
                            atr_len=atr_len, swing_look=swing_look, fib=fib)
    build_s = time.time() - t0
    years = max((pd.Timestamp(grid[-1]) - pd.Timestamp(grid[0])).days / 365.25,
                1e-9)

    combos = list(itertools.product(
        ax["stop"], ax["atrm"], ax["swb"], ax["fr"], ax["fb"], ax["tm"],
        ax["ftg"], ax["fe"], ax["ta"], ax["td"], ax["tgt"], ax["mp"],
        thr_list, *w_lists))
    rows = []
    t1 = time.time()
    for combo in combos:
        sm, am, sb, fr, fb, tm, ftg, fe, ta, td, tg, mp, thr = combo[:13]
        wvec = np.array(combo[13:])            # per-factor weights (order=names)
        tgt_arr = np.full_like(st_tgt, tg) if tg > 0 else st_tgt
        res = portsim.simulate(
            A["O"], A["H"], A["L"], A["C"], V, ENT, A["SCORE"], SIDX, EXT,
            A["ATR"], A["SW"], st_stop, st_hold, tgt_arr,
            stop_mode=STOP_MODES.get(sm, 0), atr_mult=am, swing_buf=sb,
            trail_act=ta, trail_dist=td, cost_side=cost_side, max_pos=mp,
            init_cash=capital, p1=A["P1"], p2=A["P2"], p3=A["P3"],
            fib_stop_ratio=fr, fib_buf=fb, trail_mode=TRAIL_MODES.get(tm, 0),
            fib_ext=fib_ext, use_fib_target=ftg, gate=A["GATE"],
            fib_entry=FIB_ENTRY_MODES.get(fe, 0), fib_zone_lo=fib_zone_lo,
            fib_zone_hi=fib_zone_hi, fib_bounce_look=fib_bounce_look,
            factors=A["FACTORS"], weights=wvec, conf_entry=conf_entry,
            conf_threshold=thr, conf_size=conf_size)
        eq = pd.Series(res["equity"])
        r = res["ret"]
        n = len(r)
        row = {
            "stop": sm, "atrm": am, "swb": sb, "fibr": fr, "fibbuf": fb,
            "tmode": tm, "ftgt": ftg, "entry": fe, "trail": f"{ta:.0%}/{td:.0%}"
            if ta > 0 else "-", "tgt": f"{tg:.0%}" if tg > 0 else "-", "mp": mp}
        if conf_entry:
            row["thr"] = thr
            for nm, wv in zip(FACTOR_NAMES, wvec):
                row[f"w_{nm}"] = wv
        rows.append({**row,
            "CAGR%": round(((eq.iloc[-1] / capital) ** (1 / years) - 1) * 100, 1),
            "maxDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
            "inv%": round(res["invested"].mean() * 100, 0),
            "n": n, "win%": round((r > 0).mean() * 100, 1) if n else 0,
            "avg%": round(r.mean() * 100, 2) if n else 0,
        })
    sweep_s = time.time() - t1

    df = pd.DataFrame(rows).sort_values("CAGR%", ascending=False)
    # drop constant columns for readability
    drop_cols = ["stop", "atrm", "swb", "fibr", "fibbuf", "tmode", "ftgt",
                 "entry", "trail", "tgt", "mp"]
    if conf_entry:
        drop_cols += ["thr"] + [f"w_{n}" for n in FACTOR_NAMES]
    for c in drop_cols:
        if c in df.columns and df[c].nunique() == 1:
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
