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
    --htf-screen --w-htf_trend=1 --w-htf_ema_dist=1 --htf-threshold=0,0.5,1
        (weekly SETUP screen: eligible only if the weighted htf_* score clears
        the threshold -- higher-TF screens which stocks, entry times the rest)

Every result is scored against BUY-AND-HOLD of the names it traded (hold% =
equal-weight hold over the window); exc% = active CAGR - hold CAGR, and the
board is ranked by excess -- a config only "works" if it beats just holding
the same setups (walk-forward ranks by mean per-fold excess).

Out-of-sample: --oos-split=0.7 tunes/ranks configs on the earlier 70% (TRAIN)
and scores each on the held-out later 30% (TEST) it never saw -- so a config
whose te_CAGR holds near its tr_CAGR is robust; one that craters overfit.
Walk-forward: --wf-folds=6 splits the whole history into 6 contiguous folds
and scores every config on each; ranks by mean CAGR and shows wf_min / wf_pos
(folds positive) -- a config that's green across most folds survived regimes,
not one lucky window.
    --trail-dist=0.03,0.06,0.10   --target=0,0.06   --max-positions=3,5

Fixed (rebuild the grid to change): strategies, --gate, --market, --interval,
--months, and the fib pivot window --fib-left/--fib-right. Everything else is
applied at sim time, so it sweeps cheaply. Example:
    python sweep.py --months=12 --stop-mode=fib --fib-stop=0.618,0.786 \
        --trail-mode=pct,structure --fib-target=0,1 --cost-bps=10

Strategies from bot_strategies.json (or --strategies=names). ASCII output.
"""

import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from portfolio_multi import (BOT_FILE, FACTOR_NAMES, FIB_ENTRY_MODES,
                             HTF_START, STOP_MODES, TRAIL_MODES,
                             prepare_grid_cached)
from test_strategy import arg, fail, parse_gates


from weisswave import portsim


def _trail_label(tmode, ta, td):
    """What the ENGINE will actually do, not what trail-activate looks like.

    portsim: `do_trail = (trail_mode != TRAIL_PCT) or (trail_act > 0.0)`.
    structure/fib trail whenever selected -- they anchor to structure, so they
    need no activation threshold -- while pct needs trail_act > 0. So
    trail-activate=0 means OFF for pct and ON-FROM-ENTRY for structure/fib.

    This printed "-" whenever ta == 0 regardless of mode, so a structure or fib
    row showed "no trail" while a trail was running. That is why `structure -`
    (17.0% CAGR, 95 trades) and `pct -` (24.4%, 83) disagreed in a sweep where
    the trail was supposedly off in both: it was off in one and on in the other.
    Reporting a knob you did not apply is the same failure as reporting a trade
    that never fired.
    """
    if ta > 0:
        return f"{ta:.0%}/{td:.0%}"
    return f"0%/{td:.0%}" if tmode != "pct" else "-"


def listarg(args, name, default, cast):
    v = arg(args, name, None)
    return [cast(x) for x in v.split(",")] if v not in (None, "") else default


RESULTS_DIR = "sweep_results"


def grid_sig_of(interval, gate, market, months, universe="stocks",
                gate_mode="hard", fib_anchor="1d", sig_params=None):
    """Signature of the DATA + SEMANTICS a score was computed under. Two rows
    may only be compared (or deduped) when these match.

    universe and gate_mode belong here: a crypto score means nothing for a
    stocks config, and a score from a hard-gated grid means nothing for a
    gate-as-factor one. Both were previously missing, which made a silent
    cross-universe score reuse possible."""
    from test_strategy import sig_params_sig
    return (f"{interval}|{gate}|{gate_mode}|{market}|{months}mo|"
            f"{universe}|fib@{fib_anchor}|sig@{sig_params_sig(sig_params)}")


def save_results(df, meta, spec):
    """Append a run's full results to the store: one parquet per run under
    sweep_results/, tagged with a grid signature, scoring mode, spec and
    timestamp. Accumulated runs are queryable (e.g. duckdb
    read_parquet('sweep_results/*.parquet', union_by_name=true)) so an agent
    can see what's been tested and what won instead of re-running it."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = pd.Timestamp.now()
    out = df.copy()
    out["run_ts"] = ts.isoformat()
    out["grid_sig"] = grid_sig_of(meta["interval"], meta["gate"],
                                  meta["market"], meta["months"],
                                  meta.get("universe", "stocks"),
                                  meta.get("gate_mode", "hard"),
                                  meta.get("fib_anchor", "self"),
                                  meta.get("sig_params"))
    out["scoring"] = "wf" if meta["wf"] else "oos" if meta["oos"] else "full"
    out["spec"] = json.dumps(spec, default=str)
    fname = os.path.join(RESULTS_DIR,
                         f"run_{ts.strftime('%Y%m%d_%H%M%S_%f')}.parquet")
    out.to_parquet(fname)
    return fname


def load_results():
    """All accumulated sweep results, schema-unioned across runs (empty frame
    if none). The query surface for agents: 'what's been tested, what won'."""
    files = glob.glob(os.path.join(RESULTS_DIR, "*.parquet"))
    return (pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            if files else pd.DataFrame())


def _sweep(args, shard=None):
    """Run the sweep described by CLI-style `args`; return (full results
    DataFrame sorted by the ranking column, meta dict). No printing -- the
    callable core shared by the CLI and run_sweep(). `shard=(i, n)` runs only
    configs i::n (for parallel workers); the grid is loaded from cache so every
    shard shares it cheaply."""
    interval = arg(args, "interval", "15m")
    gate = arg(args, "gate", "sma50_over_200@1d,above_50ma@4h")
    market = arg(args, "market", "none")
    universe = arg(args, "universe", "stocks")   # stocks | crypto | all | list
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
    # higher-TF weekly setup screen (tunable): eligible only if the weighted
    # htf_* score clears --htf-threshold (a sweepable axis).
    htf_screen = 1 if ("--htf-screen" in args or arg(args, "htf-screen", "0")
                       not in ("0", "no", "false", "none", "")) else 0
    htf_thr_list = listarg(args, "htf-threshold", [0.0], float) \
        if htf_screen else [0.0]
    # out-of-sample split: rank configs on the earlier TRAIN slice, then score
    # each honestly on the later TEST slice it never saw. 0 = off (full period).
    oos_split = float(arg(args, "oos-split", "0"))
    oos = oos_split > 0.0
    # walk-forward: split the whole history into N contiguous folds and score
    # every config on each -- a robust edge holds across regimes, not one window.
    wf_folds = int(arg(args, "wf-folds", "0"))
    wf = wf_folds >= 2

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
        # Trailing ON by default (ride to +10%, then trail under the high),
        # matching portfolio_multi. This was 0.0 = OFF, so a sweep that did not
        # explicitly pass --trail-activate got the DEGENERATE stop-only case:
        # no target, no trail, no exit signal, no clock means the only way out
        # is the stop, so every closed trade is a loss BY CONSTRUCTION and every
        # row reads win=0.0%. Worse, the same config scored differently here
        # than in portfolio_multi -- the exact split the hold=0 fix was about.
        # Sweep --trail-activate=0 to test no-trailing deliberately.
        "ta": listarg(args, "trail-activate", [0.10], float),
        "td": listarg(args, "trail-dist", [0.03], float),
        "tgt": listarg(args, "target", [0.10], float),
        "mp": listarg(args, "max-positions", [5], int),
        # max-hold time exit: OFF by default (0). Exits should come from
        # stops / targets / trailing / reversal -- never a clock. Sweepable
        # only so a time exit can be *tested*, not imposed.
        "hold": listarg(args, "hold", [0], int),
        # bars after a STOP before the same symbol can re-enter; 0 = off
        "rcd": listarg(args, "reentry-cooldown", [0], int),
    }

    # HOW the gate reaches the engine. prepare_grid_cached defaulted this to
    # "hard" and sweep never exposed it, so gate_mode=factor -- Paul's
    # gate-as-a-dial -- was unreachable from the tool that runs the most
    # configs. It is in the grid cache key, so switching rebuilds correctly.
    gate_mode = arg(args, "gate-mode", "hard")
    if gate_mode not in ("hard", "factor"):
        fail(f"--gate-mode must be 'hard' or 'factor', got {gate_mode!r}")
    gates_on = bool(parse_gates(gate))

    t0 = time.time()
    (A, V, ENT, SIDX, EXT, syms, grid, st_stop, st_hold, st_tgt), cached = \
        prepare_grid_cached(strategies, interval, gate, market, months,
                            atr_len=atr_len, swing_look=swing_look, fib=fib,
                            universe=universe, gate_mode=gate_mode)
    build_s = time.time() - t0

    def sim_metrics(Ax, Vx, ENTx, SIDXx, EXTx, gridx,
                    sm, am, sb, fr, fb, tm, ftg, fe, ta, td, tg, mp, hd, rcd,
                    thr, htf_thr, wvec):
        """Run one config over a (possibly sliced) grid, return its metrics."""
        yrs = max((pd.Timestamp(gridx[-1]) - pd.Timestamp(gridx[0])).days
                  / 365.25, 1e-9)
        tgt_arr = np.full_like(st_tgt, tg) if tg > 0 else st_tgt
        hold_arr = np.full(st_hold.shape, hd, np.int64)   # 0 = no time exit
        res = portsim.simulate(
            Ax["O"], Ax["H"], Ax["L"], Ax["C"], Vx, ENTx, Ax["SCORE"], SIDXx,
            EXTx, Ax["ATR"], Ax["SW"], st_stop, hold_arr, tgt_arr,
            stop_mode=STOP_MODES.get(sm, 0), atr_mult=am, swing_buf=sb,
            trail_act=ta, trail_dist=td, cost_side=cost_side, max_pos=mp,
            init_cash=capital, p1=Ax["P1"], p2=Ax["P2"], p3=Ax["P3"],
            fib_stop_ratio=fr, fib_buf=fb, trail_mode=TRAIL_MODES.get(tm, 0),
            fib_ext=fib_ext, use_fib_target=ftg, gate=Ax["GATE"],
            fib_entry=FIB_ENTRY_MODES.get(fe, 0), fib_zone_lo=fib_zone_lo,
            fib_zone_hi=fib_zone_hi, fib_bounce_look=fib_bounce_look,
            factors=Ax["FACTORS"], weights=wvec, conf_entry=conf_entry,
            conf_threshold=thr, conf_size=conf_size, htf_start=HTF_START,
            htf_screen=htf_screen, htf_threshold=htf_thr,
            # conf_entry never reads `ent`, where the hard gate is baked
            gate_hard=int(gate_mode == "hard" and gates_on),
            reentry_cd=rcd)
        eq = pd.Series(res["equity"]); r = res["ret"]; n = len(r)
        cagr = round(((eq.iloc[-1] / capital) ** (1 / yrs) - 1) * 100, 1)
        # buy & hold benchmark: equal-weight hold of the names THIS config
        # traded, over the same slice. hold each from its first valid bar to
        # the last -> the "just buy the setups and hold them" alternative.
        rr = []
        for sy in np.unique(res["sym"]) if n else ():
            vc = Ax["C"][:, sy][Vx[:, sy]]
            if len(vc) >= 2 and vc[0] > 0:
                rr.append(vc[-1] / vc[0])
        hold = round((float(np.mean(rr)) ** (1 / yrs) - 1) * 100, 1) if rr else 0.0
        return {"CAGR": cagr, "hold": hold, "exc": round(cagr - hold, 1),
                "DD": round((eq / eq.cummax() - 1).min() * 100, 1),
                "inv": round(res["invested"].mean() * 100, 0),
                "n": n, "win": round((r > 0).mean() * 100, 1) if n else 0,
                "avg": round(r.mean() * 100, 2) if n else 0}

    full = (A, V, ENT, SIDX, EXT, grid)
    sub = lambda a, x, y: {k: v[x:y] for k, v in a.items()}
    seg = lambda x, y: (sub(A, x, y), V[x:y], ENT[x:y], SIDX[x:y], EXT[x:y],
                        grid[x:y])
    if wf:
        b = [int(len(grid) * i / wf_folds) for i in range(wf_folds + 1)]
        folds = [seg(b[i], b[i + 1]) for i in range(wf_folds)]
    elif oos:
        sp = int(len(grid) * oos_split)
        tr, te = seg(0, sp), seg(sp, None)

    combos = list(itertools.product(
        ax["stop"], ax["atrm"], ax["swb"], ax["fr"], ax["fb"], ax["tm"],
        ax["ftg"], ax["fe"], ax["ta"], ax["td"], ax["tgt"], ax["mp"],
        ax["hold"], ax["rcd"], thr_list, htf_thr_list, *w_lists))
    n_all = len(combos)
    if shard is not None:
        combos = combos[shard[0]::shard[1]]     # this worker's stride
    rows = []
    t1 = time.time()
    for combo in combos:
        sm, am, sb, fr, fb, tm, ftg, fe, ta, td, tg, mp, hd, rcd = combo[:14]
        thr, htf_thr = combo[14], combo[15]
        wvec = np.array(combo[16:])            # per-factor weights (order=names)
        p = (sm, am, sb, fr, fb, tm, ftg, fe, ta, td, tg, mp, hd, rcd, thr,
             htf_thr, wvec)
        row = {
            "stop": sm, "atrm": am, "swb": sb, "fibr": fr, "fibbuf": fb,
            "tmode": tm, "ftgt": ftg, "entry": fe,
            "trail": _trail_label(tm, ta, td),
            "tgt": f"{tg:.0%}" if tg > 0 else "-", "mp": mp,
            "hold": hd, "rcd": rcd}
        if conf_entry:
            row["thr"] = thr
            for nm, wv in zip(FACTOR_NAMES, wvec):
                row[f"w_{nm}"] = wv
        if htf_screen:
            row["htf_thr"] = htf_thr
        if wf:
            # rank by EXCESS over buy-&-hold, per fold (not raw return)
            ms = [sim_metrics(*fold, *p) for fold in folds]
            exc = [m["exc"] for m in ms]
            row.update({"foldsExc%": " ".join(f"{c:g}" for c in exc),
                        "wf_exc": round(sum(exc) / len(exc), 1),
                        "wf_min": round(min(exc), 1),
                        "wf_pos": f"{sum(c > 0 for c in exc)}/{wf_folds}",
                        "wf_CAGR": round(sum(m["CAGR"] for m in ms) / len(ms), 1),
                        "wf_hold": round(sum(m["hold"] for m in ms) / len(ms), 1)})
        elif oos:
            mtr = sim_metrics(*tr, *p)
            mte = sim_metrics(*te, *p)
            row.update({"tr_exc": mtr["exc"], "te_CAGR": mte["CAGR"],
                        "te_hold": mte["hold"], "te_exc": mte["exc"],
                        "te_DD": mte["DD"], "te_n": mte["n"]})
        else:
            m = sim_metrics(*full, *p)
            row.update({"CAGR%": m["CAGR"], "hold%": m["hold"], "exc%": m["exc"],
                        "maxDD%": m["DD"], "inv%": m["inv"], "n": m["n"],
                        "win%": m["win"], "avg%": m["avg"]})
        rows.append(row)
    sweep_s = time.time() - t1

    sort_col = "wf_exc" if wf else "te_exc" if oos else "exc%"
    df = pd.DataFrame(rows).sort_values(sort_col, ascending=False) \
        .reset_index(drop=True)
    drop_cols = ["stop", "atrm", "swb", "fibr", "fibbuf", "tmode", "ftgt",
                 "entry", "trail", "tgt", "mp", "hold", "rcd"]
    if conf_entry:
        drop_cols += ["thr"] + [f"w_{n}" for n in FACTOR_NAMES]
    if htf_screen:
        drop_cols += ["htf_thr"]
    meta = {"interval": interval, "gate": gate, "market": market,
            "months": months, "universe": universe,
            "n_syms": len(syms), "n_bars": len(grid),
            "start": str(pd.Timestamp(grid[0]).date()),
            "end": str(pd.Timestamp(grid[-1]).date()), "cached": cached,
            "build_s": build_s, "sweep_s": sweep_s, "n_combos": len(combos),
            "n_sims": len(combos) * (wf_folds if wf else 2 if oos else 1),
            "sort_col": sort_col, "drop_cols": drop_cols, "top": top,
            "wf": wf, "wf_folds": wf_folds, "oos": oos, "oos_split": oos_split,
            "fold_starts": [str(pd.Timestamp(grid[b[i]]).date())
                            for i in range(wf_folds)] if wf else [],
            "oos_test_start": str(pd.Timestamp(grid[sp]).date()) if oos else ""}
    return df, meta


def _display(df, meta):
    """Pretty-print a sweep result (drops constant swept columns)."""
    d = df.copy()
    for c in meta["drop_cols"]:
        if c in d.columns and d[c].nunique() == 1:
            d = d.drop(columns=c)
    pd.set_option("display.width", 220)
    print(f"grid: {meta['n_syms']} syms x {meta['n_bars']} bars, "
          f"{meta['interval']}, gate={meta['gate']}, market={meta['market']}"
          + (f", last {meta['months']}mo" if meta["months"] else ""))
    if meta["wf"]:
        print(f"walk-forward {meta['wf_folds']} folds (foldsExc% = EXCESS over "
              f"buy&hold per fold; ranked by mean excess); fold starts: "
              f"{' | '.join(meta['fold_starts'])} | {meta['end']}")
    elif meta["oos"]:
        print(f"OOS split {meta['oos_split']:.0%}: train {meta['start']} -> "
              f"test {meta['oos_test_start']} -> {meta['end']}   (ranked by TEST "
              f"excess over buy&hold; exc = active - hold)")
    else:
        print("ranked by exc% = active CAGR - buy&hold CAGR (of the traded "
              "names over the window)")
    print(f"grid {'loaded from cache' if meta['cached'] else 'built'} in "
          f"{meta['build_s']:.1f}s; swept {meta['n_combos']} configs "
          f"({meta['n_sims']} sims) in {meta['sweep_s']:.1f}s "
          f"({meta['sweep_s'] / meta['n_sims'] * 1000:.0f} ms/sim "
          f"incl first-call compile)\n")
    print(d.head(meta["top"]).to_string(index=False))


def _spec_to_args(spec):
    """Turn a {name: value} spec dict into CLI-style --name=value args (lists
    become comma lists, True becomes a bare flag) so agents can drive the sweep
    programmatically through the same tested parser the CLI uses."""
    out = []
    for k, v in spec.items():
        if v is True:
            out.append(f"--{k}")
        elif v is False or v is None:
            continue
        elif isinstance(v, (list, tuple)):
            out.append(f"--{k}=" + ",".join(str(x) for x in v))
        else:
            out.append(f"--{k}={v}")
    return out


def _sweep_shard(args, i, n):
    """Module-level worker: run configs i::n. Loads the grid from cache."""
    return _sweep(args, shard=(i, n))


def _run(args, jobs):
    """Run a sweep serially (jobs<=1) or across `jobs` processes. The parent
    runs shard 0 first (building/caching the grid), then workers hit the cache
    -- so parallelism costs one grid build, not N."""
    if jobs <= 1:
        return _sweep(args)
    import multiprocessing as mp
    df0, meta = _sweep(args, shard=(0, jobs))      # builds + caches the grid
    with mp.Pool(jobs - 1) as pool:
        rest = pool.starmap(_sweep_shard, [(args, i, jobs)
                                           for i in range(1, jobs)])
    dfs = [df0] + [d for d, _ in rest]
    metas = [meta] + [m for _, m in rest]
    df = pd.concat(dfs, ignore_index=True) \
        .sort_values(meta["sort_col"], ascending=False).reset_index(drop=True)
    meta = {**meta, "n_combos": sum(m["n_combos"] for m in metas),
            "n_sims": sum(m["n_sims"] for m in metas),
            "sweep_s": max(m["sweep_s"] for m in metas),
            "build_s": max(m["build_s"] for m in metas)}
    return df, meta


def run_sweep(spec, save=True, jobs=1):
    """Programmatic entry point (the agent API): run a sweep from a spec dict,
    persist the full results to the store, and return the results DataFrame.
    `spec` keys are the CLI flag names, e.g. {"months": 12, "stop-mode": "fib",
    "conf-entry": True, "w-signal": [0, 1, 2], "wf-folds": 6}. `jobs` fans the
    configs across processes (the cached grid is shared, cheaply)."""
    df, meta = _run(_spec_to_args(spec), jobs)
    if save:
        save_results(df, meta, spec)
    return df


def main():
    args = sys.argv[1:]
    df, meta = _run(args, int(arg(args, "jobs", "1")))
    if "--save-results" in args:
        save_results(df, meta, {"argv": " ".join(args)})
    _display(df, meta)


if __name__ == "__main__":
    main()
