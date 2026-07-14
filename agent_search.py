#!/usr/bin/env python3
"""Agent-scale search over the confluence weight / exit space, optimizing the
one metric that can't be gamed: WALK-FORWARD EXCESS over buy-and-hold.

Loads the (cached) grid once, seeds with random configs, then evolves the best
(elitism + mutation) toward high mean per-fold excess. Every evaluated config is
persisted to sweep_results/ so agents -- and reruns -- build on what's been
tested instead of repeating it. This is the search engine; an agent can drive it
(propose seeds, read the store, request more generations) or run it as-is.

    python agent_search.py --universe=crypto --gate=minervini@1d --wf-folds=6 \
        --iters=400 --gens=6

ASCII output. Defaults target crypto, where a real active-beats-hold edge exists.
"""

import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

from portfolio_multi import (FACTOR_NAMES, FIB_ENTRY_MODES, HTF_START,
                             STOP_MODES, TRAIL_MODES, prepare_grid_cached)
from sweep import RESULTS_DIR, load_results, save_results
from test_strategy import arg
from weisswave import portsim

K = len(FACTOR_NAMES)
STOPS, TRAILS = ["fib", "swing", "atr", "pct"], ["pct", "structure", "fib"]
FIB_EXT = (1.0, 1.272, 1.618, 2.0)


def cfg_sig(c):
    """A canonical signature of a config (weights + params, coarsely rounded)
    so near-identical configs dedup. Lets a search skip anything already tested
    -- in this run or a prior one on the same data -- instead of repeating it."""
    parts = [f"{w:.1f}" for w in c["w"]] + [
        f"{c['thr']:.1f}", f"{c['htf']:.1f}", c["stop"], c["trail"],
        f"{c['fr']:.2f}", f"{c['ta']:.2f}", f"{c['td']:.2f}",
        f"{c['tgt']:.2f}", str(c["mp"])]
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def rng_cfg(r):
    """Sample a random config from the search space."""
    w = np.round(r.uniform(0, 3, K), 2)
    w[r.random(K) < 0.25] = 0.0                      # sometimes mute a factor
    return {"w": w, "thr": round(r.uniform(0.5, 3.0), 2),
            "htf": round(r.uniform(-1.0, 2.5), 2),
            "stop": r.choice(STOPS), "trail": r.choice(TRAILS),
            "fr": round(r.uniform(0.5, 0.9), 3),
            "ta": round(float(r.choice([0, 0, 0.04, 0.06])), 3),
            "td": round(r.uniform(0.03, 0.15), 3),
            "tgt": float(r.choice([0, 0, 0, 0.1, 0.2])),
            "mp": int(r.choice([3, 5, 8]))}


def mutate(c, r):
    """A small random perturbation of a config (for the evolutionary step)."""
    n = {**c, "w": c["w"].copy()}
    for i in range(K):                               # jiggle ~2 weights
        if r.random() < 0.25:
            n["w"][i] = round(max(0.0, c["w"][i] + r.normal(0, 0.8)), 2)
    if r.random() < 0.4:
        n["thr"] = round(max(0.1, c["thr"] + r.normal(0, 0.4)), 2)
    if r.random() < 0.4:
        n["htf"] = round(c["htf"] + r.normal(0, 0.4), 2)
    if r.random() < 0.2:
        n["stop"] = r.choice(STOPS)
    if r.random() < 0.2:
        n["trail"] = r.choice(TRAILS)
    if r.random() < 0.3:
        n["td"] = round(min(0.2, max(0.02, c["td"] + r.normal(0, 0.03))), 3)
    return n


def main():
    args = sys.argv[1:]
    universe = arg(args, "universe", "crypto")
    gate = arg(args, "gate", "minervini@1d")
    market = arg(args, "market", "none")
    months = int(arg(args, "months", "0"))
    wf_folds = int(arg(args, "wf-folds", "6"))
    holdout = float(arg(args, "holdout", "0.2"))     # final unseen slice
    iters = int(arg(args, "iters", "400"))           # random seeds
    gens = int(arg(args, "gens", "6"))               # evolutionary generations
    elite = int(arg(args, "elite", "20"))
    cost_side = float(arg(args, "cost-bps", "10")) / 10000.0 / 2
    capital = float(arg(args, "capital", "100000"))
    seed = int(arg(args, "seed", "0"))
    r = np.random.default_rng(seed)

    with open(arg(args, "file", "bot_strategies.json"), encoding="utf-8") as f:
        import json
        strategies = json.load(f)

    t0 = time.time()
    (A, V, ENT, SIDX, EXT, syms, grid, st_stop, st_hold, st_tgt), cached = \
        prepare_grid_cached(strategies, "15m", gate, market, months,
                            universe=universe)
    # HOLDOUT: reserve the last `holdout` fraction of history that the search
    # NEVER sees; walk-forward folds live entirely in the earlier "train"
    # region, and the final winners are scored once on the holdout. This stops
    # the search from overfitting the folds it optimizes on.
    T = len(grid)
    tr_end = int(T * (1 - holdout))
    b = [int(tr_end * i / wf_folds) for i in range(wf_folds + 1)]
    folds = [slice(b[i], b[i + 1]) for i in range(wf_folds)]
    hold_sl = slice(tr_end, None)
    print(f"grid {'cached' if cached else 'built'} in {time.time()-t0:.0f}s: "
          f"{len(syms)} syms x {T} bars, universe={universe}; search "
          f"{pd.Timestamp(grid[0]).date()}->{pd.Timestamp(grid[tr_end-1]).date()}"
          f", HELD-OUT {pd.Timestamp(grid[tr_end]).date()}->"
          f"{pd.Timestamp(grid[-1]).date()}", flush=True)
    hold0 = np.zeros_like(st_hold)                   # no time-clock exit, ever
    ext_arr = np.asarray(FIB_EXT)

    # dedup: reuse scores of configs already tested on THIS grid in prior runs
    gsig = f"15m|{gate}|{market}|{months}mo"
    prior = load_results()
    seen = {}
    if len(prior) and {"cfg_sig", "grid_sig", "wf_fit"} <= set(prior.columns):
        for _, row in prior[prior["grid_sig"] == gsig].iterrows():
            pos = int(str(row.get("wf_pos", "0/0")).split("/")[0])
            seen[row["cfg_sig"]] = (row["wf_exc"], row["wf_min"], pos)
    reused = [0]

    def fitness(mean, mn):
        return round(mean + 0.5 * min(0.0, mn), 1)

    def excess_fold(sl, c):
        """One config on one fold -> excess over buy-&-hold of traded names."""
        gx = grid[sl]
        yrs = max((pd.Timestamp(gx[-1]) - pd.Timestamp(gx[0])).days / 365.25,
                  1e-9)
        tgt = np.full_like(st_tgt, c["tgt"]) if c["tgt"] > 0 else st_tgt
        res = portsim.simulate(
            A["O"][sl], A["H"][sl], A["L"][sl], A["C"][sl], V[sl], ENT[sl],
            A["SCORE"][sl], SIDX[sl], EXT[sl], A["ATR"][sl], A["SW"][sl],
            st_stop, hold0, tgt, stop_mode=STOP_MODES.get(c["stop"], 0),
            swing_buf=0.005, trail_act=c["ta"], trail_dist=c["td"],
            cost_side=cost_side, max_pos=c["mp"], init_cash=capital,
            p1=A["P1"][sl], p2=A["P2"][sl], p3=A["P3"][sl], fib_stop_ratio=c["fr"],
            trail_mode=TRAIL_MODES.get(c["trail"], 0), fib_ext=ext_arr,
            gate=A["GATE"][sl], factors=A["FACTORS"][sl], weights=c["w"],
            conf_entry=1, conf_threshold=c["thr"], htf_start=HTF_START,
            htf_screen=1, htf_threshold=c["htf"])
        eq = pd.Series(res["equity"])
        cagr = (eq.iloc[-1] / capital) ** (1 / yrs) - 1
        rr = [A["C"][sl][:, s2][V[sl][:, s2]] for s2 in np.unique(res["sym"])] \
            if len(res["sym"]) else []
        rr = [v[-1] / v[0] for v in rr if len(v) >= 2 and v[0] > 0]
        hold = (float(np.mean(rr)) ** (1 / yrs) - 1) if rr else 0.0
        return (cagr - hold) * 100

    def score(c):
        c["_sig"] = sig = cfg_sig(c)
        if sig in seen:                              # already tested -> reuse
            c["_exc"], c["_min"], c["_pos"] = seen[sig]
            reused[0] += 1
        else:
            exc = [excess_fold(sl, c) for sl in folds]
            c["_exc"] = round(float(np.mean(exc)), 1)
            c["_min"] = round(float(np.min(exc)), 1)
            c["_pos"] = int(sum(e > 0 for e in exc))
            seen[sig] = (c["_exc"], c["_min"], c["_pos"])
        # robust fitness recomputed every time so a formula change never leaves
        # stale scores in the population (see fitness()).
        c["_fit"] = fitness(c["_exc"], c["_min"])
        return c

    pop, n = [], 0
    t1 = time.time()
    for _ in range(iters):                           # random seeding
        pop.append(score(rng_cfg(r))); n += 1
    for g in range(gens):                            # evolution
        pop.sort(key=lambda c: c["_fit"], reverse=True)
        pop = pop[:elite]
        best = pop[0]
        print(f"  gen {g}: best wf_exc={best['_exc']} (min={best['_min']}, "
              f"pos={best['_pos']}/{wf_folds}, stop={best['stop']}, "
              f"trail={best['trail']})", flush=True)
        children = [score(mutate(pop[int(r.integers(len(pop)))], r))
                    for _ in range(iters)]
        n += len(children)
        pop += children
    pop.sort(key=lambda c: c["_fit"], reverse=True)

    # HELD-OUT validation: score the top configs once on the slice the search
    # never saw. This is the honest number -- if it still beats buy-and-hold
    # here, the edge is real; if it craters, the search overfit its folds.
    for c in pop[:elite]:
        c["_ho"] = round(excess_fold(hold_sl, c), 1)

    # persist every evaluated config to the shared store (with its signature
    # so future runs dedup instead of repeating it)
    rows = []
    for c in pop:
        row = {f"w_{nm}": wv for nm, wv in zip(FACTOR_NAMES, c["w"])}
        row.update({"thr": c["thr"], "htf_thr": c["htf"], "stop": c["stop"],
                    "trail": c["trail"], "fibr": c["fr"], "ta": c["ta"],
                    "td": c["td"], "tgt": c["tgt"], "mp": c["mp"],
                    "cfg_sig": c["_sig"], "wf_fit": c["_fit"],
                    "wf_exc": c["_exc"], "wf_min": c["_min"],
                    "wf_pos": f"{c['_pos']}/{wf_folds}",
                    "holdout_exc": c.get("_ho")})
        rows.append(row)
    df = pd.DataFrame(rows)
    meta = {"interval": "15m", "gate": gate, "market": market, "months": months,
            "wf": True, "oos": False}
    save_results(df, meta, {"search": "agent_search", "universe": universe,
                            "iters": iters, "gens": gens, "holdout": holdout})

    print(f"\nevaluated {n} configs in {time.time()-t1:.0f}s "
          f"({reused[0]} reused from prior runs; persisted to {RESULTS_DIR}/).")
    print("Top 10 by ROBUST train fitness -- with the HELD-OUT excess as the "
          "honest check (holdout% > 0 = still beat buy&hold on unseen data):")
    show = ["wf_fit", "wf_exc", "wf_min", "wf_pos", "holdout_exc", "stop",
            "trail", "td", "thr", "htf_thr", "mp"]
    pd.set_option("display.width", 240)
    print(df[show].head(10).to_string(index=False))
    top = df.iloc[0]
    print(f"\nbest-by-train-fitness held out at {top['holdout_exc']}% excess "
          f"over buy&hold on data it never saw.")


if __name__ == "__main__":
    main()
