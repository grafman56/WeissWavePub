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

import sys
import time

import numpy as np
import pandas as pd

from portfolio_multi import (FACTOR_NAMES, FIB_ENTRY_MODES, HTF_START,
                             STOP_MODES, TRAIL_MODES, prepare_grid_cached)
from sweep import RESULTS_DIR, save_results
from test_strategy import arg
from weisswave import portsim

K = len(FACTOR_NAMES)
STOPS, TRAILS = ["fib", "swing", "atr", "pct"], ["pct", "structure", "fib"]
FIB_EXT = (1.0, 1.272, 1.618, 2.0)


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
    print(f"grid {'cached' if cached else 'built'} in {time.time()-t0:.0f}s: "
          f"{len(syms)} syms x {len(grid)} bars, universe={universe}", flush=True)
    b = [int(len(grid) * i / wf_folds) for i in range(wf_folds + 1)]
    folds = [(slice(b[i], b[i + 1])) for i in range(wf_folds)]
    hold0 = np.zeros_like(st_hold)                   # no time-clock exit, ever
    ext_arr = np.asarray(FIB_EXT)

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
        exc = [excess_fold(sl, c) for sl in folds]
        mean, mn = float(np.mean(exc)), float(np.min(exc))
        c["_exc"] = round(mean, 1)
        c["_min"] = round(mn, 1)
        c["_pos"] = int(sum(e > 0 for e in exc))
        # ROBUST fitness: reward mean excess, heavily penalize any fold that
        # loses badly to buy-and-hold -- a config must beat holding across
        # regimes, not average out a blow-up with a lucky bull fold.
        c["_fit"] = round(mean + 1.5 * min(0.0, mn), 1)
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

    # persist every evaluated config to the shared store
    rows = []
    for c in pop:
        row = {f"w_{nm}": wv for nm, wv in zip(FACTOR_NAMES, c["w"])}
        row.update({"thr": c["thr"], "htf_thr": c["htf"], "stop": c["stop"],
                    "trail": c["trail"], "fibr": c["fr"], "ta": c["ta"],
                    "td": c["td"], "tgt": c["tgt"], "mp": c["mp"],
                    "wf_fit": c["_fit"], "wf_exc": c["_exc"],
                    "wf_min": c["_min"], "wf_pos": f"{c['_pos']}/{wf_folds}"})
        rows.append(row)
    df = pd.DataFrame(rows)
    meta = {"interval": "15m", "gate": gate, "market": market, "months": months,
            "wf": True, "oos": False}
    save_results(df, meta, {"search": "agent_search", "universe": universe,
                            "iters": iters, "gens": gens})

    print(f"\nevaluated {n} configs in {time.time()-t1:.0f}s "
          f"(persisted to {RESULTS_DIR}/). Top 10 by ROBUST fitness "
          f"(mean excess penalized for bad worst-fold):")
    show = ["wf_fit", "wf_exc", "wf_min", "wf_pos", "stop", "trail", "td",
            "thr", "htf_thr", "mp"] + [f"w_{n}" for n in FACTOR_NAMES]
    pd.set_option("display.width", 240)
    print(df[show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
