#!/usr/bin/env python3
"""Agent-scale search over the confluence weight / exit space, optimizing the
one metric that can't be gamed: WALK-FORWARD EXCESS over buy-and-hold, with the
best TIME-TESTED TEXTBOOK strategy as a second bar (goal #2).

Loads the (cached) grid once, seeds with random configs, then evolves the best
(elitism + mutation). Every evaluated config is persisted to sweep_results/ so
agents -- and reruns -- build on what's been tested instead of repeating it.

Nothing here is hardcoded: every bound, default and sim literal comes from
search_space.json (--space=FILE swaps it, --set=a.b=v bends one knob).

    python agent_search.py --universe=crypto --wf-folds=6 --iters=400 --gens=6
    python agent_search.py --jobs=8            # fan configs across processes

PARALLELISM: configs within a generation are independent, so they fan out across
processes; the generations themselves stay sequential (you need the elite before
you can mutate it). Each worker holds its OWN copy of the grid -- loaded from the
disk cache, not pickled through the parent -- so --jobs trades RAM for speed.
Watch memory on full-history grids.

MEASURED (12mo crypto grid, 16-core box, 2026-07-15) -- --jobs is NOT free:
    workload            serial      --jobs=8
    450 configs        148/sec      115/sec   <- parallel LOSES
    3200 configs       192/sec      455/sec   <- 2.4x
Windows SPAWNS rather than forks, so every worker re-imports numba/pandas,
re-JITs portsim and reloads the grid: ~5s of fixed startup. Below roughly 1300
configs that never pays back. Default runs (iters=400 x gens=6 = 2800) are above
the line; short probe runs are not. Measure before assuming more cores is more
speed -- on this workload 16 jobs was slower than 8.

ASCII output.
"""

import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

# --space= must reach the environment BEFORE portfolio_multi is imported:
# FACTOR_NAMES is built from the spec's factor block at import time, so a flag
# parsed later in main() would arrive too late to change the factor stack.
from search_space import bootstrap_space
bootstrap_space(sys.argv[1:])

from portfolio_multi import (FACTOR_NAMES, HTF_START,      # noqa: E402
                             MAINSTREAM_COLUMNS, STOP_MODES, TRAIL_MODES,
                             prepare_grid_cached)
from search_space import (DEFAULT_SPACE, load_space,       # noqa: E402
                          mutate_cfg, parse_set_args, sample_cfg, space_sig)
from sweep import RESULTS_DIR, grid_sig_of, load_results, save_results
from test_strategy import arg
from weisswave import portsim

K = len(FACTOR_NAMES)


def cfg_sig(c):
    """A canonical signature of a config (weights + params, coarsely rounded)
    so near-identical configs dedup. Lets a search skip anything already tested
    -- in this run or a prior one on the same data -- instead of repeating it."""
    parts = [f"{w:.1f}" for w in c["w"]] + [
        f"{c['thr']:.1f}", f"{c['htf']:.1f}", c["stop"], c["trail"],
        f"{c['fr']:.2f}", f"{c['ta']:.2f}", f"{c['td']:.2f}",
        f"{c['tgt']:.2f}", str(c["mp"])]
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def parse_job(args):
    """CLI + spec -> every scalar the engine needs. Flags win over the spec, so
    orchestrate/agents can vary a job without writing a file."""
    sp = load_space(arg(args, "space", DEFAULT_SPACE), parse_set_args(args))
    G, SIM, VAL = sp["grid"], sp["sim"], sp["validation"]
    j = {
        "sp": sp,
        "universe": arg(args, "universe", G["universe"]),
        "gate": arg(args, "gate", G["gate"]),
        "gate_mode": arg(args, "gate-mode", G["gate_mode"]),
        "fib_anchor": arg(args, "fib-anchor", G["fib_anchor"]),
        "interval": arg(args, "interval", G["interval"]),
        "market": arg(args, "market", G["market"]),
        "months": int(arg(args, "months", str(G["months"]))),
        "wf_folds": int(arg(args, "wf-folds", str(VAL["wf_folds"]))),
        "holdout": float(arg(args, "holdout", str(VAL["holdout"]))),
        "iters": int(arg(args, "iters", "400")),
        "gens": int(arg(args, "gens", "6")),
        "elite": int(arg(args, "elite", "20")),
        "cost_side": float(arg(args, "cost-bps",
                                str(SIM["cost_bps"]))) / 10000.0 / 2,
        "capital": float(arg(args, "capital", str(SIM["capital"]))),
        "seed": int(arg(args, "seed", "0")),
        "jobs": int(arg(args, "jobs", "1")),
        "file": arg(args, "file", "bot_strategies.json"),
    }
    j["min_trades"] = int(sp["fitness"].get("min_trades", 1))
    return j


class Engine:
    """The grid + the scoring of one config against it. Built once per PROCESS
    (the parent, and each worker via the pool initializer) -- never pickled."""

    def __init__(self, job, quiet=False):
        self.j = job
        sp = job["sp"]
        self.SIM, self.BM, self.FIT = sp["sim"], sp["benchmark"], sp["fitness"]
        with open(job["file"], encoding="utf-8") as f:
            strategies = json.load(f)
        t0 = time.time()
        (self.A, self.V, self.ENT, self.SIDX, self.EXT, self.syms, self.grid,
         self.st_stop, self.st_hold, self.st_tgt), cached = \
            prepare_grid_cached(strategies, job["interval"], job["gate"],
                                job["market"], job["months"],
                                universe=job["universe"],
                                gate_mode=job["gate_mode"],
                                fib_anchor=job["fib_anchor"])
        # HOLDOUT: reserve the last `holdout` fraction of history the search
        # NEVER sees; the walk-forward folds live entirely in the earlier train
        # region and the winners are scored on the holdout once.
        T = len(self.grid)
        self.tr_end = int(T * (1 - job["holdout"]))
        wf = job["wf_folds"]
        b = [int(self.tr_end * i / wf) for i in range(wf + 1)]
        self.folds = [slice(b[i], b[i + 1]) for i in range(wf)]
        self.hold_sl = slice(self.tr_end, None)
        # time-clock exit: a knob from the spec, not a constant. 0 = off.
        self.hold_bars = np.full_like(self.st_hold,
                                      int(self.SIM.get("hold_bars", 0)))
        self.ext_arr = np.asarray(self.SIM["fib_ext"])
        self._ms_cache = {}
        if not quiet:
            g, te = self.grid, self.tr_end
            print(f"grid {'cached' if cached else 'built'} in "
                  f"{time.time()-t0:.0f}s: {len(self.syms)} syms x {T} bars, "
                  f"{job['interval']} {job['universe']} "
                  f"gate={job['gate']}/{job['gate_mode']} "
                  f"fib@{job['fib_anchor']}; search "
                  f"{pd.Timestamp(g[0]).date()}->{pd.Timestamp(g[te-1]).date()}"
                  f", HELD-OUT {pd.Timestamp(g[te]).date()}->"
                  f"{pd.Timestamp(g[-1]).date()}", flush=True)

    def _cagr(self, res, yrs):
        return (pd.Series(res["equity"]).iloc[-1]
                / self.j["capital"]) ** (1 / yrs) - 1

    def _yrs(self, sl):
        gx = self.grid[sl]
        return max((pd.Timestamp(gx[-1]) - pd.Timestamp(gx[0])).days / 365.25,
                   1e-9)

    def mainstream_cagr(self, sl, yrs):
        """GOAL #2: run each textbook strategy through the SAME engine, stops,
        costs and slots -- ungated, as it is actually taught. {name: CAGR}.

        NOTE the exits here are an UNAPPROVED PLACEHOLDER (see search_space.json
        benchmark._UNAPPROVED_PLACEHOLDER): a hard stop with no trail is not
        Paul's rule, so these CAGRs are partly an artifact of an arbitrary exit.
        Do not present them as findings until he specifies the exits."""
        A, BM = self.A, self.BM
        out = {}
        for nm in BM["mainstream"]:
            if nm not in MAINSTREAM_COLUMNS:
                continue
            k = MAINSTREAM_COLUMNS.index(nm)
            res = portsim.simulate(
                A["O"][sl], A["H"][sl], A["L"][sl], A["C"][sl], self.V[sl],
                A["MS"][sl][:, :, k] > 0.5, A["SCORE"][sl], self.SIDX[sl],
                self.EXT[sl], A["ATR"][sl], A["SW"][sl],
                np.full_like(self.st_stop, BM["stop_pct"]), self.hold_bars,
                np.zeros_like(self.st_tgt),
                stop_mode=STOP_MODES.get(BM["stop_mode"], 0),
                swing_buf=self.SIM["swing_buf"], trail_act=0.0, trail_dist=0.0,
                cost_side=self.j["cost_side"], max_pos=int(BM["max_pos"]),
                init_cash=self.j["capital"], p1=A["P1"][sl], p2=A["P2"][sl],
                p3=A["P3"][sl], fib_stop_ratio=0.7, trail_mode=0,
                fib_ext=self.ext_arr, gate=None, factors=None, weights=None,
                conf_entry=0, htf_start=-1, htf_screen=0)
            out[nm] = self._cagr(res, yrs) if len(res["sym"]) else 0.0
        return out

    def best_mainstream(self, sl, yrs):
        """Strongest textbook strategy on this fold. Cached per slice: the
        benchmark is a property of the DATA, not of the config being scored."""
        key = (sl.start, sl.stop)
        if key not in self._ms_cache:
            d = self.mainstream_cagr(sl, yrs)
            self._ms_cache[key] = (max(d.values()) if d else 0.0, d)
        return self._ms_cache[key][0]

    def excess_fold(self, sl, c):
        """One config on one fold -> (excess vs buy-&-hold of traded names,
        excess vs the best textbook strategy, trade count).

        The trade count matters: a config that never trades produces excess 0.0
        (flat equity AND an empty traded-names baseline), which would otherwise
        rank ABOVE every config that traded and lost. Doing nothing is not a
        neutral result -- it is no result."""
        A, V, SIM = self.A, self.V, self.SIM
        yrs = self._yrs(sl)
        tgt = (np.full_like(self.st_tgt, c["tgt"]) if c["tgt"] > 0
               else self.st_tgt)
        res = portsim.simulate(
            A["O"][sl], A["H"][sl], A["L"][sl], A["C"][sl], V[sl],
            self.ENT[sl], A["SCORE"][sl], self.SIDX[sl], self.EXT[sl],
            A["ATR"][sl], A["SW"][sl], self.st_stop, self.hold_bars, tgt,
            stop_mode=STOP_MODES.get(c["stop"], 0), swing_buf=SIM["swing_buf"],
            trail_act=c["ta"], trail_dist=c["td"], cost_side=self.j["cost_side"],
            max_pos=c["mp"], init_cash=self.j["capital"], p1=A["P1"][sl],
            p2=A["P2"][sl], p3=A["P3"][sl], fib_stop_ratio=c["fr"],
            trail_mode=TRAIL_MODES.get(c["trail"], 0), fib_ext=self.ext_arr,
            gate=A["GATE"][sl], factors=A["FACTORS"][sl], weights=c["w"],
            conf_entry=SIM["conf_entry"], conf_threshold=c["thr"],
            htf_start=HTF_START, htf_screen=SIM["htf_screen"],
            htf_threshold=c["htf"])
        ntr = len(res["sym"])
        cagr = self._cagr(res, yrs)
        rr = [A["C"][sl][:, s2][V[sl][:, s2]]
              for s2 in np.unique(res["sym"])] if ntr else []
        rr = [v[-1] / v[0] for v in rr if len(v) >= 2 and v[0] > 0]
        hold = (float(np.mean(rr)) ** (1 / yrs) - 1) if rr else 0.0
        # GOAL #2: two bars, not one -- buy-and-hold AND the best textbook
        # strategy over the same fold.
        return ((cagr - hold) * 100,
                (cagr - self.best_mainstream(sl, yrs)) * 100, ntr)

    def score_raw(self, c, folds=None):
        """Score one config across `folds` (default: the whole-history training
        folds). Returns a PLAIN DICT so it can cross a process boundary; no
        dedup here (the parent owns that).

        The folds are a PARAMETER, not a property of the engine, so nested
        validation can re-run a search inside an arbitrary training window
        without rebuilding the grid."""
        out = [self.excess_fold(sl, c) for sl in (folds or self.folds)]
        exc = [e for e, _, _ in out]
        return {"sig": cfg_sig(c),
                "exc": round(float(np.mean(exc)), 1),
                "min": round(float(np.min(exc)), 1),
                "pos": int(sum(e > 0 for e in exc)),
                "ms": round(float(np.mean([m for _, m, _ in out])), 1),
                "ntr": int(sum(nt for _, _, nt in out))}


# ---- process pool: each worker builds its OWN Engine, once -------------------
_W = {}


def _worker_init(args):
    _W["eng"] = Engine(parse_job(args), quiet=True)


def _worker_score(task):
    c, folds = task
    return c, _W["eng"].score_raw(c, folds)


def evolve(eng, sp, folds, seed, iters, gens, elite, min_trades,
           pool=None, seen=None, quiet=False, wf_label=""):
    """Seed random configs, then evolve the elite against `folds`. Returns the
    population sorted best-first.

    THE FOLDS ARE AN ARGUMENT. That is what makes nested validation possible:
    the same search can be re-run inside an arbitrary training window, having
    never seen the slice it will later be judged on."""
    r = np.random.default_rng(seed)
    seen = {} if seen is None else seen
    reused = [0]

    def apply(c, s):
        c["_sig"] = s.get("sig") or cfg_sig(c)
        c["_exc"], c["_min"] = s["exc"], s["min"]
        c["_pos"], c["_ntr"], c["_ms"] = s["pos"], s["ntr"], s.get("ms", np.nan)
        # A config that never traded is UNFIT, not neutral -- otherwise its 0.0
        # outranks every config that actually took risk and lost.
        c["_fit"] = (-np.inf if 0 <= c["_ntr"] < min_trades
                     else round(c["_exc"] + sp["fitness"]["worst_fold_penalty"]
                                * min(0.0, c["_min"]), 1))
        return c

    def score_batch(cfgs):
        todo, done = [], []
        for c in cfgs:
            s = seen.get(cfg_sig(c))
            if s is not None:
                reused[0] += 1
                done.append(apply(c, {**s, "sig": cfg_sig(c)}))
            else:
                todo.append(c)
        if todo:
            res = (pool.map(_worker_score, [(c, folds) for c in todo],
                            chunksize=4) if pool
                   else [(c, eng.score_raw(c, folds)) for c in todo])
            for c, s in res:
                seen[s["sig"]] = s
                done.append(apply(c, s))
        return done

    pop = score_batch([sample_cfg(r, sp, K, HTF_START) for _ in range(iters)])
    n = len(pop)
    for g in range(gens):                            # evolution (sequential)
        pop.sort(key=lambda c: c["_fit"], reverse=True)
        pop = pop[:elite]
        best = pop[0]
        if not quiet:
            dead = sum(1 for c in pop if 0 <= c["_ntr"] < min_trades)
            print(f"  {wf_label}gen {g}: best wf_exc={best['_exc']} "
                  f"(min={best['_min']}, pos={best['_pos']}/{len(folds)}, "
                  f"trades={best['_ntr']}, stop={best['stop']}, "
                  f"trail={best['trail']}); {dead}/{len(pop)} elite unfit",
                  flush=True)
        kids = [mutate_cfg(pop[int(r.integers(len(pop)))], r, sp, K)
                for _ in range(iters)]
        pop += score_batch(kids)
        n += len(kids)
    pop.sort(key=lambda c: c["_fit"], reverse=True)
    return pop, n, reused[0]


def main():
    args = sys.argv[1:]
    job = parse_job(args)
    sp = job["sp"]
    FIT, VAL = sp["fitness"], sp["validation"]
    wf_folds, elite = job["wf_folds"], job["elite"]
    iters, gens, min_trades = job["iters"], job["gens"], job["min_trades"]
    r = np.random.default_rng(job["seed"])

    eng = Engine(job)                       # parent builds/caches the grid first

    # dedup: reuse scores of configs already tested on THIS grid in prior runs.
    # gsig MUST name every input that changes what a score MEANS -- universe and
    # gate_mode were missing, so a crypto score could be silently reused for a
    # stocks config with the same gate/market/months.
    gsig = grid_sig_of(job["interval"], job["gate"], job["market"],
                       job["months"], job["universe"], job["gate_mode"],
                       job["fib_anchor"])
    prior = load_results()
    seen = {}
    if len(prior) and {"cfg_sig", "grid_sig", "wf_fit"} <= set(prior.columns):
        # The store is SCHEMA-UNIONED across every tool that writes to it, so a
        # matching grid_sig does NOT mean a matching row shape: validate.py
        # writes per-FOLD rows (no cfg_sig, no wf_pos) that land right next to
        # agent_search's per-CONFIG rows. Take only rows that are actually a
        # scored config, and never assume a field parses.
        sub = prior[prior["grid_sig"] == gsig]
        sub = sub.dropna(subset=[c for c in ("cfg_sig", "wf_exc", "wf_pos")
                                 if c in sub.columns])
        for _, row in sub.iterrows():
            try:
                pos = int(str(row["wf_pos"]).split("/")[0])
            except (ValueError, TypeError, KeyError):
                continue                     # not a per-config row; skip it
            # wf_trades is absent on rows written before it existed; -1 means
            # "unknown", which must not be mistaken for "never traded".
            ntr = row.get("wf_trades", -1)
            ntr = -1 if pd.isna(ntr) else int(ntr)
            seen[row["cfg_sig"]] = {"exc": row["wf_exc"], "min": row["wf_min"],
                                    "pos": pos, "ntr": ntr,
                                    "ms": row.get("wf_ms_exc", np.nan)}
    jobs = max(1, job["jobs"])
    pool = None
    if jobs > 1:
        pool = mp.Pool(jobs, initializer=_worker_init, initargs=(args,))
        print(f"fanning configs across {jobs} worker processes "
              f"(each holds its own grid copy)", flush=True)

    t1 = time.time()
    pop, n, n_reused = evolve(eng, sp, eng.folds, job["seed"], iters, gens,
                              elite, min_trades, pool=pool, seen=seen)
    reused = [n_reused]
    if pool:
        pool.close()
        pool.join()

    # HELD-OUT validation: score the top configs once on the slice the search
    # never saw. This is the honest number -- if it still beats buy-and-hold
    # here the edge is real; if it craters, the search overfit its folds.
    for c in pop[:elite]:
        ho, ho_ms, ho_n = eng.excess_fold(eng.hold_sl, c)
        c["_ho"] = round(ho, 1) if ho_n >= 1 else None   # no trades -> no claim
        c["_ho_ms"] = round(ho_ms, 1) if ho_n >= 1 else None
        c["_ho_n"] = ho_n

    rows = []
    for c in pop:
        row = {f"w_{nm}": wv for nm, wv in zip(FACTOR_NAMES, c["w"])}
        fit = c["_fit"]
        row.update({"thr": c["thr"], "htf_thr": c["htf"], "stop": c["stop"],
                    "trail": c["trail"], "fibr": c["fr"], "ta": c["ta"],
                    "td": c["td"], "tgt": c["tgt"], "mp": c["mp"],
                    "cfg_sig": c["_sig"],
                    "wf_fit": None if fit == -np.inf else fit,
                    "wf_exc": c["_exc"], "wf_min": c["_min"],
                    "wf_pos": f"{c['_pos']}/{wf_folds}",
                    "wf_trades": c["_ntr"],
                    "unfit": bool(0 <= c["_ntr"] < min_trades),
                    "wf_ms_exc": c["_ms"],
                    "holdout_exc": c.get("_ho"),
                    "holdout_ms_exc": c.get("_ho_ms"),
                    "holdout_trades": c.get("_ho_n")})
        rows.append(row)
    df = pd.DataFrame(rows)
    meta = {"interval": job["interval"], "gate": job["gate"],
            "market": job["market"], "months": job["months"],
            "universe": job["universe"], "gate_mode": job["gate_mode"],
            "fib_anchor": job["fib_anchor"], "wf": True, "oos": False}
    save_results(df, meta, {"search": "agent_search",
                            "universe": job["universe"], "iters": iters,
                            "gens": gens, "holdout": job["holdout"],
                            "gate_mode": job["gate_mode"],
                            "interval": job["interval"], "jobs": jobs,
                            "fib_anchor": job["fib_anchor"],
                            "space_sig": space_sig(sp)})

    dt = time.time() - t1
    n_unfit = int(df["unfit"].sum())
    print(f"\nevaluated {n} configs in {dt:.0f}s ({n/max(dt,1e-9):.1f}/sec; "
          f"{reused[0]} reused; {n_unfit} never traded -> unfit; persisted to "
          f"{RESULTS_DIR}/).")
    print("Top 10 by ROBUST train fitness -- with the HELD-OUT excess as the "
          "honest check (holdout% > 0 = still beat buy&hold on unseen data):")
    show = ["wf_fit", "wf_exc", "wf_ms_exc", "wf_min", "wf_pos", "wf_trades",
            "holdout_exc", "holdout_ms_exc", "stop", "trail", "td", "thr",
            "htf_thr", "mp"]
    pd.set_option("display.width", 260)
    print(df[show].head(10).to_string(index=False))
    top = df.iloc[0]
    if top["unfit"]:
        print("\nNo config traded: the space as specified cannot fire on this "
              "data. Widen search_space.json (thresholds/weights) -- do NOT "
              "read this as 'matched buy&hold'.")
    else:
        print(f"\nbest-by-train-fitness held out at {top['holdout_exc']}% "
              f"excess over buy&hold on data it never saw "
              f"({top['holdout_trades']} trades).")


if __name__ == "__main__":
    mp.freeze_support()          # Windows spawns, so this must be guarded
    main()
