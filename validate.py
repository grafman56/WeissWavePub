#!/usr/bin/env python3
"""NESTED WALK-FORWARD VALIDATION -- does the SEARCH generalize?

This does NOT validate a config you hand it. That is the trap it exists to
close: a config found by searching all of history has already seen every
window you would then "validate" it on, so the result looks great and means
nothing. Contamination with extra steps.

Instead it re-runs the WHOLE SEARCH inside each fold's training window, takes
that fold's own winner, and scores it once on the next window -- data that
search never touched. Each fold produces a DIFFERENT winner. That is the point.

    sliding (default)          anchored
    [train][test]              [train][test]
       [train][test]           [--train--][test]
          [train][test]        [---train---][test]

WHAT IT REPORTS: a property of the METHOD, not of any config -- "across 4
unseen windows the search's winner beat buy-and-hold in 3, mean excess +X%".
You cannot trade the aggregate. The workflow is:

    1. validate.py            -> does the search procedure generalize?
    2. if yes: agent_search   -> on all data, for the config you actually trade
    3. trust that config exactly as much as step 1 earned

WHY THIS EXISTS. The fixed holdout was one slice, scored against every config
ever run, with survivors PICKED by their score on it -- which makes it a second
training set, not a holdout. Worse, it is a bear stretch, and the trend filters
correctly stand bull strategies down there: of 252 configs scored on it, 75
made ZERO trades and the median made 7. A bull strategy abstaining in a bear is
not evidence about the strategy; it is evidence the screen works. Rolling gives
every config unseen windows in regimes where it actually trades.

    python validate.py --universe=crypto --outer-folds=4 --jobs=8
    python validate.py --set=validation.mode=anchored

ASCII output.
"""

import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

from search_space import bootstrap_space      # must precede the imports below
bootstrap_space(sys.argv[1:])

from agent_search import (Engine, _worker_init,             # noqa: E402
                          _worker_score, cap_jobs, evolve, parse_job)
from portfolio_multi import FACTOR_NAMES
from sweep import RESULTS_DIR, save_results
from search_space import space_sig
from test_strategy import arg


def outer_folds(T, k, mode, train_frac):
    """(train_slice, test_slice) per outer fold.

    sliding  : train is a FIXED-WIDTH window ending right before its test --
               regime-adaptive, discards stale history.
    anchored : train always starts at bar 0 and grows -- uses all history, and
               assumes the distant past still teaches you something.
    Either way the test window sits strictly AFTER its training window, so no
    fold is ever judged on data its own search could have seen."""
    W = int(T * train_frac)
    step = max(1, (T - W) // k)
    out = []
    for i in range(k):
        tr_end = W + i * step
        te_end = T if i == k - 1 else tr_end + step
        if tr_end >= T or te_end <= tr_end:
            break
        train = (slice(max(0, tr_end - W), tr_end) if mode == "sliding"
                 else slice(0, tr_end))
        out.append((train, slice(tr_end, te_end)))
    return out


def inner_folds(train, n):
    """Split a training window into n contiguous folds for the fitness score."""
    a, b = train.start, train.stop
    edges = [a + int((b - a) * i / n) for i in range(n + 1)]
    return [slice(edges[i], edges[i + 1]) for i in range(n)]


def main():
    args = sys.argv[1:]
    job = parse_job(args)
    sp = job["sp"]
    VAL = sp["validation"]
    k = int(arg(args, "outer-folds", str(VAL.get("outer_folds", 4))))
    mode = arg(args, "mode", VAL.get("mode", "sliding"))
    train_frac = float(arg(args, "train-frac",
                           str(VAL.get("train_frac", 0.5))))
    iters, gens = job["iters"], job["gens"]
    elite, min_trades = job["elite"], job["min_trades"]

    eng = Engine(job)
    T = len(eng.grid)
    ofolds = outer_folds(T, k, mode, train_frac)
    if not ofolds:
        print("not enough history for that many outer folds")
        return

    # Every worker holds its OWN copy of the grid, so RAM is jobs x grid_bytes.
    # agent_search caps this and validate did not -- while this file's own
    # docstring recommends --jobs=8. Harmless on a 12-symbol crypto grid
    # (~0.1GB); on a ~500-symbol stocks grid that is ~37GB and the machine does
    # not refuse, it thrashes and dies. Same cap, one implementation.
    jobs = cap_jobs(max(1, job["jobs"]), eng)
    pool = (mp.Pool(jobs, initializer=_worker_init, initargs=(args,))
            if jobs > 1 else None)
    print(f"NESTED walk-forward: {len(ofolds)} outer folds, mode={mode}, "
          f"train_frac={train_frac:.0%}, {iters}x{gens} search per fold"
          f"{f', {jobs} procs' if pool else ''}", flush=True)
    print("Each fold trains its OWN search and is scored on data that search "
          "never saw.\n", flush=True)

    t0 = time.time()
    rows = []
    for i, (train, test) in enumerate(ofolds):
        g = eng.grid
        d = lambda s: pd.Timestamp(g[s]).date()
        print(f"[fold {i+1}/{len(ofolds)}] train {d(train.start)}->"
              f"{d(train.stop-1)} | TEST {d(test.start)}->{d(test.stop-1)}",
              flush=True)
        ifolds = inner_folds(train, job["wf_folds"])
        # a FRESH seen{} per fold: reusing scores across folds would leak a
        # config's performance on one fold's test window into another's search
        pop, n, _ = evolve(eng, sp, ifolds, job["seed"] + i, iters, gens,
                           elite, min_trades, pool=pool, seen={}, quiet=True)
        win = pop[0]
        exc, ms, ntr = eng.excess_fold(test, win)      # the honest number
        rows.append({"fold": i + 1, "train_start": str(d(train.start)),
                     "train_end": str(d(train.stop - 1)),
                     "test_start": str(d(test.start)),
                     "test_end": str(d(test.stop - 1)),
                     "train_fit": win["_fit"], "train_exc": win["_exc"],
                     "test_exc": round(exc, 1), "test_ms_exc": round(ms, 1),
                     "test_trades": ntr, "n_configs": n,
                     "stop": win["stop"], "trail": win["trail"],
                     "thr": win["thr"], "htf_thr": win["htf"], "mp": win["mp"],
                     **{f"w_{nm}": wv for nm, wv in zip(FACTOR_NAMES,
                                                        win["w"])}})
        print(f"           winner: train_exc={win['_exc']} -> "
              f"TEST exc={exc:+.1f}% vs hold, {ms:+.1f}% vs textbook, "
              f"{ntr} trades\n", flush=True)
    if pool:
        pool.close(); pool.join()

    df = pd.DataFrame(rows)
    meta = {"interval": job["interval"], "gate": job["gate"],
            "market": job["market"], "months": job["months"],
            "universe": job["universe"], "gate_mode": job["gate_mode"],
            "fib_anchor": job["fib_anchor"], "wf": True, "oos": False}
    save_results(df, meta, {"search": "validate_nested", "mode": mode,
                            "outer_folds": len(ofolds),
                            "train_frac": train_frac, "iters": iters,
                            "gens": gens, "universe": job["universe"],
                            "space_sig": space_sig(sp)})

    show = ["fold", "test_start", "test_end", "train_exc", "test_exc",
            "test_ms_exc", "test_trades"]
    pd.set_option("display.width", 220)
    print(f"done in {time.time()-t0:.0f}s (persisted to {RESULTS_DIR}/)\n")
    print(df[show].to_string(index=False))

    tr = df[df["test_trades"] >= min_trades]
    print(f"\nVERDICT -- this measures the SEARCH, not a config (each fold has "
          f"a different winner):")
    if not len(tr):
        print("  no fold's winner traded enough on its unseen window to judge. "
              "That is a real answer: the search is not producing configs that "
              "act in these regimes.")
    else:
        pos = int((tr["test_exc"] > 0).sum())
        posm = int((tr["test_ms_exc"] > 0).sum())
        print(f"  beat buy-and-hold on {pos}/{len(tr)} unseen windows "
              f"(mean {tr['test_exc'].mean():+.1f}%, median "
              f"{tr['test_exc'].median():+.1f}%)")
        print(f"  beat the best textbook strat on {posm}/{len(tr)} "
              f"(mean {tr['test_ms_exc'].mean():+.1f}%)")
        print(f"  median trades per unseen window: "
              f"{tr['test_trades'].median():.0f}")
        skipped = len(df) - len(tr)
        if skipped:
            print(f"  ({skipped} fold(s) excluded: winner traded < "
                  f"{min_trades} times on the unseen window -- no claim)")
    print("\nIf this generalizes, run agent_search on ALL data for the config "
          "you would actually trade, and trust it this much -- no more.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
