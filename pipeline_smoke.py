#!/usr/bin/env python3
"""Pipeline smoke: can the local-agent stack run a job and report it readably?

NOT a bug hunt, NOT a search. Bug-finding is Claude's job; the agents are not
set up for it and are not meant to be. This answers one question:

    store summary -> strategist -> job lines -> validation -> python -> store

does that work, end to end, fast, with output a human can read? That is all.
The point is a pipeline that can later run real indicator-combination jobs.

    python pipeline_smoke.py              # stages 0-2, no execution (~60s)
    python pipeline_smoke.py --execute    # + one real job on a WARM grid

STAGED ON PURPOSE, AND THIS IS THE WHOLE DESIGN NOTE:
The expensive stage is the grid build, and `grid_sig` includes fib_anchor and
gate_mode -- both of which the strategist is ALLOWED TO CHOOSE. So a strategist
doing its job well (picking ground not yet covered) picks a COLD grid, and the
"quick" pipeline check turns into a 10-minute rebuild. Measured: >100s per job
even at iters=20 gens=1 on a cold grid.

So testing the PIPELINE and testing the SEARCH are different jobs, and only one
of them can be fast. Stages 0-2 need no grid at all. Stage 3 pins the grid to
the warm default and hands the strategist's choice back to Python -- because
what stage 3 checks is "does execution land an artifact", not "was the model's
idea good".

VERIFY BY ARTIFACT. Stage 3 passes only if a new .parquet appears in the store.
Never because a model said it ran something.
"""

import subprocess
import sys
import time

import orchestrate as orc
from search_space import DEFAULT_SPACE, load_space

# Hard cap on stage 3. A cold 15m stocks grid at months=0 hit 14.1GB resident
# and had to be killed by hand. A "quick pipeline check" must never be able to
# do that, so the cap is the feature, not a safety net.
CAP_S = 120

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")
    if not cond:
        fails.append(name)
    return cond


def main():
    execute = "--execute" in sys.argv
    t0 = time.time()

    # ── 0. preflight ─────────────────────────────────────────────────────────
    # A stopped LM Studio is INDISTINGUISHABLE from a slow one. This is the
    # failure that has cost the most time here: waiting minutes on something
    # that was never going to answer.
    print("\n[0] preflight")
    models = orc.model_loaded()
    if not check("LM Studio is serving", bool(models),
                 f"{len(models)} models, incl: {', '.join(models[:3])}" if models
                 else "nothing at :1234. START LM STUDIO. This is not a hang."):
        return 1

    # ── 1. the summary the model is given ────────────────────────────────────
    print("\n[1] store summary (the model's whole input)")
    summary = orc.store_summary()
    lines = [l for l in summary.strip().splitlines() if l.strip()]
    check("summary is non-empty", len(lines) > 1, f"{len(lines)} lines")
    check("summary describes GRIDS, not raw rows",
          any("|" in l and "scored" in l for l in lines),
          "per-grid: a matching grid_sig is the only thing that makes two "
          "configs comparable")
    for l in lines[1:4]:
        print(f"        | {l[:100]}")

    # ── 2. strategist -> job lines -> whitelist ──────────────────────────────
    print("\n[2] strategist round-trip (the only place a model is used)")
    t = time.time()
    raw, err = orc.ask_strategist(summary, timeout_s=90)
    dt = time.time() - t
    if not check("strategist answered", err is None,
                 err or f"{dt:.0f}s for a {len(summary)}-char prompt"):
        return 1

    proposed, rejected = orc.parse_jobs(raw)
    check("emitted at least one VALID job line", bool(proposed),
          f"{len(proposed)} valid, {len(rejected)} rejected")
    for line, why in rejected:
        print(f"        REJECTED {line!r}: {why}")
    for j in proposed:
        print(f"        job: {j}")
    if not proposed:
        # The 4B is not deterministic and sometimes returns nothing usable.
        # orchestrate's --director loop already shrugs and continues (3 misses
        # in a row = the stack is broken, not the model unlucky). Print what it
        # actually said -- "0 valid, 0 rejected" means it emitted no job-shaped
        # lines at ALL, which is a different failure from emitting a bad one.
        print("        raw reply was:")
        for ln in (raw.strip().splitlines() or ["<empty>"])[-6:]:
            print(f"        | {ln[:110]}")

    # The validator is the load-bearing part. The model's text is UNTRUSTED
    # INPUT: regex-matched, range-checked, never eval'd, never shell-
    # interpolated. It has invented `gate_mode=none` before and this caught it.
    check("every accepted job carries the required fields",
          all({"universe", "seed", "iters", "gens"} <= set(j)
              for j in proposed) if proposed else False)
    check("no accepted job is out of range",
          all(j["universe"] in ("crypto", "stocks")
              and 0 < j["iters"] <= 2000 and 0 < j["gens"] <= 20
              for j in proposed) if proposed else False,
          "a job that fails this never reaches a subprocess")

    if not execute or not proposed:
        if execute and not proposed:
            print("\n[3] SKIPPED -- the strategist gave nothing to run. That is "
                  "a strategist result, not an execution failure.")
        print(f"\nstages 0-2 in {time.time()-t0:.0f}s"
              + ("" if execute else ". --execute adds one real job."))
        return 1 if fails else 0

    # ── 3. execution -> artifact ─────────────────────────────────────────────
    # EVERY grid-defining field comes from the space, INCLUDING universe. Only
    # `seed` is taken from the model.
    #
    # The first version of this took the model's `universe` and still printed
    # "(warm)". universe is IN grid_sig, so that landed a cold 15m stocks grid:
    # 513 symbols of intraday bars at months=0 (= ALL history), 14.1GB resident
    # before it got killed. The label was a lie -- the exact "tool reporting
    # something it did not do" shape this repo is full of, written into the
    # thing meant to check for it.
    #
    # So: pin the whole grid, and TIME-BOX it. A pipeline check that can become
    # a 14GB grid build is not a pipeline check.
    print("\n[3] execution -> artifact")
    sp = load_space(DEFAULT_SPACE, {})
    G = sp["grid"]
    before = orc._n_runs()

    seed = proposed[0]["seed"]               # the ONLY thing taken from the model
    args = orc.job_args(G["universe"], G["gate"], G["gate_mode"],
                        G["interval"], G["market"], G["months"],
                        seed, 20, 1,         # smallest thing that still scores
                        sp["validation"]["holdout"], sp["validation"]["wf_folds"],
                        DEFAULT_SPACE, {}, 1)
    args.append(f"--fib-anchor={G.get('fib_anchor', '1d')}")
    grid = (f"{G['interval']}|{G['gate']}|{G['gate_mode']}|{G['market']}|"
            f"{G['months']}mo|{G['universe']}|fib@{G.get('fib_anchor')}")
    print(f"        grid : {grid}")
    print(f"        job  : seed={seed} (from the model) iters=20 gens=1")
    print(f"        store: {before} result files before")

    t = time.time()
    timed_out = False
    try:
        subprocess.run([sys.executable, "-W", "ignore", "agent_search.py", *args],
                       check=False, timeout=CAP_S,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        timed_out = True
    dt = time.time() - t
    after = orc._n_runs()

    if timed_out:
        # Not a plumbing failure. Report it as the COST it is, by name.
        check(f"job finished inside the {CAP_S}s cap", False,
              f"still building after {CAP_S}s -- this grid is COLD. The build, "
              f"not the pipeline, is the cost. Warm it once or pin the grid.")
    else:
        check("a NEW result file landed in the store", after > before,
              f"{before} -> {after} files in {dt:.0f}s. The artifact is the "
              f"proof; a model saying it ran something is not.")
        check("the job was warm (seconds, not a rebuild)", dt < 60,
              f"{dt:.0f}s. Warm = seconds. A cold grid is minutes and GBs, "
              f"which is what the strategist buys when it picks new ground.")

    print(f"\nall stages in {time.time()-t0:.0f}s")
    return 1 if fails else 0


if __name__ == "__main__":
    code = main()
    print("\n" + ("PIPELINE OK" if not code
                  else f"{len(fails)} FAILED: " + ", ".join(fails)))
    sys.exit(code)
