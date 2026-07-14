#!/usr/bin/env python3
"""Trading search orchestrator. Runs many agent_search jobs across universes and
seeds, curates the configs that SURVIVE THE HOLDOUT (beat buy-and-hold on data
the search never saw), and reports them. The deduped result store means the jobs
never repeat each other's work -- more compute accumulates instead of churning.

Two execution modes:
  default   : run each job as a local subprocess (reliable, testable now)
  --via-agents : dispatch each job as a task file to the opencode `runner`
                 agent in WSL, so your local qwen agents literally run the
                 searches (the "agents test combinations" mode)

    python orchestrate.py --universe=crypto --seeds=1,2,3,4 --iters=300 --gens=5
    python orchestrate.py --via-agents --seeds=1,2,3,4

ASCII output. The Python search (agent_search.py) is the workhorse; agents
orchestrate and curate -- what small local models are actually good at."""

import os
import subprocess
import sys
import time

import pandas as pd

from sweep import load_results
from test_strategy import arg

TASK_DIR = "agent-tasks"


def job_args(universe, gate, seed, iters, gens, holdout, wf):
    return [f"--universe={universe}", f"--gate={gate}", f"--seed={seed}",
            f"--iters={iters}", f"--gens={gens}", f"--holdout={holdout}",
            f"--wf-folds={wf}"]


def run_direct(args):
    """Run one agent_search job in-process as a subprocess."""
    subprocess.run([sys.executable, "agent_search.py", *args], check=False)


def run_via_agent(args, i):
    """Write a task file and dispatch it to the opencode `runner` agent in WSL
    (the exact invocation from the local-agents playbook: paths through the
    boss are avoided here -- we drive the runner directly, one job at a time)."""
    os.makedirs(TASK_DIR, exist_ok=True)
    tf = os.path.join(TASK_DIR, f"search_job{i}.txt")
    with open(tf, "w", newline="\n") as f:                 # LF for WSL
        f.write("python3 agent_search.py " + " ".join(args) + "\n")
    wsl_tf = "/mnt/c/Users/graf/Documents/WeissWave/" + tf.replace("\\", "/")
    inner = (f"cd /mnt/c/Users/graf/Documents/WeissWave && "
             f"timeout 600 ~/.opencode/bin/opencode run --agent runner "
             f'"$(tr -d \'\\r\' < {wsl_tf})" </dev/null 2>/dev/null')
    subprocess.run(["wsl", "bash", "-lc", inner], check=False)


def survivors(min_pos=0.66, min_worst=-25.0):
    """The only trustworthy candidates: configs that pass BOTH gates -- robust
    across the training walk-forward folds (>= min_pos of folds positive, worst
    fold above min_worst) AND still beat buy-and-hold on the HELD-OUT slice.
    Beating hold on the holdout alone isn't enough (one lucky period); it has to
    have been robust in training too, or it's a fluke."""
    df = load_results()
    if "holdout_exc" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["holdout_exc"]).copy()

    def frac(p):
        try:
            a, b = str(p).split("/")
            return int(a) / int(b)
        except (ValueError, ZeroDivisionError):
            return 0.0

    df["_posf"] = df["wf_pos"].apply(frac)
    keep = ((df["holdout_exc"] > 0) & (df["wf_min"] > min_worst)
            & (df["_posf"] >= min_pos))
    return df[keep].sort_values("holdout_exc", ascending=False)


def main():
    args = sys.argv[1:]
    universe = arg(args, "universe", "crypto")
    gate = arg(args, "gate", "minervini@1d")
    seeds = [int(x) for x in arg(args, "seeds", "1,2,3,4").split(",")]
    iters = int(arg(args, "iters", "300"))
    gens = int(arg(args, "gens", "5"))
    holdout = float(arg(args, "holdout", "0.2"))
    wf = int(arg(args, "wf-folds", "6"))
    via = "--via-agents" in args

    print(f"orchestrating {len(seeds)} search jobs on universe={universe} "
          f"(mode={'opencode runner' if via else 'direct'}); each: {iters} "
          f"seeds x {gens} gens, holdout={holdout:.0%}", flush=True)
    t0 = time.time()
    for i, s in enumerate(seeds):
        ja = job_args(universe, gate, s, iters, gens, holdout, wf)
        print(f"  [{i+1}/{len(seeds)}] seed={s} ...", flush=True)
        (run_via_agent(ja, i) if via else run_direct(ja))

    sv = survivors()
    print(f"\ndone in {time.time()-t0:.0f}s. {len(sv)} config(s) across the "
          f"store beat buy-and-hold on their HELD-OUT slice.")
    if len(sv):
        cols = [c for c in ["holdout_exc", "wf_exc", "wf_min", "wf_pos", "stop",
                            "trail", "td", "thr", "htf_thr", "mp"]
                if c in sv.columns]
        pd.set_option("display.width", 220)
        print("Top holdout survivors (the only trustworthy candidates):")
        print(sv[cols].head(15).to_string(index=False))
    else:
        print("None yet -- the search space explored so far overfits its folds. "
              "Run more seeds/iters, try another universe, or add factors.")


if __name__ == "__main__":
    main()
