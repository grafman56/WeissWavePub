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

import glob
import os
import subprocess
import sys
import time

import pandas as pd

from search_space import DEFAULT_SPACE, load_space, parse_set_args
from sweep import RESULTS_DIR, grid_sig_of, load_results
from test_strategy import arg

TASK_DIR = "agent-tasks"
WSL_REPO = "/mnt/c/Users/graf/Documents/WeissWave"


def job_args(universe, gate, gate_mode, interval, market, months, seed, iters,
             gens, holdout, wf, space, sets):
    """Every grid-defining flag MUST be forwarded. Anything accepted here but
    not passed through is a silent no-op: the job quietly runs a different
    grid than the one the survivors are then queried against."""
    return [f"--universe={universe}", f"--gate={gate}",
            f"--gate-mode={gate_mode}", f"--interval={interval}",
            f"--market={market}", f"--months={months}",
            f"--seed={seed}", f"--iters={iters}", f"--gens={gens}",
            f"--holdout={holdout}", f"--wf-folds={wf}",
            f"--space={space}"] + [f"--set={k}={v}" for k, v in sets.items()]


def _n_runs():
    """How many result files exist right now -- the artifact we verify by."""
    return len(glob.glob(os.path.join(RESULTS_DIR, "*.parquet")))


def run_direct(args):
    """Run one agent_search job in-process as a subprocess."""
    subprocess.run([sys.executable, "agent_search.py", *args], check=False)


def run_via_agent(args, i, timeout_s=1800):
    """Write a task file and dispatch it to the opencode `runner` agent in WSL.

    Two things this gets right that are easy to get wrong:

    * `python.exe`, NOT `python3`. The agent's bash runs in WSL, where Linux
      python has no pandas/duckdb -- the whole stack is WINDOWS python.
      `python.exe` is the WSL->Windows interop shim. `python3 agent_search.py`
      dies instantly on ModuleNotFoundError.
    * The runner agent is documented to receive a PATH and `cat` it itself, so
      we pass the path -- not the file's contents.

    Failures are surfaced, never swallowed: stderr is captured and the caller
    verifies a new results artifact actually landed, because a small local
    model reporting "DONE" is not evidence that anything ran."""
    os.makedirs(TASK_DIR, exist_ok=True)
    tf = os.path.join(TASK_DIR, f"search_job{i}.txt")
    with open(tf, "w", newline="\n") as f:                 # LF for WSL
        f.write("python.exe agent_search.py " + " ".join(args) + "\n")
    wsl_tf = WSL_REPO + "/" + tf.replace("\\", "/")
    inner = (f"cd {WSL_REPO} && "
             f"timeout {timeout_s} ~/.opencode/bin/opencode run "
             f"--agent runner {wsl_tf} </dev/null")
    p = subprocess.run(["wsl", "bash", "-lc", inner], check=False,
                       capture_output=True, text=True)
    return p


def survivors(cur=None, grid_sig=None):
    """The only trustworthy candidates: configs that pass BOTH gates -- robust
    across the training walk-forward folds AND still beating buy-and-hold on the
    HELD-OUT slice. Beating hold on the holdout alone isn't enough (one lucky
    period); it has to have been robust in training too, or it's a fluke.

    The bar itself comes from search_space.json ("curation"), so it is a
    decision you can read rather than a number buried in a default arg.

    `grid_sig` scopes the query to rows computed under the SAME data and
    semantics. Without it this pools crypto with stocks, and hard-gated grids
    with gate-as-factor ones, then ranks them against each other as if the
    numbers meant the same thing. They do not."""
    cur = cur or load_space()["curation"]
    df = load_results()
    if "holdout_exc" not in df.columns:
        return pd.DataFrame()
    if grid_sig is not None and "grid_sig" in df.columns:
        df = df[df["grid_sig"] == grid_sig]
    df = df.dropna(subset=["holdout_exc"]).copy()
    if not len(df):
        return df
    # a config that never traded scores a deceptive 0.0 -- never a survivor
    if "unfit" in df.columns:
        df = df[df["unfit"] != True]                       # noqa: E712
    if "holdout_trades" in df.columns:
        df = df[df["holdout_trades"].fillna(1) >= 1]

    def frac(p):
        try:
            a, b = str(p).split("/")
            return int(a) / int(b)
        except (ValueError, ZeroDivisionError):
            return 0.0

    df["_posf"] = df["wf_pos"].apply(frac)
    keep = ((df["holdout_exc"] > cur["min_holdout_exc"])
            & (df["wf_min"] > cur["min_worst_fold"])
            & (df["_posf"] >= cur["min_pos_frac"]))
    return df[keep].sort_values("holdout_exc", ascending=False)


def main():
    args = sys.argv[1:]
    space = arg(args, "space", DEFAULT_SPACE)
    sets = parse_set_args(args)
    sp = load_space(space, sets)
    G, VAL = sp["grid"], sp["validation"]

    universe = arg(args, "universe", G["universe"])
    gate = arg(args, "gate", G["gate"])
    gate_mode = arg(args, "gate-mode", G["gate_mode"])
    interval = arg(args, "interval", G["interval"])
    market = arg(args, "market", G["market"])
    months = int(arg(args, "months", str(G["months"])))
    seeds = [int(x) for x in arg(args, "seeds", "1,2,3,4").split(",")]
    iters = int(arg(args, "iters", "300"))
    gens = int(arg(args, "gens", "5"))
    holdout = float(arg(args, "holdout", str(VAL["holdout"])))
    wf = int(arg(args, "wf-folds", str(VAL["wf_folds"])))
    tmo = int(arg(args, "agent-timeout", "1800"))
    via = "--via-agents" in args
    gsig = grid_sig_of(interval, gate, market, months, universe, gate_mode)

    print(f"orchestrating {len(seeds)} search jobs: {interval} {universe} "
          f"gate={gate}/{gate_mode} "
          f"(mode={'opencode runner' if via else 'direct'}); each: {iters} "
          f"seeds x {gens} gens, holdout={holdout:.0%}", flush=True)
    t0 = time.time()
    ok = 0
    for i, s in enumerate(seeds):
        ja = job_args(universe, gate, gate_mode, interval, market, months, s,
                      iters, gens, holdout, wf, space, sets)
        print(f"  [{i+1}/{len(seeds)}] seed={s} ...", flush=True)
        before = _n_runs()
        if via:
            p = run_via_agent(ja, i, tmo)
            # VERIFY BY ARTIFACT, not by the agent's word: a small local model
            # will happily reply DONE having run nothing at all.
            landed = _n_runs() > before
            if not landed:
                tail = (p.stderr or p.stdout or "").strip().splitlines()[-6:]
                print(f"      FAILED: no new {RESULTS_DIR}/ artifact. "
                      f"rc={p.returncode}. last output:", flush=True)
                for ln in tail:
                    print(f"        {ln}", flush=True)
            else:
                ok += 1
                print("      ok: results artifact landed", flush=True)
        else:
            run_direct(ja)
            ok += _n_runs() > before

    print(f"\n{ok}/{len(seeds)} job(s) produced results.")
    if via and not ok:
        print("No agent job landed anything. Check LM Studio has a model "
              "loaded, and that the runner agent exists in WSL at "
              "~/.config/opencode/agent/runner.md.")
    sv = survivors(sp["curation"], gsig)
    print(f"\ndone in {time.time()-t0:.0f}s. {len(sv)} config(s) matching this "
          f"grid beat buy-and-hold on their HELD-OUT slice.\n  grid_sig={gsig}")
    if len(sv):
        cols = [c for c in ["holdout_exc", "wf_exc", "wf_min", "wf_pos",
                            "wf_trades", "stop", "trail", "td", "thr",
                            "htf_thr", "mp"] if c in sv.columns]
        pd.set_option("display.width", 220)
        print("Top holdout survivors (the only trustworthy candidates):")
        print(sv[cols].head(15).to_string(index=False))
        print("\nCAUTION: the holdout is ONE FIXED slice. The more configs you "
              "score against it, the more a 'survivor' is just the one that "
              "got lucky on it. Treat these as candidates to re-test, not "
              "answers.")
    else:
        print("None yet -- the search space explored so far overfits its folds. "
              "Run more seeds/iters, try another universe, or add factors.")


if __name__ == "__main__":
    main()
