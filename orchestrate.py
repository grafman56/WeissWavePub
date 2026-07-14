#!/usr/bin/env python3
"""Trading search orchestrator. Runs many agent_search jobs, curates the configs
that survive, and reports them. The deduped result store means jobs never repeat
each other's work -- compute accumulates instead of churning.

    python orchestrate.py --universe=crypto --seeds=1,2,3,4 --iters=300 --gens=5
    python orchestrate.py --director        # let the local LLM choose the jobs

THE MODEL DECIDES, PYTHON EXECUTES.
Execution is ALWAYS a subprocess -- deterministic, ~4s, incapable of fabricating
a result. The local model is used for the one thing worth paying prefill for:
DIRECTION. `--director` hands ww-strategist a summary of what has been tested
and asks which ground to cover next; parse_jobs() validates every line it
returns against a whitelist before anything runs, and its text is never
executed.

This replaced a `runner` agent whose whole job was to `cat` a file and run the
one command inside it -- a 4B model doing `bash -c`. It cost 133-598s instead of
4s, could silently lose the output it relayed, and was mode:subagent, which
`opencode run --agent` cannot invoke at all: dispatches fell through to whatever
primary agent was default, which is another project's `coder`. A game-building
agent was being handed trading jobs. Execution is not a job for a language model.

ww-strategist lives in THIS REPO at .opencode/agent/ (opencode reads
project-local agents), so it is versioned with the code, scoped to this project,
and other projects' global agents stay untouched.

ASCII output."""

import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import pandas as pd

from search_space import DEFAULT_SPACE, load_space, parse_set_args
from sweep import RESULTS_DIR, grid_sig_of, load_results
from test_strategy import arg

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


PROMPT_FILE = os.path.join(".opencode", "_prompt.txt")


def model_loaded(url="http://127.0.0.1:1234/v1/models", timeout=4):
    """PREFLIGHT. Verify LM Studio is serving before dispatching, so a JIT model
    load -- or a stopped server -- cannot masquerade as a hang. Cheap insurance
    against the failure mode that has cost the most time here: waiting minutes
    on something that was never going to answer."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return [m["id"] for m in json.load(r).get("data", [])]
    except Exception:                                        # noqa: BLE001
        return []


def ask_strategist(summary, timeout_s=60):
    """Ask ww-strategist what to search next. Returns (raw_text, error_or_None).

    THE MODEL DECIDES, PYTHON EXECUTES. The strategist has ZERO tools and never
    touches bash -- it reads a summary and emits job lines, which parse_jobs()
    validates before anything runs. Its text is never executed.

    THREE HARD-WON CAVEATS ARE BAKED IN HERE. Change them at your peril:

    1. THE PROMPT GOES VIA STDIN FROM A FILE, never argv. opencode 1.17.18 hangs
       FOREVER at init when an argv prompt contains double quotes and stdin is a
       non-TTY (which it always is when invoked from a script). Prompts read
       from files were never affected. This one silently burned 4x240s timeouts
       in a past session.
    2. PREFLIGHT THE MODEL. A stopped LM Studio looks exactly like a slow model.
    3. TIMEOUT FROM A BASELINE, NOT PADDING. A zero-tool agent answers in
       ~4-20s; 60s is already 3x headroom. If it needs more than this, something
       is wrong and waiting will not fix it -- fail fast and diagnose.

    This replaced a `runner` agent that existed only to `cat` a file and run the
    one command inside it: a 4B model doing `bash -c`, at 133-598s instead of
    4s, able to lose the output it relayed, and -- being mode:subagent, which
    `opencode run --agent` cannot invoke -- silently falling through to whatever
    primary agent was default. That is another project's `coder`: a game-builder
    handed trading jobs. Execution is not a job for a language model.

    ww-strategist lives in THIS REPO at .opencode/agent/ (opencode reads
    project-local agents), so it is versioned with the code, scoped here, and
    other projects' global agents stay untouched."""
    have = model_loaded()
    if not have:
        return "", ("LM Studio is not serving on :1234 -- start it and load a "
                    "model (`lms ps`). Not dispatching.")
    os.makedirs(".opencode", exist_ok=True)
    with open(PROMPT_FILE, "w", newline="\n", encoding="utf-8") as f:
        f.write(summary + "\n")
    inner = (f"cd {WSL_REPO} && timeout {timeout_s} ~/.opencode/bin/opencode "
             f"run --agent ww-strategist < {WSL_REPO}/.opencode/_prompt.txt")
    t0 = time.time()
    p = subprocess.run(["wsl", "bash", "-lc", inner], check=False,
                       capture_output=True, text=True)
    dt = time.time() - t0
    raw = (p.stdout or "") + (p.stderr or "")
    if p.returncode == 124 or dt >= timeout_s - 1:
        return raw, (f"opencode timed out after {dt:.0f}s. A zero-tool agent "
                     f"should answer in ~4-20s -- this is the known hang, not "
                     f"a slow model. Check `lms ps` and the LM Studio log.")
    return raw, None


JOB_RE = re.compile(r"universe=(\w+)\s+seed=(\d+)\s+iters=(\d+)\s+gens=(\d+)"
                    r"(?:\s+fib_anchor=(\w+))?(?:\s+gate_mode=(\w+))?")
_UNIVERSES = {"crypto", "stocks"}
_ANCHORS = {"self", "4h", "1d", "1w"}
_GATE_MODES = {"hard", "factor"}


def parse_jobs(text, limit=4):
    """Model text -> validated job dicts. Anything that does not match the
    grammar, or falls outside the whitelists/ranges, is DROPPED with a reason.

    The model's output is untrusted input: it is matched against a regex and
    range-checked, never eval'd, never interpolated into a shell string. A 4B
    model will happily emit `universe=; rm -rf /` if it gets confused."""
    jobs, rejected = [], []
    for m in JOB_RE.finditer(text or ""):
        u, seed, iters, gens, anchor, gm = m.groups()
        why = None
        if u not in _UNIVERSES:
            why = f"universe {u!r} not in {sorted(_UNIVERSES)}"
        elif not (10 <= int(iters) <= 2000):
            why = f"iters={iters} outside 10..2000"
        elif not (0 <= int(gens) <= 20):
            why = f"gens={gens} outside 0..20"
        elif anchor and anchor not in _ANCHORS:
            why = f"fib_anchor {anchor!r} not in {sorted(_ANCHORS)}"
        elif gm and gm not in _GATE_MODES:
            why = f"gate_mode {gm!r} not in {sorted(_GATE_MODES)}"
        if why:
            rejected.append((m.group(0), why))
            continue
        j = {"universe": u, "seed": int(seed), "iters": int(iters),
             "gens": int(gens)}
        if anchor:
            j["fib_anchor"] = anchor
        if gm:
            j["gate_mode"] = gm
        jobs.append(j)
    return jobs[:limit], rejected


def store_summary(gsig=None):
    """A compact plain-text account of what has been tested, for the strategist.
    Small on purpose: prefill is the bottleneck on the local card, and the model
    only needs enough to choose the next ground.

    Survivor stats are reported PER GRID, never pooled. A score from a
    hard-gated 12-month crypto grid says nothing about a gate-as-factor
    full-history one, so a pooled "86 configs beat buy-and-hold" is a number
    with no referent -- and it would push the model toward ground that only
    looked good because incompatible rows were counted together."""
    df = load_results()
    if not len(df) or "spec" not in df.columns:
        return "Nothing has been tested yet."
    d = df[df["spec"].str.contains("agent_search", na=False)]
    if not len(d) or "grid_sig" not in d.columns:
        return "Nothing has been tested yet."
    lines = []
    for sig, g in d.groupby("grid_sig"):
        if "holdout_exc" not in g.columns:
            continue
        h = g.dropna(subset=["holdout_exc"])
        if "unfit" in h.columns:
            h = h[h["unfit"] != True]                       # noqa: E712
        if "wf_trades" in h.columns:            # a verdict needs actual trades
            h = h[h["wf_trades"].fillna(-1) != 0]
        if not len(h):
            lines.append(f"{sig}: searched, no scored config traded")
            continue
        surv = h[h["holdout_exc"] > 0]
        cur = " <-- the grid you are choosing for" if sig == gsig else ""
        if len(surv):
            b = surv.sort_values("holdout_exc", ascending=False).iloc[0]
            lines.append(f"{sig}: {len(h)} scored, {len(surv)} beat "
                         f"buy-and-hold on the holdout; best {b['holdout_exc']}%"
                         f"{cur}")
        else:
            lines.append(f"{sig}: {len(h)} scored, NONE beat buy-and-hold on "
                         f"the holdout (this ground overfits){cur}")
    if not lines:
        return "Nothing has been tested yet."
    return ("What has been searched so far (one line per grid; a grid is "
            "interval|gate|gate_mode|market|months|universe|fib_anchor):\n"
            + "\n".join(lines[:8]))


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
    tmo = int(arg(args, "agent-timeout", "60"))   # baseline, not padding
    director = "--director" in args or "--via-agents" in args
    gsig = grid_sig_of(interval, gate, market, months, universe, gate_mode,
                       arg(args, "fib-anchor", G.get("fib_anchor", "1d")))

    if director:
        # THE MODEL DECIDES, PYTHON EXECUTES. The strategist reads a summary and
        # proposes job lines; every line is validated here before anything runs.
        print("asking ww-strategist what to search next "
              "(zero tools, it never touches bash) ...", flush=True)
        summary = store_summary(gsig)
        raw, err = ask_strategist(summary, tmo)
        if err:
            print(f"  {err}", flush=True)
            return
        proposed, rejected = parse_jobs(raw)
        for line, why in rejected:
            print(f"  REJECTED {line!r}: {why}", flush=True)
        if not proposed:
            print("  the strategist proposed nothing usable. Raw reply:",
                  flush=True)
            for ln in raw.strip().splitlines()[-8:]:
                print(f"      | {ln}", flush=True)
            print("  (is a model loaded? `lms ps`. Is ww-strategist listed? "
                  "`opencode agent list` from this repo.)")
            return
        print(f"  strategist proposed {len(proposed)} job(s):", flush=True)
        for j in proposed:
            print(f"    {j}", flush=True)
        t0 = time.time()
        ok = 0
        for i, j in enumerate(proposed):
            ja = job_args(j["universe"], gate, j.get("gate_mode", gate_mode),
                          interval, market, months, j["seed"], j["iters"],
                          j["gens"], holdout, wf, space, sets)
            if "fib_anchor" in j:
                ja.append(f"--fib-anchor={j['fib_anchor']}")
            print(f"  [{i+1}/{len(proposed)}] running {j['universe']} "
                  f"seed={j['seed']} ...", flush=True)
            before = _n_runs()
            run_direct(ja)                    # a subprocess. Always.
            ok += _n_runs() > before
        print(f"\n{ok}/{len(proposed)} job(s) produced results in "
              f"{time.time()-t0:.0f}s.")
        sv = survivors(sp["curation"], None)
        print(f"{len(sv)} config(s) in the store beat buy-and-hold on a "
              f"holdout. (Holdout caveat below still applies.)")
        return

    print(f"orchestrating {len(seeds)} search jobs: {interval} {universe} "
          f"gate={gate}/{gate_mode} "
          f"(direct subprocess); each: {iters} "
          f"seeds x {gens} gens, holdout={holdout:.0%}", flush=True)
    t0 = time.time()
    ok = 0
    for i, s in enumerate(seeds):
        ja = job_args(universe, gate, gate_mode, interval, market, months, s,
                      iters, gens, holdout, wf, space, sets)
        print(f"  [{i+1}/{len(seeds)}] seed={s} ...", flush=True)
        before = _n_runs()
        run_direct(ja)                       # a subprocess. Always.
        ok += _n_runs() > before

    print(f"\n{ok}/{len(seeds)} job(s) produced results.")
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
