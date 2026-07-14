#!/usr/bin/env python3
"""AGENT PIPELINE SMOKE TEST -- does the local-LLM path actually work?

PASS/FAIL on the plumbing ONLY. It answers "did the agent really run a search
and land a result", never "is the strategy any good". Deliberately tiny: a few
configs on a short window, because the point is the pipe, not the numbers.

EVERY CHECK IS BY ARTIFACT. A small local model will happily reply "DONE"
having run nothing at all -- so nothing here trusts what the agent SAYS. It
compares the results store before and after, opens what landed, and verifies
the contents match the job that was dispatched.

    python agent_check.py                 # direct subprocess (no LLM)
    python agent_check.py --via-agents    # through the WSL opencode runner

Exit code 0 = pass, 1 = fail. ASCII output.
"""

import glob
import os
import subprocess
import sys
import time

import pandas as pd

from sweep import RESULTS_DIR, grid_sig_of
from test_strategy import arg

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          f"{' -- ' + detail if detail else ''}", flush=True)
    return ok


def snapshot():
    return set(glob.glob(os.path.join(RESULTS_DIR, "*.parquet")))


def main():
    args = sys.argv[1:]
    via = "--via-agents" in args
    seed = arg(args, "seed", str(int(time.time()) % 100000))  # fresh: no dedup
    universe = arg(args, "universe", "crypto")
    months = arg(args, "months", "12")
    tmo = arg(args, "agent-timeout", "600")

    print(f"AGENT PIPELINE CHECK ({'via opencode runner' if via else 'direct'})"
          f" -- plumbing only, not strategy quality\n")

    before = snapshot()
    cmd = [sys.executable, "orchestrate.py", f"--seeds={seed}",
           f"--universe={universe}", f"--months={months}",
           "--iters=6", "--gens=1", "--wf-folds=3",
           f"--agent-timeout={tmo}"] + (["--via-agents"] if via else [])
    print("$ " + " ".join(cmd[1:]) + "\n")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    tail = "\n".join(f"      | {ln}" for ln in out.strip().splitlines()[-12:])

    print(f"\nchecks (ran in {dt:.0f}s):")
    ok = check("orchestrate exited cleanly", p.returncode == 0,
               f"rc={p.returncode}")
    new = snapshot() - before
    # THE check: an artifact landed. Not "the agent said DONE".
    ok &= check("a NEW results artifact landed", len(new) > 0,
                f"{len(new)} new file(s) in {RESULTS_DIR}/")
    ok &= check("orchestrate reported the job produced results",
                "1/1 job(s) produced results" in out
                or "job(s) produced results" in out)

    if new:
        f = sorted(new)[0]
        try:
            df = pd.read_parquet(f)
        except Exception as e:                       # noqa: BLE001
            check("artifact is readable", False, str(e)[:80])
            df = None
        if df is not None:
            ok &= check("artifact is readable and non-empty", len(df) > 0,
                        f"{len(df)} rows")
            want = {"cfg_sig", "grid_sig", "wf_exc", "wf_trades", "unfit"}
            ok &= check("artifact has the expected columns",
                        want <= set(df.columns),
                        f"missing: {sorted(want - set(df.columns))}"
                        if not want <= set(df.columns) else "")
            if "grid_sig" in df.columns:
                # the job that landed must be the job we asked for -- a silently
                # unforwarded flag would show up right here
                got = str(df["grid_sig"].iloc[0])
                exp_u = f"|{universe}|"
                ok &= check("artifact's grid_sig matches the dispatched job",
                            exp_u in got and f"{months}mo" in got, got)
            if "wf_exc" in df.columns:
                ok &= check("scores are finite (the engine actually ran)",
                            bool(df["wf_exc"].notna().any()),
                            f"{int(df['wf_exc'].notna().sum())}/{len(df)} scored")
    else:
        print("\n  last output from the run:")
        print(tail)

    if via:
        ok &= check("the agent path was used, not the direct fallback",
                    "opencode runner" in out)

    n_pass = sum(1 for _, o, _ in CHECKS if o)
    print(f"\n{'PASS' if ok else 'FAIL'}: {n_pass}/{len(CHECKS)} checks")
    if not ok:
        print("\nTroubleshooting: is a model loaded in LM Studio (`lms ps`)? "
              "does ~/.config/opencode/agent/runner.md exist in WSL? "
              "the runner must call python.exe, NOT python3 -- WSL's python3 "
              "has no pandas.")
        print(tail)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
