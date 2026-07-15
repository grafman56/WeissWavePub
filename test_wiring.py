#!/usr/bin/env python3
"""Wiring smoke gate: does every documented flag actually MOVE THE NUMBER?

This is the repo's bug shape mechanised. Nine smoke tests on 2026-07-14 found
nine real bugs and every one was "a tool reporting something it did not do" or
"two tools disagreeing on a default" -- never arithmetic. `--symbols` was dead
for every symbol (a swallowed NameError), `--exit` was dead once before, and
`--gate-mode` was never read by two of three tools. All of them printed a
perfectly reasonable header describing the filter they were not applying.

A flag that cannot change the output is a lie in the header. So: run the tool
twice, change ONE flag, and assert the result moved the way the flag claims.

    python test_wiring.py                 # the gate (exit 0 = all pass)
    python test_wiring.py --verbose       # show every command and metric
    python test_wiring.py --only=cost-bps # one case, for debugging

WHY THIS SHAPE, AND NOT A UNIT TEST:
A unit test asserts the function does what the function does. It cannot catch a
CLI that parses `--symbols` into a variable nobody reads, because the parse is
correct and the read is missing. Only end-to-end, through the real argv, with
the real DB, catches that -- so this drives the actual command line.

READ BEFORE ADDING A CASE -- "verify the innocent explanation first":
A flag that does not move the number is NOT automatically dead. It has nearly
always turned out to be a real reason wearing a bug's clothes:
  * `--filter=buy_dominant` changed nothing because it is IMPLIED by heavy_buy.
  * `--fib-target` changed nothing because a -62.7% year had no rally to hit.
  * `tdi_long` x `above_50ma` was exactly 0 because tdi_long fires a mean 12%
    BELOW the 50-SMA, so the intersection is genuinely empty.
So every case here carries a `why` naming the mechanism, and the baseline is
chosen to make the mechanism reachable (a wide universe, a frequent signal, a
window with both directions in it). If a case fails, the FIRST move is to check
its `why` still holds -- not to file a bug.

DIRECTION, NOT JUST MOVEMENT. "n changed" is a weak assertion: noise, a
re-anchor, or an off-by-one all satisfy it. Where the flag's own docs commit to
a direction, this asserts the direction. Where it commits to a QUANTITY
(--cost-bps=100 is one round-trip percent off every trade), it asserts the
quantity, which is the strongest form available and the only one that would
catch the flag being read but applied at the wrong scale.

Baseline: heavy_buy 1d over the last 24mo, all 513 stocks -- n=4695, ~6.5s on a
warm cache. Deliberately wide: "a wider universe settles it in one run", and a
3-symbol run gives n=2, where no knob can move anything and every case would
pass by being unable to fail.
"""

import re
import subprocess
import sys
import time

PY = sys.executable
TOOL = "test_strategy.py"

# The baseline every case perturbs. One flag changes per case; everything else
# is pinned here so a case's delta is attributable to its own flag.
BASE = ["heavy_buy", "--interval=1d", "--months=24"]

RESULT_RE = re.compile(
    r"^all\s+n=\s*(?P<n>\d+)\s+win=\s*(?P<win>[-+\d.]+|nan)%"
    r"\s+avg=\s*(?P<avg>[-+\d.]+|nan)%"
    r"\s+excess=\s*(?P<excess>[-+\d.]+|nan)%"
    r"\s+PF=\s*(?P<pf>[-+\d.]+|nan|inf)",
    re.M)

FAILS = []
NOTES = []


def _merge(base, extra):
    """base flags minus any flag `extra` also sets, so a case can override the
    baseline without passing the same flag twice.

    This is not hygiene, it is a bug this file already had: `arg()` is
    `next(a for a in args if a.startswith(f"--{name}="))`, so it takes the
    FIRST occurrence and silently drops the rest. `--months=24 --months=6` runs
    24 months, both sides of the case measure the same thing, and --months looks
    stone dead. That is the same "reporting something it did not do" shape the
    gate exists to catch -- pointed at the gate itself.
    """
    names = {a.split("=", 1)[0] for a in extra if a.startswith("--")}
    return [a for a in base if a.split("=", 1)[0] not in names] + list(extra)


def run(base, extra, verbose=False):
    """Drive the real CLI and parse the 'all' row. Returns a metrics dict.

    Never swallows a non-zero exit: a tool that CRASHED on a flag and a tool
    that IGNORED it both produce 'no delta' if you only look at the number, and
    they are opposite bugs. Bare `except: continue` over a loop like this has
    hidden a real error every single time it was examined in this repo.
    """
    cmd = [PY, "-W", "ignore", TOOL, *_merge(base, extra)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if verbose:
        print(f"      $ {' '.join(cmd[3:])}   ({dt:.1f}s)")
    if p.returncode != 0:
        raise RuntimeError(
            f"exit {p.returncode} for {' '.join(extra) or '<baseline>'}\n"
            f"{(p.stderr or p.stdout).strip()[:400]}")
    m = RESULT_RE.search(p.stdout)
    if not m:
        raise RuntimeError(
            f"no 'all' row for {' '.join(extra) or '<baseline>'} -- the tool "
            f"ran but printed nothing parseable:\n{p.stdout.strip()[-400:]}")
    d = {k: float(v) for k, v in m.groupdict().items()}
    d["n"] = int(d["n"])
    if verbose:
        print(f"        -> n={d['n']} win={d['win']:.1f}% avg={d['avg']:+.2f}%")
    return d


# ── cases ────────────────────────────────────────────────────────────────────
# (flag_label, variant_args, check(base, var) -> (ok, detail), why)
#
# `why` is the mechanism that makes the flag reachable on THIS baseline. If a
# case fails, re-read the why before filing anything.

def case_costbps(b, v):
    """--cost-bps=100 is 100 basis points = 1.00% round-trip off EVERY trade.
    n must not move (cost changes the return, not whether the trade fired) and
    avg must drop by ~1.00%. This is the strongest case in the file: it pins a
    QUANTITY, so it fails if the flag is read but applied at the wrong scale
    (bps-vs-percent is a 100x error that 'the number moved' would bless)."""
    if v["n"] != b["n"]:
        return False, (f"cost changed the TRADE COUNT {b['n']}->{v['n']}; a "
                       f"haircut must not change whether an entry fired")
    drop = b["avg"] - v["avg"]
    ok = 0.85 <= drop <= 1.15
    return ok, (f"avg {b['avg']:+.2f}% -> {v['avg']:+.2f}% = {drop:.2f}% drop; "
                f"100bps should be ~1.00% (0.85-1.15 allowed)")


def case_gate_off(b, v):
    """--gate defaults to sma50_over_200@1d and is ON. Turning it off must let
    MORE trades through -- the gate can only ever remove entries. The docstring
    quantifies it on this very signal: n=4990 ungated vs 3017 gated."""
    return v["n"] > b["n"], (f"n {b['n']} (gated) -> {v['n']} (ungated); "
                             f"ungated must be strictly larger")


def case_symbols(b, v):
    """The one that was DEAD: --symbols parsed into a variable a swallowed
    NameError kept anyone from reading, so every run used all 513 symbols and
    the header cheerfully printed the restriction. One symbol must collapse n."""
    return v["n"] < b["n"] / 10, (f"n {b['n']} (513 syms) -> {v['n']} (1 sym); "
                                  f"expected a collapse, got {v['n'] / max(b['n'], 1):.1%}")


def case_months(b, v):
    """A 6-month window is a subset of a 24-month one, so it must hold strictly
    fewer trades. Catches a --months that is parsed but not applied to the
    cutoff -- which would silently make every 'quick' run a full-history run."""
    return v["n"] < b["n"], (f"n {b['n']} (24mo) -> {v['n']} (6mo); "
                             f"a shorter window cannot hold more trades")


def case_filter(b, v):
    """A regime filter can only remove entries.

    THE COLUMN CHOICE IS THE WHOLE CASE. Measured over 40 symbols of 1d, on the
    50,228 bars where heavy_buy fires:
        in_up_wave      100.0% overlap  -> IMPLIED, cannot ever remove a trade
        buy_dominant    100.0% overlap  -> IMPLIED (the known one)
        above_50ma       68.9% overlap  -> partial, so it can actually bite
        sma50_over_200   64.6% overlap  -> partial, but it is the default GATE
                                           column, so a filter case on it would
                                           be confounded with --gate
    above_50ma is the only clean probe. This file first shipped with in_up_wave
    and a comment asserting it was not implied; it is, 100.0%, and the case
    "failed" by being unable to pass. Verify the innocent explanation first."""
    return v["n"] < b["n"], (f"n {b['n']} (no filter) -> {v['n']} (above_50ma, "
                             f"68.9% overlap); a filter cannot add trades")


def case_stop(b, v):
    """A 2% stop is inside the noise of a daily bar; a 10% stop is not. Tighter
    must stop out more often, so win% must fall. Direction only -- the size of
    the fall is a market fact, not a wiring fact."""
    return v["win"] < b["win"], (f"win {b['win']:.1f}% (10% stop) -> "
                                 f"{v['win']:.1f}% (2% stop); tighter must win less")


def case_target(b, v):
    """A 2% target is hit far more often than a 10% one, so win% must rise.
    (This is exactly why a target caps winners -- but that is Paul's call to
    make, not this file's. Here it is only a wiring probe.)"""
    return v["win"] > b["win"], (f"win {b['win']:.1f}% (10% tgt) -> "
                                 f"{v['win']:.1f}% (2% tgt); easier must win more")


def case_window(b, v):
    """--window is the confluence lookback. With --min-count=2 it becomes
    load-bearing: a 1-bar window demands both signals on the SAME bar, a 20-bar
    window accepts them 20 apart, so n must rise with the window. At the default
    min-count=1 this flag is INERT BY DESIGN (one signal needs no window), which
    is why both variants here carry --min-count=2."""
    return v["n"] > b["n"], (f"n {b['n']} (window=1) -> {v['n']} (window=20), "
                             f"both at min-count=2; a wider window must admit more")


def case_mincount(b, v):
    """Requiring 2 distinct signals instead of 1 must cut n. Needs a real second
    column or the case is vacuous, so both runs use two entry signals."""
    return v["n"] < b["n"], (f"n {b['n']} (min-count=1) -> {v['n']} "
                             f"(min-count=2); more required signals means fewer")


# ── the re-entry rule that both cases below turn on ──────────────────────────
# study._simulate ends every trade with `i = exit_i + 1  # no overlapping
# positions`: the scan resumes AFTER the exit bar, one position per symbol at a
# time. So ANY exit path that fires earlier hands the remaining bars back to the
# entry scan and the trade count RISES.
#
# This is the opposite of the intuition it replaced. Both cases here originally
# asserted "an exit cannot change the entry count", which read as obviously true
# and is obviously false the moment you read line 104. n went 4695 -> 10834 and
# looked like a serious bug; it is the design, working.

def case_exit(b, v):
    """--exit adds a bearish-reversal exit. It was dead once before, so it needs
    a probe. Per the re-entry rule above, an extra exit path ends trades sooner
    and therefore ADMITS MORE TRADES -- n must rise. An --exit that is parsed
    but never read leaves n exactly equal, which is the dead-flag signature."""
    return v["n"] > b["n"], (f"n {b['n']} (no exit) -> {v['n']} "
                             f"(wt_cross_down); earlier exits free the symbol "
                             f"to re-enter, so n must rise")


def case_hold(b, v):
    """--hold=5 is a TIME CLOCK and defaults to 0 on purpose ("never sell for no
    reason just because days went by"). This case does not endorse it -- it only
    proves the knob is live, because --hold=N exists solely to TEST a time exit
    as a value. A 5-bar cap ends trades early, so by the re-entry rule n rises."""
    return v["n"] > b["n"], (f"n {b['n']} (hold=0) -> {v['n']} (hold=5); "
                             f"a 5-bar cap ends trades early, freeing the "
                             f"symbol to re-enter, so n must rise")


CASES = [
    # label,      baseline_extra,                variant_extra,                          check
    ("cost-bps",  ["--cost-bps=0"],              ["--cost-bps=100"],                     case_costbps),
    ("gate",      [],                            ["--gate=none"],                        case_gate_off),
    ("symbols",   [],                            ["--symbols=AAPL"],                     case_symbols),
    ("months",    [],                            ["--months=6"],                         case_months),
    ("filter",    [],                            ["--filter=above_50ma"],                case_filter),
    ("stop",      ["--stop=0.10"],               ["--stop=0.02"],                        case_stop),
    ("target",    ["--target=0.10"],             ["--target=0.02"],                      case_target),
    ("exit",      [],                            ["--exit=wt_cross_down"],               case_exit),
    ("hold",      [],                            ["--hold=5"],                           case_hold),
]

# Cases needing a different entry set than BASE's single heavy_buy.
TWO_SIG = "heavy_buy,tdi_long"
CASES_2SIG = [
    ("min-count", ["--min-count=1"],             ["--min-count=2"],                      case_mincount),
    ("window",    ["--min-count=2", "--window=1"], ["--min-count=2", "--window=20"],     case_window),
]


def main():
    args = sys.argv[1:]
    verbose = "--verbose" in args
    only = next((a.split("=", 1)[1] for a in args if a.startswith("--only=")),
                None)

    print(f"wiring gate: {' '.join(BASE)} (all stocks)\n")
    t0 = time.time()

    todo = [(lbl, BASE[:1], be, ve, fn) for lbl, be, ve, fn in CASES]
    todo += [(lbl, [TWO_SIG], be, ve, fn) for lbl, be, ve, fn in CASES_2SIG]
    if only:
        todo = [t for t in todo if t[0] == only]
        if not todo:
            print(f"no case named {only!r}; have: "
                  + ", ".join(t[0] for t in todo or CASES))
            return 2

    for label, entry, base_extra, var_extra, check in todo:
        base = entry + BASE[1:]      # swap entry columns for the 2-signal cases
        try:
            b = run(base, base_extra, verbose)
            v = run(base, var_extra, verbose)
            ok, detail = check(b, v)
        except RuntimeError as e:
            ok, detail = False, f"RUN ERROR: {e}"
        print(f"  {'PASS' if ok else 'FAIL'}  --{label}")
        print(f"        {detail}")
        if not ok:
            FAILS.append(label)

    dt = time.time() - t0
    print(f"\n{len(todo)} flags probed in {dt:.0f}s")
    if FAILS:
        print(f"{len(FAILS)} FLAGS DID NOT DO WHAT THEY CLAIM: "
              + ", ".join(FAILS))
        print("Before filing: re-read each case's `why`. A flag that cannot "
              "move the number on THIS baseline is not necessarily dead.")
    else:
        print("ALL FLAGS MOVE THE NUMBER IN THE DIRECTION THEY CLAIM")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
