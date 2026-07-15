#!/usr/bin/env python3
"""The search space as DATA, not literals.

Every bound, default, probability and sim constant the search uses is loaded
from search_space.json (or a copy an agent points at with --space=FILE) so that
what is discoverable is a decision you can read, diff and change -- rather than
a number a coding session picked and buried in a function call.

Two things worth knowing:

1. If the search cannot sample it, it cannot find it, AND it will not tell you
   it was looking through a keyhole. Widening a bound is a real decision.

2. Thresholds are sampled RELATIVE by default. Weights and thresholds used to
   be drawn independently, so a config could draw muted weights (capping the
   score it could ever produce near zero) and then draw a threshold that score
   could never reach -- rejecting every bar, forever, and scoring a deceptively
   neutral-looking 0.0. 27% of the accumulated store is that bug. Sampling a
   threshold as a fraction of the achievable score makes it reachable by
   construction.

ASCII only.
"""

import copy
import json
import os

import numpy as np

# The active space file. WEISSWAVE_SPACE lets it be chosen BEFORE import, which
# matters because portfolio_multi builds FACTOR_NAMES from the factor block at
# import time -- a --space= parsed later in main() would arrive too late to
# change the factor stack. bootstrap_space() (below) sets it from argv early;
# child processes inherit it through the environment, so pool workers agree.
DEFAULT_SPACE = os.environ.get("WEISSWAVE_SPACE", "search_space.json")


def bootstrap_space(argv):
    """Call BEFORE importing anything that reads the factor spec (i.e. before
    `from portfolio_multi import ...`). Promotes --space=FILE into the
    environment so the factor stack is built from the right file, in this
    process and in every worker it spawns."""
    global DEFAULT_SPACE
    for a in argv:
        if a.startswith("--space="):
            path = a.split("=", 1)[1]
            os.environ["WEISSWAVE_SPACE"] = path
            DEFAULT_SPACE = path
            return path
    return DEFAULT_SPACE


def load_space(path=None, overrides=None):
    """Load the spec; `overrides` is a dotted-key dict e.g.
    {"grid.interval": "5m", "space.weights.hi": 6.0} so a CLI flag or an agent
    can bend one knob without editing the file.

    `path=None` resolves DEFAULT_SPACE at CALL time, not at def time -- a
    default argument would bind the value when this function was defined, i.e.
    before bootstrap_space() could change it."""
    path = path or DEFAULT_SPACE
    with open(path, encoding="utf-8") as f:
        sp = json.load(f)
    for k, v in (overrides or {}).items():
        node, *rest = k.split(".")
        if not rest:                     # a top-level key, e.g. --set=_README=x
            sp[node] = v
            continue
        d = sp.setdefault(node, {})
        for p in rest[:-1]:
            d = d.setdefault(p, {})
        d[rest[-1]] = v
    return sp


def strip_docs(o):
    """Drop _-prefixed documentation keys at EVERY depth.

    The space file carries prose for the reader (and for the strategist model)
    inline with the values. Anything that feeds those values to real code has to
    remove it FIRST, or the prose arrives as a keyword argument.

    This lived as a closure inside space_sig(); agent_search needed the same
    thing, could not reach it, and reimplemented it as a flat one-level
    comprehension. That shipped `signals.combined.tdi._oversold_doc` straight
    into tdi_signals() as a kwarg -- TypeError on every symbol, swallowed by a
    bare except, surfaced as "no usable 1d data in market.duckdb". A docstring
    in a JSON file blamed the database. One implementation, exported, so the
    next caller cannot get it subtly wrong.
    """
    if isinstance(o, dict):
        return {k: strip_docs(v) for k, v in o.items()
                if not k.startswith("_")}
    if isinstance(o, list):
        return [strip_docs(v) for v in o]
    return o


def space_sig(sp):
    """A stable signature of the space, minus the _doc noise. Recorded on every
    results row so a survivor is always traceable to the bar it cleared."""
    return json.dumps(strip_docs(sp), sort_keys=True, default=str)


def _draw(r, d, n=None):
    """One draw from a distribution spec."""
    dist = d.get("dist", "uniform")
    if dist == "choice":
        v = r.choice(d["values"], size=n)
        return v
    v = r.uniform(d.get("lo", 0.0), d.get("hi", 1.0), size=n)
    if "round" in d:
        v = np.round(v, int(d["round"]))
    return v


def sample_cfg(r, sp, K, htf_start, fscale=None, fsample=None):
    """Draw one random config from the space.

    THRESHOLD SAMPLING IS THE PART THAT MATTERS. With dist='relative', thr is a
    multiple of the score the drawn weights can TYPICALLY produce:

        typical_score = sum(w_k * fscale_k)   where fscale_k = mean|factor_k|

    `fscale` is what makes this honest. Without it we assume every factor
    reaches ~1 -- true for `trend` (fires on 100% of bars), wildly false for
    `adp_bull_div` (0.16%). Adding sparse indicator factors made sum-of-weights
    overestimate the achievable score ~4x, so thresholds landed above anything a
    config could hit and 7 of 12 configs never traded. Fall back to 1.0 per
    factor only when no scale is supplied (old grids)."""
    s = sp["space"]
    wd = s["weights"]
    w = np.round(r.uniform(wd.get("lo", 0.0), wd.get("hi", 3.0), K), 2)
    w[r.random(K) < wd.get("mute_prob", 0.25)] = 0.0
    fs = np.ones(K) if fscale is None else np.asarray(fscale, float)

    def thresh(key, typical, sample=None):
        d = s[key]
        dist = d.get("dist")
        if dist == "percentile" and (sample is None or not len(sample)):
            # No FSAMPLE (a pre-v13 grid, or a caller with no data). lo/hi are
            # PERCENTILES here (50..99.5) -- returning one as an absolute
            # threshold would be unreachable by construction, i.e. the exact bug
            # this whole mechanism exists to prevent. Degrade to a bounded
            # fraction of the typical score instead.
            q = r.uniform(d.get("lo", 50.0), d.get("hi", 99.5))
            return round(float((q / 100.0) * max(typical, 1e-9)), 4)
        if dist == "percentile" and sample is not None and len(sample):
            # THE HONEST ONE: the threshold is a percentile of the scores THIS
            # config's own weights actually produce on real bars. q=99 means
            # "enter on the top 1% of bars by confluence" -- which is what a
            # threshold is FOR, and it cannot become unreachable no matter how
            # the factor stack changes.
            q = r.uniform(d.get("lo", 50.0), d.get("hi", 99.5))
            return round(float(np.percentile(sample, q)), 4)
        if dist == "relative":
            frac = r.uniform(d.get("lo", -0.1), d.get("hi", 0.8))
            return round(float(frac * max(typical, 1e-9)), 3)
        return round(float(r.uniform(d.get("lo", 0.0), d.get("hi", 1.0))), 3)

    es = hs = None
    if fsample is not None and len(fsample):
        es = np.asarray(fsample)[:, :htf_start] @ w[:htf_start]
        hs = np.asarray(fsample)[:, htf_start:] @ w[htf_start:]
    cfg = {"w": w,
           "thr": thresh("thr", float((w[:htf_start] * fs[:htf_start]).sum()),
                         es),
           "htf": thresh("htf", float((w[htf_start:] * fs[htf_start:]).sum()),
                         hs),
           "stop": str(_draw(r, s["stop"])),
           "trail": str(_draw(r, s["trail"])),
           "fr": round(float(_draw(r, s["fr"])), 3),
           "ta": round(float(_draw(r, s["ta"])), 3),
           "td": round(float(_draw(r, s["td"])), 3),
           "tgt": float(_draw(r, s["tgt"])),
           "mp": int(_draw(r, s["mp"]))}
    return cfg


def mutate_cfg(c, r, sp, K):
    """Perturb an elite config using the mutation rates from the space."""
    m = sp["mutate"]
    s = sp["space"]
    n = {**c, "w": c["w"].copy()}
    for i in range(K):
        if r.random() < m.get("weight_prob", 0.25):
            n["w"][i] = round(max(0.0, c["w"][i]
                                  + r.normal(0, m.get("weight_sigma", 0.8))), 2)
    # Thresholds are perturbed PROPORTIONALLY, not by a fixed step. thr scales
    # with the achievable score (~10 with the current factor stack, ~2 with a
    # smaller one), so an absolute sigma of 0.4 is either a huge jump or a
    # rounding error depending on how many factors exist. A fraction is
    # scale-free and survives adding factors.
    if r.random() < m.get("thr_prob", 0.4):
        n["thr"] = round(max(0.0, c["thr"]
                             * (1.0 + r.normal(0, m.get("thr_sigma", 0.25)))),
                         3)
    if r.random() < m.get("htf_prob", 0.4):
        n["htf"] = round(c["htf"]
                         * (1.0 + r.normal(0, m.get("htf_sigma", 0.25))), 3)
    if r.random() < m.get("stop_prob", 0.2):
        n["stop"] = str(_draw(r, s["stop"]))
    if r.random() < m.get("trail_prob", 0.2):
        n["trail"] = str(_draw(r, s["trail"]))
    if r.random() < m.get("td_prob", 0.3):
        n["td"] = round(min(m.get("td_hi", 0.2),
                            max(m.get("td_lo", 0.02),
                                c["td"] + r.normal(0, m.get("td_sigma", 0.03)))
                            ), 3)
    return n


def parse_set_args(args):
    """--set=grid.interval=5m --set=space.weights.hi=6 -> dotted override dict.
    Values are JSON-parsed when possible so numbers/bools/lists work."""
    out = {}
    for a in args:
        if not a.startswith("--set="):
            continue
        body = a[len("--set="):]
        if "=" not in body:
            continue
        k, v = body.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out
