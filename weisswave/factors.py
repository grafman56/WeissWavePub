#!/usr/bin/env python3
"""FACTORS AS DATA -- setups defined in JSON, not Python.

Adding "sell pressure is fading" used to mean editing build_grid, appending to
FACTOR_NAMES and bumping a schema version. That is a code change, which means
only a programmer can add a setup, which means every factor that exists is one
somebody chose to write. This module makes a factor a JSON object instead:

    "sell_exhaustion": {"op": "falling", "src": "volumedn", "bars": 5}
    "holds_200ema":    {"op": "dist_above", "src": "Close", "ref": "ema200"}
    "buy_surge":       {"op": "cross_up", "a": "volumeup", "b": "volumedn"}

A SETUP IS THEN A WEIGHT VECTOR, NOT CODE. "Catching the knife" = nonzero
weights on those three. The reversal-at-a-fib = prox + wt_bear_div + stall. The
coil = hh_hl + holds_200ema + fails_to_break. Same machinery, different JSON --
so Paul, or a local agent, or eventually an end user can define a setup without
touching the engine.

TWO RULES THIS MODULE ENFORCES:

1. NEVER eval(). Every op is a fixed function in OPS, and every parameter is
   validated against a whitelist. A strategy file is untrusted input the moment
   anyone but the author can supply one; "just let them write an expression"
   turns a JSON upload into remote code execution.

2. SIGN IS EXPLICIT. Factors are signed ~[-1, 1] where + argues for a long, and
   weights are sampled non-negative, so the DEFINITION must say what it means.
   Falling sell-volume is bullish; falling price is not -- the op cannot know
   which you meant. Say it with "sign": -1. (Allowing negative weights would let
   the search discover the sense instead, but that doubles the space to
   rediscover something you already know.)

NO LOOKAHEAD: every op reads bars <= t (shift/trailing rolling only).
ASCII only.
"""

import numpy as np
import pandas as pd

EPS = 1e-12


class FactorError(ValueError):
    """A factor definition that cannot be compiled. The message is meant to be
    read by whoever wrote the JSON -- not a stack trace."""


def _series(sig, ref, what):
    """Resolve a src/ref: a column in the signal frame, or a constant."""
    if isinstance(ref, bool):
        raise FactorError(f"{what}: expected a column name or number, got a "
                          f"bool")
    if isinstance(ref, (int, float)):
        return pd.Series(float(ref), index=sig.index)
    if not isinstance(ref, str):
        raise FactorError(f"{what}: expected a column name or number, got "
                          f"{type(ref).__name__}")
    if ref not in sig.columns:
        raise FactorError(f"{what}: unknown column {ref!r}")
    return sig[ref].astype(float)


def _num(p, key, default):
    v = p.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise FactorError(f"{key!r} must be a number, got {v!r}")
    return v


# ---- ops ---------------------------------------------------------------------
# Each takes (signal_frame, params) -> pd.Series. Sign/clip applied by compile().

def _op_falling(sig, p):
    """Relative decline of `src` over `bars`. 1.0 = fell to zero.
    e.g. sell pressure exhausting: {"op":"falling","src":"volumedn","bars":5}"""
    s = _series(sig, p["src"], "src")
    n = int(_num(p, "bars", 5))
    prev = s.shift(n)
    return ((prev - s) / (prev.abs() + EPS)).clip(0.0, 1.0)


def _op_rising(sig, p):
    """Relative rise of `src` over `bars`. 1.0 = doubled."""
    s = _series(sig, p["src"], "src")
    n = int(_num(p, "bars", 5))
    prev = s.shift(n)
    return ((s - prev) / (prev.abs() + EPS)).clip(0.0, 1.0)


def _op_cross_up(sig, p):
    """`a` crossed above `b` within the last `within` bars.
    e.g. buy pressure taking over: a=volumeup, b=volumedn."""
    a = _series(sig, p["a"], "a")
    b = _series(sig, p["b"], "b")
    w = max(1, int(_num(p, "within", 1)))
    x = (a > b) & (a.shift(1) <= b.shift(1))
    return x.astype(float).rolling(w, min_periods=1).max()


def _op_cross_down(sig, p):
    a = _series(sig, p["a"], "a")
    b = _series(sig, p["b"], "b")
    w = max(1, int(_num(p, "within", 1)))
    x = (a < b) & (a.shift(1) >= b.shift(1))
    return x.astype(float).rolling(w, min_periods=1).max()


def _op_dist_above(sig, p):
    """Signed distance of `src` above `ref`, scaled. +1 = `scale` above.
    e.g. holding the 200 EMA: src=Close, ref=ema200, scale=0.02."""
    s = _series(sig, p["src"], "src")
    r = _series(sig, p["ref"], "ref")
    sc = _num(p, "scale", 0.05)
    return ((s - r) / (r.abs() * sc + EPS)).clip(-1.0, 1.0)


def _op_prox(sig, p):
    """Closeness of `src` to `ref`: 1 = on it, 0 = `band` away or further.
    e.g. price at a fib level: src=Close, ref=fib_p2, band=0.05."""
    s = _series(sig, p["src"], "src")
    r = _series(sig, p["ref"], "ref")
    band = _num(p, "band", 0.05)
    return (1.0 - (s - r).abs() / (r.abs() * band + EPS)).clip(0.0, 1.0)


def _op_stall(sig, p):
    """`src` has gone nowhere over `bars`. 1 = dead flat, 0 = moved `scale`.
    e.g. price teetering at a level instead of pushing through.

    NET DISPLACEMENT ONLY -- there is no path term here, and that is a real
    limit, not an oversight to fix in place. `stall` cannot tell "price sat
    still" (no travel, no result) from "price churned violently and came back"
    (huge travel, no result): both are dead flat over `bars`. They are opposite
    situations. `churn` below is the path-aware sibling; use it when the
    distinction matters and leave this one alone when it does not.
    """
    s = _series(sig, p["src"], "src")
    n = int(_num(p, "bars", 5))
    sc = _num(p, "scale", 0.02)
    move = (s - s.shift(n)).abs() / (s.abs() * sc + EPS)
    return (1.0 - move).clip(0.0, 1.0)


def _op_churn(sig, p):
    """How much of `src`'s travel over `bars` was WASTED: 1 - |net| / path.
    0 = a straight line (every step went somewhere). 1 = travelled and ended
    where it started.

    THE QUANTITY BEHIND "BUILDS BEFORE MOVES". Paul gave five dated ranges he
    reads as builds; measured at each range's own length, this ranks them
    1.7th / 6.2th / 13.6th / 13.8th percentile -- and puts the one he called "a
    different type" at 46.7th, correctly excluding it. It reproduces his own
    strong/weak/different labels without being told them.

    IT IS THE RATIO, NOT THE VOLUME. Relative volume on those builds is 44th-
    57th percentile, i.e. ordinary. The effort/result read is right but it is
    driven entirely by the denominator collapsing (no result), not by extra
    effort. And it is not raw wasted travel either: that scales with the window
    and ranks the excluded example highest. Only the ratio separates them.

    `bars` IS THE WHOLE POINT AND MUST STAY DATA. Paul's builds run 8, 15, 44,
    64 and 89 bars -- an 11x spread. At a fixed 20 the same examples read
    85th/64th/93rd percentile, which is noise. The window is a SEARCH OVER
    SCALES, the same way his fib anchors are ("I usually try several different
    past swings to find what works"). A frozen default here would rebuild
    RNG_LOOK=20, which is the exact class of constant that keeps biting.

    `min_path` is a degeneracy guard, not a threshold on the idea: with no
    travel there is no wasted motion to measure and net/path is 0/0. A series
    that never moved would otherwise score 1.0 -- maximum build -- which is the
    same "dead vs coiling" confusion `stall` has, inverted.

    {"op":"churn","src":"Close","bars":20}
    """
    s = _series(sig, p["src"], "src")
    n = max(1, int(_num(p, "bars", 20)))
    base = s.shift(n).abs()
    net = (s - s.shift(n)).abs() / (base + EPS)
    path = s.diff().abs().rolling(n, min_periods=n).sum() / (base + EPS)
    v = (1.0 - net / (path + EPS)).clip(0.0, 1.0)
    return v.where(path > _num(p, "min_path", 0.01), 0.0)


def _op_spread(sig, p):
    """Signed distance between `a` and `b`, squashed onto ~[-1,1] by `scale`.
    +-scale maps to +-0.76 (tanh(1)). `a` or `b` may be a constant.

    THE OSCILLATOR AS A STATE INSTEAD OF AN EVENT. Paul: "maybe we can use the
    wavetrend process a different way than just firing them off like indicators,
    because thats what the indicator part already does." He is right: 14 of the
    16 wt_* columns are BOOLEAN. Only wt1 and wt2 are continuous and nothing
    weighs either. Measured over 226,305 bars of 1d, `wt_oversold` (wt2 <= -60)
    is 15,615 bars whose median wt2 is -68 and whose deepest is -322 -- a 5x
    range handed to the search as ONE BIT. wt1-wt2 spans +-630 and only its SIGN
    CHANGE survives, as wt_cross_up; the magnitude is the force behind the cross
    and is discarded.

    TANH, NOT CLIP. Clipping would map -60 and -322 to the same -1.0, which is
    the exact information loss this op exists to undo. tanh is monotonic
    everywhere, so order is preserved out into the tail; it only compresses.

    `sign` IS LEGITIMATE HERE, unlike on pivot_confirm. There 0 means "no
    candidate" and flipping the sign makes empty bars score highest. Here 0
    means "a is AT b" -- a real midpoint -- so sign=-1 honestly flips the axis:
    with src=wt2, b=0, sign=-1, deep oversold becomes +0.76 and overbought
    -0.76. Whether that is the useful direction is FOR THE SEARCH.

    {"op":"spread","a":"wt2","b":0,"scale":60}      # depth: how stretched
    {"op":"spread","a":"wt1","b":"wt2","scale":20}  # gap: force behind the cross
    """
    a = _series(sig, p["a"], "a")
    b = _series(sig, p["b"], "b")
    sc = abs(_num(p, "scale", 60.0))
    return np.tanh((a - b) / (sc + EPS))


def _op_slope(sig, p):
    """Signed change in `src` over `bars`, squashed onto ~[-1,1] by `scale`.
    Is the oscillator TURNING? A cross at -300 still falling and a cross at -65
    turning up are opposite setups wearing the same boolean.

    Composes with `spread`: depth x slope as two weights is "deep AND turning",
    which is Paul's never-catch-a-falling-knife rule as a dial rather than a law.

    {"op":"slope","src":"wt1","bars":1,"scale":20}
    """
    s = _series(sig, p["src"], "src")
    n = max(1, int(_num(p, "bars", 1)))
    sc = abs(_num(p, "scale", 20.0))
    return np.tanh((s - s.shift(n)) / (sc + EPS))


def _op_hh_hl(sig, p):
    """Higher highs AND higher lows over `bars` -- the coil before expansion.
    A trailing-window proxy for swing structure (no pivot confirmation lag)."""
    n = int(_num(p, "bars", 10))
    hh = sig["High"].astype(float).rolling(n, min_periods=n).max()
    ll = sig["Low"].astype(float).rolling(n, min_periods=n).min()
    return ((hh > hh.shift(n)) & (ll > ll.shift(n))).astype(float)


def _op_fails_to_break(sig, p):
    """`src` pierced `ref` within `bars` but Close is back above it -- the
    absorption read: sellers pushed and could not make it stick."""
    s = _series(sig, p.get("src", "Low"), "src")
    r = _series(sig, p["ref"], "ref")
    n = max(1, int(_num(p, "bars", 3)))
    pierced = (s < r).astype(float).rolling(n, min_periods=1).max() > 0.5
    return (pierced & (sig["Close"].astype(float) > r)).astype(float)


def _op_pivot_confirm(sig, p):
    """How many bars of RIGHT-side evidence the most recent still-unbeaten pivot
    in `src` has survived, as a fraction of `cap`. 0 = nothing live.

    This is Pine's lbR as a WEIGHT instead of a threshold, and that is the whole
    point. lbR ("look back range") is applied to a bar that has already
    happened: pivotlow/pivothigh only DECLARE the pivot lbR bars later, once
    enough right-side bars exist to prove it was an extreme. So lbR is doing two
    jobs that fight each other -- how much evidence you demand, and how late you
    act. Frozen at one number it forces a permanent choice: fire at lbR=1 on a
    pivot that may not be one and eat false positives, or wait for lbR=4 and
    confirm after the move is gone.

    Emitting the evidence instead of thresholding it lets the search price that
    tradeoff per bar, which is the difference between a gate and a dial. A pivot
    with one bar of confirmation still contributes -- it just contributes little,
    so it can clear the entry threshold when other factors agree and cannot when
    it is alone. Earliness becomes a confluence decision rather than a global
    constant nobody revisits.

    A candidate beaten on the right falls to 0 by itself; no special case, the
    evidence simply stops supporting it. On BTC 1d the volumedn peak of
    2025-04-22 ramps 1..6 over the following week and collapses to 0 on 04-29
    when volumedn prints straight through it. It was never a pivot.

    Do NOT reach for sign=-1 to mean "prefer early": 0 means no candidate, not
    maximum earliness, and inverting makes empty bars score highest. Earliness
    is expressed by the WEIGHT and the entry threshold, not by the sign.

    {"op":"pivot_confirm","src":"volumedn","side":"high","lbL":6,"cap":6}
    """
    s = _series(sig, p["src"], "src")
    side = p.get("side", "high")
    if side not in ("high", "low"):
        raise FactorError(f"'side' must be 'high' or 'low', got {side!r}")
    lbL = max(1, int(_num(p, "lbL", 6)))
    cap = max(1, int(_num(p, "cap", 6)))

    out = pd.Series(0.0, index=s.index)
    # Walk k downwards so the SMALLEST surviving k wins: the freshest candidate
    # is the one being judged, not some older pivot still technically unbeaten.
    for k in range(cap, 0, -1):
        centre = s.shift(k)                       # the bar under judgement
        left = s.shift(k + 1).rolling(lbL, min_periods=lbL)
        right = s.rolling(k, min_periods=k)       # exactly the k bars after it
        if side == "high":
            ok = (centre > left.max()) & (centre > right.max())
        else:
            ok = (centre < left.min()) & (centre < right.min())
        out = out.where(~ok.fillna(False), float(k))
    return (out / cap).clip(0.0, 1.0)


def _op_column(sig, p):
    """Pass a signal column straight through (booleans -> 0/1). The escape
    hatch for anything already computed in the signal layer."""
    return _series(sig, p["src"], "src")


OPS = {
    "falling": _op_falling,
    "rising": _op_rising,
    "cross_up": _op_cross_up,
    "cross_down": _op_cross_down,
    "dist_above": _op_dist_above,
    "prox": _op_prox,
    "stall": _op_stall,
    "churn": _op_churn,
    "spread": _op_spread,
    "slope": _op_slope,
    "hh_hl": _op_hh_hl,
    "fails_to_break": _op_fails_to_break,
    "pivot_confirm": _op_pivot_confirm,
    "column": _op_column,
}
REQUIRED = {
    "falling": ("src",), "rising": ("src",),
    "cross_up": ("a", "b"), "cross_down": ("a", "b"),
    "dist_above": ("src", "ref"), "prox": ("src", "ref"),
    "stall": ("src",), "churn": ("src",),
    "spread": ("a", "b"), "slope": ("src",),
    "hh_hl": (), "fails_to_break": ("ref",),
    "pivot_confirm": ("src",),
    "column": ("src",),
}


def validate_spec(factors):
    """Check a factor spec WITHOUT any data. Returns a list of human-readable
    errors (empty = ok) so bad JSON is reported as a message, not a traceback.
    This is the gate any user-supplied strategy file goes through."""
    errs = []
    if not isinstance(factors, dict):
        return ["'factors' must be an object mapping name -> definition"]
    for name, d in factors.items():
        if name.startswith("_"):
            continue                       # doc keys
        if not isinstance(d, dict):
            errs.append(f"{name}: definition must be an object")
            continue
        op = d.get("op")
        if op not in OPS:
            errs.append(f"{name}: unknown op {op!r} (have: "
                        f"{', '.join(sorted(OPS))})")
            continue
        for req in REQUIRED[op]:
            if req not in d:
                errs.append(f"{name}: op {op!r} needs {req!r}")
        sign = d.get("sign", 1)
        if sign not in (1, -1, 1.0, -1.0):
            errs.append(f"{name}: 'sign' must be 1 or -1, got {sign!r}")
    return errs


MISSING = set()      # columns a factor wanted that the signal layer lacks


def compile_factor(sig, name, d, strict=True):
    """One factor definition -> a signed ~[-1,1] array over the frame.

    strict=False zeroes a factor whose column is absent instead of raising, and
    records it in MISSING so the caller can warn ONCE by name. That is for the
    public checkout: search_space.json names indicators from the proprietary
    suite (combined.py) which simply are not there, and a missing indicator
    should not brick the whole backtester.

    A zeroed factor is still a DEAD SEARCH DIMENSION -- it is warned about, not
    hidden. Silence here would be the `dip_bias` mistake again: something that
    claims to be a factor while contributing nothing."""
    op = d.get("op")
    if op not in OPS:
        raise FactorError(f"{name}: unknown op {op!r}")
    for req in REQUIRED[op]:
        if req not in d:
            raise FactorError(f"{name}: op {op!r} needs {req!r}")
    try:
        v = OPS[op](sig, d)
    except FactorError:
        if strict:
            raise
        for k in ("src", "ref", "a", "b"):
            r = d.get(k)
            if isinstance(r, str) and r not in sig.columns:
                MISSING.add(f"{name} (needs '{r}')")
        return np.zeros(len(sig))
    except KeyError as e:
        raise FactorError(f"{name}: missing parameter {e}") from None
    v = pd.Series(v, index=sig.index).astype(float)
    v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (v * float(d.get("sign", 1))).clip(-1.0, 1.0).to_numpy()


def custom_names(factors):
    """The factor names a spec defines, in a STABLE order (doc keys skipped).
    Order matters: it fixes the column layout of the factor stack."""
    return [k for k in factors if not k.startswith("_")]


def compile_all(sig, factors):
    """{name: signed array} for one symbol's signal frame."""
    return {n: compile_factor(sig, n, factors[n]) for n in custom_names(factors)}
