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
    e.g. price teetering at a level instead of pushing through."""
    s = _series(sig, p["src"], "src")
    n = int(_num(p, "bars", 5))
    sc = _num(p, "scale", 0.02)
    move = (s - s.shift(n)).abs() / (s.abs() * sc + EPS)
    return (1.0 - move).clip(0.0, 1.0)


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
    "hh_hl": _op_hh_hl,
    "fails_to_break": _op_fails_to_break,
    "column": _op_column,
}
REQUIRED = {
    "falling": ("src",), "rising": ("src",),
    "cross_up": ("a", "b"), "cross_down": ("a", "b"),
    "dist_above": ("src", "ref"), "prox": ("src", "ref"),
    "stall": ("src",), "hh_hl": (), "fails_to_break": ("ref",),
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


def compile_factor(sig, name, d):
    """One factor definition -> a signed ~[-1,1] array over the frame."""
    op = d.get("op")
    if op not in OPS:
        raise FactorError(f"{name}: unknown op {op!r}")
    for req in REQUIRED[op]:
        if req not in d:
            raise FactorError(f"{name}: op {op!r} needs {req!r}")
    try:
        v = OPS[op](sig, d)
    except FactorError:
        raise
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
