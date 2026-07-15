#!/usr/bin/env python3
"""Trust tests for the grid disk cache (portfolio_multi). The cache only pays
off if a hit is INDISTINGUISHABLE from a fresh build, so these check two
things: (1) save->load reproduces every array byte-for-byte with dtypes and
python-str symbols intact; (2) the cache key changes iff a build input
changes. No DB needed. Run: python test_gridcache.py (exit 0 = all pass)."""

import ast
import inspect
import os
import sys
import tempfile

import numpy as np

import portfolio_multi as pm

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def synthetic_grid(T=40, S=5, n_strat=3):
    """A grid tuple shaped exactly like build_grid's return, filled with
    distinctive values so a lossy round-trip would show up."""
    rng = np.random.default_rng(0)
    A = {k: rng.random((T, S)) for k in
         ("O", "H", "L", "C", "SCORE", "ATR", "SW")}
    V = rng.random((T, S)) > 0.3
    ENT = rng.random((T, S)) > 0.8
    SIDX = rng.integers(0, n_strat, (T, S)).astype(np.int64)
    EXT = rng.random((T, S)) > 0.9
    syms = [f"SYM{i}" for i in range(S)]
    grid = np.arange("2020-01-01", T, dtype="datetime64[D]").astype(
        "datetime64[ns]")
    st_stop = rng.random(n_strat)
    st_hold = rng.integers(1, 50, n_strat).astype(np.int64)
    st_tgt = rng.random(n_strat)
    return (A, V, ENT, SIDX, EXT, syms, grid, st_stop, st_hold, st_tgt)


# 1. round-trip fidelity ------------------------------------------------------
g = synthetic_grid()
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "grid_15m_123_abc.npz")
    pm.GRID_CACHE_DIR = d
    pm._save_grid(path, g)
    check("save is atomic (no leftover .tmp)",
          not os.path.exists(path + ".tmp.npz"))
    loaded = pm._load_grid(path)

A0, V0, ENT0, SIDX0, EXT0, syms0, grid0, ss0, sh0, sg0 = g
A1, V1, ENT1, SIDX1, EXT1, syms1, grid1, ss1, sh1, sg1 = loaded

check("A dict keys preserved", set(A0) == set(A1))
check("A arrays identical",
      all(np.array_equal(A0[k], A1[k]) for k in A0))
check("A dtype preserved (float64)",
      all(A1[k].dtype == A0[k].dtype for k in A0))
check("V bool identical", np.array_equal(V0, V1) and V1.dtype == bool)
check("ENT bool identical", np.array_equal(ENT0, ENT1) and ENT1.dtype == bool)
check("EXT bool identical", np.array_equal(EXT0, EXT1) and EXT1.dtype == bool)
check("SIDX int64 identical",
      np.array_equal(SIDX0, SIDX1) and SIDX1.dtype == np.int64)
check("syms round-trip to python str",
      syms0 == syms1 and all(type(s) is str for s in syms1),
      f"{[type(s).__name__ for s in syms1]}")
check("grid datetime64 identical",
      np.array_equal(grid0, grid1) and grid1.dtype == grid0.dtype)
check("st_stop identical", np.array_equal(ss0, ss1))
check("st_hold int64 identical",
      np.array_equal(sh0, sh1) and sh1.dtype == np.int64)
check("st_tgt identical", np.array_equal(sg0, sg1))


# 2. cache-key determinism ----------------------------------------------------
strategies = [{"name": "a", "entry_cols": ["x"], "window": 5, "stop_pct": 0.05}]
base = dict(strategies=strategies, interval="15m",
            gate_arg="minervini@1d", market="none", cutoff=None,
            exit_cols=[], gstop=None, ghold=None, gtarget=None,
            atr_len=14, swing_look=20, db_mtime=111)
p_base = pm._grid_cache_path(**base)

check("identical inputs -> identical path",
      pm._grid_cache_path(**base) == p_base)


def changed(**over):
    return pm._grid_cache_path(**{**base, **over})


check("strategy change -> different path",
      changed(strategies=[{**strategies[0], "window": 6}]) != p_base)
check("gate change -> different path", changed(gate_arg="none") != p_base)
check("market change -> different path", changed(market="sma100") != p_base)
check("cutoff change -> different path",
      changed(cutoff="2024-01-01") != p_base)
check("atr_len change -> different path", changed(atr_len=20) != p_base)
check("swing_look change -> different path", changed(swing_look=10) != p_base)
check("exit_cols change -> different path",
      changed(exit_cols=["wt_cross_down"]) != p_base)
check("db_mtime change -> different path (and prunable token)",
      changed(db_mtime=222) != p_base
      and "_222_" in os.path.basename(changed(db_mtime=222)))


# 3. code-signature invalidation ----------------------------------------------
# The guard these tests exist for. Every check above varies an INPUT; none
# varies the CODE, which is how a fixed rising() and a stale cache coexisted.
from weisswave import codesig  # noqa: E402

FAKE = {
    "signals": b"from .core import ema\nfrom .rci import rci\n",
    "core": b"def ema(x): return x\n",
    "rci": b"def rci(x): return x\n",
    "unrelated": b"def nobody_imports_me(): pass\n",
}


def fake_sig(*roots, **over):
    src = {**FAKE, **over}
    codesig.package_sig.cache_clear()
    real = codesig._module_src
    codesig._module_src = lambda n: src.get(n)
    try:
        return codesig.package_sig(*roots)
    finally:
        codesig._module_src = real
        codesig.package_sig.cache_clear()


s_base = fake_sig("signals")
check("identical source -> identical sig", fake_sig("signals") == s_base)

# the rising() shape: same name, same signature, same columns, different math
check("value-only edit to a reached module -> different sig",
      fake_sig("signals", core=b"def ema(x): return x * 2\n") != s_base)
check("edit to a TRANSITIVELY reached module -> different sig",
      fake_sig("signals", rci=b"def rci(x): return x + 1\n") != s_base)
check("edit to an unreached module -> SAME sig (no false rebuilds)",
      fake_sig("signals", unrelated=b"def nobody(): return 99\n") == s_base)
check("root itself is hashed",
      fake_sig("signals", signals=b"from .core import ema\n") != s_base)
check("optional/absent module is not fatal",
      isinstance(fake_sig("signals", signals=b"from .missing import x\n"), str))

# nested imports: signals.py really does import .combined inside a try/except,
# so a module-level-only scan would drop it and miss the whole suite
check("import nested in a try/except is still reached",
      fake_sig("signals",
               signals=b"try:\n    from .core import ema\nexcept ImportError:\n    ema = None\n",
               core=b"def ema(x): return 1\n")
      != fake_sig("signals",
                  signals=b"try:\n    from .core import ema\nexcept ImportError:\n    ema = None\n",
                  core=b"def ema(x): return 2\n"))

# and the real graph, not just the fake one: the modules the bug lived in must
# actually be covered, or all of the above is theatre
def reached(*roots):
    seen, stack = set(), list(roots)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        src = codesig._module_src(n)
        if src is None:
            continue
        seen.add(n)
        stack.extend(codesig._relative_imports(ast.parse(src, filename=n)))
    return seen


_sig_graph = reached("signals")
check("real signals graph reaches core (where rising lives)",
      "core" in _sig_graph)
check("real signals graph reaches weiswave (the wave engine)",
      "weiswave" in _sig_graph)
check("real grid graph reaches factors and structure",
      {"factors", "structure"} <= reached("signals", "factors", "structure"))
check("grid cache key actually carries the code signature",
      '"code"' in inspect.getsource(pm._grid_cache_path))


if __name__ == "__main__":
    print("\n" + ("ALL GRID-CACHE TRUST TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
