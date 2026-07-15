#!/usr/bin/env python3
"""Trust tests for the grid disk cache (portfolio_multi). The cache only pays
off if a hit is INDISTINGUISHABLE from a fresh build, so these check two
things: (1) save->load reproduces every array byte-for-byte with dtypes and
python-str symbols intact; (2) the cache key changes iff a build input
changes. No DB needed. Run: python test_gridcache.py (exit 0 = all pass)."""

import ast
import inspect
import json
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
            gate_arg="sma50_over_200@1d", market="none", cutoff=None,
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

# ...and carries it IN THE FILENAME, not only inside the hash. A hash is
# one-way: a grid built by dead code is indistinguishable from a live one you
# have not requested yet, so the only safe move is to keep it forever. That is
# how 33 files / 1.55GB accumulated after the minervini rename changed
# package_sig and invalidated every grid with nothing to identify them by.
# db_mtime was already in the name for exactly this reason.
_cur_code = codesig.package_sig("signals", "factors", "structure")
_p = os.path.basename(pm._grid_cache_path(**base))
check("code signature is a FILENAME token, not just a hash input",
      _cur_code in _p, _p)
check("db_mtime is still a filename token", "_111_" in _p, _p)


# 3b. grid-cache pruning ------------------------------------------------------
# The rule must drop data-stale and code-stale grids while KEEPING siblings that
# differ only by the hash -- those are different gates/universes/anchors and are
# every bit as valid. Deleting them is the bug stale_cache_paths already had:
# a crypto run and a stocks run each nuking the other's multi-GB grid and paying
# a full rebuild on every alternation.
def _grid_prune(current, present):
    real_glob = pm.glob.glob
    pm.glob.glob = lambda _pat: [os.path.join(pm.GRID_CACHE_DIR, p)
                                 for p in present]
    try:
        return sorted(os.path.basename(p)
                      for p in pm.stale_grid_paths(
                          os.path.join(pm.GRID_CACHE_DIR, current), "15m"))
    finally:
        pm.glob.glob = real_glob


CUR_G = "grid_15m_1784052866_61ac34d4_aaaaaaaaaaaaaaaa.npz"
SIB_G = "grid_15m_1784052866_61ac34d4_bbbbbbbbbbbbbbbb.npz"  # other gate/universe
OLDDATA_G = "grid_15m_1700000000_61ac34d4_aaaaaaaaaaaaaaaa.npz"
OLDCODE_G = "grid_15m_1784052866_73a5e0ee_aaaaaaaaaaaaaaaa.npz"   # pre-rename
LEGACY_G = "grid_15m_1784052866_aaaaaaaaaaaaaaaa.npz"             # pre-codesig name

check("a hash SIBLING is KEPT (different gate/universe, equally valid)",
      SIB_G not in _grid_prune(CUR_G, [CUR_G, SIB_G]),
      detail=str(_grid_prune(CUR_G, [CUR_G, SIB_G])))
check("the grid just written is KEPT", CUR_G not in _grid_prune(CUR_G, [CUR_G]))
check("a grid built on older DATA is dropped",
      OLDDATA_G in _grid_prune(CUR_G, [CUR_G, OLDDATA_G]))
check("a grid built by older CODE is dropped (the 1.55GB leak)",
      OLDCODE_G in _grid_prune(CUR_G, [CUR_G, OLDCODE_G]))
check("the pre-codesig legacy name is dropped",
      LEGACY_G in _grid_prune(CUR_G, [CUR_G, LEGACY_G]))
check("alternating crypto/stocks runs prune NOTHING",
      _grid_prune(CUR_G, [CUR_G, SIB_G]) == []
      and _grid_prune(SIB_G, [CUR_G, SIB_G]) == [],
      detail=f"{_grid_prune(CUR_G, [CUR_G, SIB_G])} / "
             f"{_grid_prune(SIB_G, [CUR_G, SIB_G])}")

# one rule, not two: the inline copy in prepare_grid_cached only ever compared
# db_mtime, which is why code-stale grids survived forever
check("prepare_grid_cached uses stale_grid_paths, not a private copy",
      "stale_grid_paths(" in inspect.getsource(pm.prepare_grid_cached)
      and "mtoken" not in inspect.getsource(pm.prepare_grid_cached))


# 4. signals-cache pruning ----------------------------------------------------
# This decides which 360MB files get DELETED. It used to remove every cache but
# the current key, which was right until sig_params joined the key: then a
# default run and a --space run were each deleting the other's still-valid
# cache and paying a full rebuild on every alternation.
import test_strategy as ts  # noqa: E402

_pruned = []


def prune(current, present):
    """Which of `present` would be deleted after writing `current`."""
    real_glob = ts.glob.glob
    ts.glob.glob = lambda _pat: [os.path.join(ts.CACHE_DIR, p) for p in present]
    try:
        return sorted(os.path.basename(p)
                      for p in ts.stale_cache_paths(
                          os.path.join(ts.CACHE_DIR, current), "1d"))
    finally:
        ts.glob.glob = real_glob


CUR = "signals_1d_1784052866_v3_73a5e0ee_def.parquet"
SIBLING = "signals_1d_1784052866_v3_73a5e0ee_2af7d248.parquet"   # other params
OLD_DATA = "signals_1d_1700000000_v3_73a5e0ee_def.parquet"       # older mtime
OLD_SCHEMA = "signals_1d_1784052866_v2_73a5e0ee_def.parquet"     # older schema
OLD_CODE = "signals_1d_1784052866_v3_deadbeef_def.parquet"       # older code

check("a sibling with different sig_params is KEPT",
      SIBLING not in prune(CUR, [CUR, SIBLING]),
      detail=str(prune(CUR, [CUR, SIBLING])))
check("the cache just written is KEPT", CUR not in prune(CUR, [CUR, SIBLING]))
check("a cache built on older DATA is dropped",
      OLD_DATA in prune(CUR, [CUR, OLD_DATA]))
check("a cache built on an older SCHEMA is dropped",
      OLD_SCHEMA in prune(CUR, [CUR, OLD_SCHEMA]))
check("a cache built by older CODE is dropped",
      OLD_CODE in prune(CUR, [CUR, OLD_CODE]))
check("the pre-sig_params legacy name is dropped",
      "signals_1d_1784052866_v3.parquet" in
      prune(CUR, [CUR, "signals_1d_1784052866_v3.parquet"]))
check("alternating default/--space runs prune NOTHING (the actual bug)",
      prune(CUR, [CUR, SIBLING]) == [] and prune(SIBLING, [CUR, SIBLING]) == [],
      detail=f"{prune(CUR, [CUR, SIBLING])} / {prune(SIBLING, [CUR, SIBLING])}")


# 5. NO CLOCK BY DEFAULT ------------------------------------------------------
# "NEVER SELL FOR NO REASON JUST BECAUSE DAYS WENT BY THAT'S STUPID" -- Paul,
# who has said it since 2026-07-13. A time exit dumps winners mid-trend; real
# exits are stops, targets, trailing, or a bearish reversal signal. The fix
# landed in portfolio_multi and the search space and MISSED test_strategy and
# portfolio_sim, so the same strategy scored differently per tool and the
# harness quietly amputated runners (tdi_long 1d/12mo: hold=20 -> avg +1.51%
# PF 1.58; hold=0 -> avg +5.41% PF 2.11). Every tool that takes --hold is
# pinned here so it cannot drift back in one of them again.
import re  # noqa: E402

for tool in ("test_strategy.py", "portfolio_multi.py", "portfolio_sim.py"):
    src = open(tool, encoding="utf-8").read()
    m = re.search(r'arg\(args,\s*"hold",\s*"(\d+)"\)', src)
    check(f"{tool}: --hold defaults to 0 (no time clock)",
          m is not None and m.group(1) == "0",
          detail=f"default is {m.group(1) if m else 'NOT FOUND'}")

# sweep uses listarg(), a DIFFERENT form -- the arg() regex above silently
# matched nothing here, so the tool that runs the most configs was the one this
# test did not cover. A pin that skips a tool is not a pin.
_sw = open("sweep.py", encoding="utf-8").read()
m = re.search(r'listarg\(args,\s*"hold",\s*\[(\d+)\]', _sw)
check("sweep.py: --hold defaults to 0 (no time clock)",
      m is not None and m.group(1) == "0",
      detail=f"default is {m.group(1) if m else 'NOT FOUND'}")

# ...and the same disease one knob over: trailing OFF by default means the only
# exit is the stop, so win% is 0.0 BY CONSTRUCTION. The tools must agree, or the
# same config scores differently depending on which one you ran.
m = re.search(r'listarg\(args,\s*"trail-activate",\s*\[([0-9.]+)\]', _sw)
check("sweep.py: --trail-activate defaults ON (not the stop-only degenerate)",
      m is not None and float(m.group(1)) > 0,
      detail=f"default is {m.group(1) if m else 'NOT FOUND'}")
_pm = open("portfolio_multi.py", encoding="utf-8").read()
m2 = re.search(r'arg\(args,\s*"trail-activate",\s*"([0-9.]+)"\)', _pm)
check("portfolio_multi.py: --trail-activate defaults ON",
      m2 is not None and float(m2.group(1)) > 0,
      detail=f"default is {m2.group(1) if m2 else 'NOT FOUND'}")
check("sweep and portfolio_multi agree on the trailing default",
      m is not None and m2 is not None and float(m.group(1)) == float(m2.group(1)),
      detail=f"sweep={m.group(1) if m else '?'} pm={m2.group(1) if m2 else '?'}")

_spec = json.load(open("search_space.json", encoding="utf-8"))
check("search_space.json: sim.hold_bars is 0",
      _spec.get("sim", {}).get("hold_bars") == 0,
      detail=str(_spec.get("sim", {}).get("hold_bars")))


# 6. ONE gate parser -----------------------------------------------------------
# Four copies of "COL@IV -> (col, iv)" existed and they DISAGREED. portfolio_multi
# used `[... for g in gate_arg.split(",") if "@" in g]`, which silently DROPS a
# malformed entry: `--gate=sma50_over_200` (forgetting @1d) ran UNGATED while the
# header still printed `gate=sma50_over_200`. 443 trades -> 537 and a gate that never
# existed. Reporting a filter you did not apply is the same failure as reporting
# a trade that never fired.
_pg = ts.parse_gates

check("none -> no gates", _pg("none") == [] and _pg("") == [])
check("single gate parses", _pg("sma50_over_200@1d") == [("sma50_over_200", "1d")])
check("stacked gates parse",
      _pg("sma50_over_200@1d,above_50ma@4h")
      == [("sma50_over_200", "1d"), ("above_50ma", "4h")])


def _rejects(s):
    try:
        _pg(s)
        return False
    except SystemExit:
        return True


check("a gate with no @interval is REJECTED, not dropped", _rejects("sma50_over_200"))
check("a missing interval is rejected", _rejects("sma50_over_200@"))
check("a missing column is rejected", _rejects("@1d"))
check("one bad entry rejects the whole list",
      _rejects("sma50_over_200@1d,above_50ma"))

# ...and nobody kept a private copy. Compare CODE, not prose: ast.unparse drops
# docstrings and comments, so the notes explaining what was removed cannot be
# mistaken for the thing itself. (My first version of this check flagged
# parse_gates' own docstring as a private parser.)
def _code_only(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef,
                             ast.AsyncFunctionDef)):
            b = getattr(node, "body", None)
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                b.pop(0)
    return ast.unparse(tree)


check("test_strategy owns the one parser",
      "def parse_gates(" in _code_only("test_strategy.py"))
for tool in ("portfolio_multi.py", "portfolio_sim.py",
             "finder_gated.py", "finder_5m_gated.py"):
    code = _code_only(tool)
    check(f"{tool}: uses parse_gates", "parse_gates(" in code)
    check(f"{tool}: no hardcoded GATES constant", "GATES = [(" not in code)
    check(f"{tool}: no private silent-drop parse", 'if "@" in g' not in code)


# 7. ONE market-regime default ------------------------------------------------
# Same disease as --hold, one knob over. portfolio_multi's CLI defaulted to
# sma100 while search_space.json, sweep.py, and prepare_grid_cached's OWN
# signature all said none -- so a bare portfolio_multi run was regime-filtered
# (~31% fewer trades, 55 -> 38) while the search space it feeds was not, and the
# two disagreed about what a "default run" meant. It also vetoes every knife
# entry by construction: the broad market is always below its MA in a crash,
# which is exactly when the reversal setup Paul wants to test exists.
# Five declarations, so five pins. Each is read in the form that tool uses.
_MKT = {}

m = re.search(r'arg\(args,\s*"market",\s*"(\w+)"\)', _pm)
_MKT["portfolio_multi.py CLI"] = m.group(1) if m else None
check("portfolio_multi.py: --market defaults to none",
      _MKT["portfolio_multi.py CLI"] == "none",
      detail=f"default is {_MKT['portfolio_multi.py CLI'] or 'NOT FOUND'}")

m = re.search(r'arg\(args,\s*"market",\s*"(\w+)"\)', _sw)
_MKT["sweep.py CLI"] = m.group(1) if m else None
check("sweep.py: --market defaults to none",
      _MKT["sweep.py CLI"] == "none",
      detail=f"default is {_MKT['sweep.py CLI'] or 'NOT FOUND'}")

_MKT["search_space.json"] = _spec.get("grid", {}).get("market")
check("search_space.json: grid.market is none",
      _MKT["search_space.json"] == "none",
      detail=str(_MKT["search_space.json"]))

# the engine's own signature, not just the CLIs in front of it -- this is the
# one portfolio_multi's CLI was contradicting inside a single file
for fn in (pm.prepare_grid, pm.prepare_grid_cached):
    d = inspect.signature(fn).parameters["market"].default
    _MKT[f"pm.{fn.__name__}()"] = d
    check(f"pm.{fn.__name__}(): market= defaults to none", d == "none",
          detail=repr(d))

check("every tool AGREES on the market default",
      len(set(_MKT.values())) == 1,
      detail=" ".join(f"{k}={v!r}" for k, v in _MKT.items()))

# agent_search and orchestrate read G["market"] from the space rather than
# keeping a literal. That is the shape the others should have; pin that they
# have not quietly grown a private default that could drift from the space.
for tool in ("agent_search.py", "orchestrate.py"):
    src = open(tool, encoding="utf-8").read()
    check(f"{tool}: reads the market default from the space, not a literal",
          re.search(r'arg\(args,\s*"market",\s*G\["market"\]\)', src)
          is not None
          and re.search(r'arg\(args,\s*"market",\s*"', src) is None)


if __name__ == "__main__":
    print("\n" + ("ALL GRID-CACHE TRUST TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
