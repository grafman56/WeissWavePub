#!/usr/bin/env python3
"""Multi-strategy portfolio simulator — the bot. Runs SEVERAL validated
strategies at once on one pool of capital with a shared position cap. When
more setups fire than there are free slots, candidates across ALL strategies
are ranked by confluence score. Each position carries its own strategy's
stop / target / hold / exit-signal. This is how deployment rises: any single
edge is low-frequency, but a stack of them keeps the account working.

    python portfolio_multi.py --interval=15m --gate=minervini@1d,above_50ma@4h \
        --cost-bps=10 --max-positions=5

Strategies are read from bot_strategies.json (see that file's format). Shared
across all: --interval, --gate, --cost-bps, --max-positions, --capital,
--target (optional global take-profit override). ASCII output.
"""

import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from test_strategy import apply_gates, arg, emit, fail, load_universe
from weisswave import portsim
from weisswave.db import DB_PATH
from weisswave.optimize import benchmark_index
from weisswave.signals import combine_signals, recent

STOP_MODES = {"pct": portsim.PCT, "atr": portsim.ATR, "swing": portsim.SWING}


def build_grid(frames, strategies, exit_cols, gstop, ghold, gtarget,
               atr_len, swing_look):
    """Align all symbols onto a common time grid -> 2D (T x S) arrays for the
    numba engine, plus per-bar best-strategy entry/score/index and stop refs.
    Returns (arrays dict, symbols, grid, strat_stop/hold/target)."""
    syms = list(frames.keys())
    S = len(syms)
    grid = np.array(sorted(set().union(
        *[set(f.index.values) for f in frames.values()])))
    T = len(grid)
    gpos = {ts: i for i, ts in enumerate(grid)}
    A = {k: np.zeros((T, S)) for k in ("O", "H", "L", "C", "SCORE", "ATR", "SW")}
    V = np.zeros((T, S), bool)
    ENT = np.zeros((T, S), bool)
    SIDX = np.zeros((T, S), np.int64)
    EXT = np.zeros((T, S), bool)
    st_stop = np.array([gstop or float(s.get("stop_pct", 0.08))
                        for s in strategies])
    st_hold = np.array([ghold if ghold is not None else int(s.get("hold", 0))
                        for s in strategies], np.int64)
    st_tgt = np.array([(s.get("target", gtarget) or 0.0) for s in strategies])

    for j, sname in enumerate(syms):
        sig = frames[sname]
        pos = np.array([gpos[ts] for ts in sig.index.values])
        A["O"][pos, j] = sig["Open"].to_numpy()
        A["H"][pos, j] = sig["High"].to_numpy()
        A["L"][pos, j] = sig["Low"].to_numpy()
        A["C"][pos, j] = sig["Close"].to_numpy()
        V[pos, j] = True
        g = (sig["xtf_gate"].to_numpy(bool) if "xtf_gate" in sig.columns
             else np.ones(len(sig), bool))
        hi, lo, cl = (sig["High"].to_numpy(), sig["Low"].to_numpy(),
                      sig["Close"].to_numpy())
        pc = np.concatenate([[cl[0]], cl[:-1]])
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
        A["ATR"][pos, j] = pd.Series(tr).rolling(atr_len, min_periods=1).mean()
        A["SW"][pos, j] = pd.Series(lo).rolling(swing_look, min_periods=1).min()

        best_score = np.full(len(sig), -1.0)
        best_idx = np.zeros(len(sig), np.int64)
        any_ent = np.zeros(len(sig), bool)
        ext_any = np.zeros(len(sig), bool)
        for k, st in enumerate(strategies):
            cols = [c for c in st["entry_cols"] if c in sig.columns]
            if not cols:
                continue
            w = int(st.get("window", 5))
            ent = combine_signals(sig, cols, int(st.get("min_count", 1)),
                                  w).to_numpy(bool) & g
            score = sum(recent(sig[c], w).astype(int)
                        for c in cols).to_numpy(float)
            better = ent & (score > best_score)
            best_idx[better] = k
            best_score[better] = score[better]
            any_ent |= ent
            ex = st.get("exit_cols") or exit_cols
            if ex:
                cs = [c for c in ex if c in sig.columns]
                if cs:
                    ext_any |= sig[cs].astype(bool).any(axis=1).to_numpy()
        ENT[pos, j] = any_ent
        A["SCORE"][pos, j] = np.where(best_score < 0, 0.0, best_score)
        SIDX[pos, j] = best_idx
        EXT[pos, j] = ext_any
    return A, V, ENT, SIDX, EXT, syms, grid, st_stop, st_hold, st_tgt


def _apply_market(frames, market):
    if market == "none":
        return frames
    ma_win = int(market.replace("sma", "") or 100)
    didx = benchmark_index(load_universe("1d", None))
    up = (didx > didx.rolling(ma_win, min_periods=ma_win).mean())
    upc = pd.Series(up.to_numpy(bool),
                    index=pd.DatetimeIndex(up.index) + pd.Timedelta(days=1))
    upc = upc[~upc.index.duplicated(keep="last")].sort_index()
    for s, sig in frames.items():
        m = upc.reindex(sig.index, method="ffill").fillna(False).to_numpy(bool)
        base = (sig["xtf_gate"].to_numpy(bool) if "xtf_gate" in sig.columns
                else np.ones(len(sig), bool))
        sig["xtf_gate"] = base & m
    return frames


def prepare_grid(strategies, interval="15m",
                 gate_arg="minervini@1d,above_50ma@4h", market="none",
                 months=0, exit_cols=None, gstop=None, ghold=None,
                 gtarget=None, atr_len=14, swing_look=20, cutoff=None):
    """All the expensive, param-independent work: load -> gate -> market ->
    build 2D grid. Do this ONCE, then sweep exit params via portsim.simulate
    on the returned arrays (each config = a numba call = milliseconds).
    Pass an explicit `cutoff` to override the months-derived window (the grid
    cache uses this so a hit is a byte-identical rebuild)."""
    gates = [tuple(g.split("@", 1)) for g in gate_arg.split(",")
             if "@" in g] if gate_arg != "none" else []
    if cutoff is None and months:
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=months)
    frames = load_universe(interval, cutoff)
    if gates:
        frames = apply_gates(frames, gates, None)
    frames = _apply_market(frames, market)
    grid = build_grid(frames, strategies, exit_cols or [], gstop, ghold,
                      gtarget, atr_len, swing_look)
    return grid, frames


GRID_CACHE_DIR = "grid_cache"


def _grid_cache_path(strategies, interval, gate_arg, market, cutoff,
                     exit_cols, gstop, ghold, gtarget, atr_len, swing_look,
                     db_mtime):
    """A file path uniquely determined by every input build_grid consumes.
    Same key <-> byte-identical grid, so a cache hit is always trustworthy.
    db_mtime is in the filename (not just the hash) so stale-data grids are
    prunable when the DB is refreshed."""
    payload = json.dumps({
        "strategies": strategies,      # full defs: entry/exit cols, window...
        "interval": interval, "gate": gate_arg, "market": market,
        "cutoff": None if cutoff is None else pd.Timestamp(cutoff).isoformat(),
        "exit_cols": exit_cols or [], "gstop": gstop, "ghold": ghold,
        "gtarget": gtarget, "atr_len": atr_len, "swing_look": swing_look,
    }, sort_keys=True, default=str)
    h = hashlib.sha1(payload.encode()).hexdigest()[:16]
    return os.path.join(GRID_CACHE_DIR,
                        f"grid_{interval.replace(':', '')}_{db_mtime}_{h}.npz")


def _save_grid(path, g):
    A, V, ENT, SIDX, EXT, syms, grid, st_stop, st_hold, st_tgt = g
    os.makedirs(GRID_CACHE_DIR, exist_ok=True)
    tmp = path + ".tmp.npz"          # .npz so numpy won't append its own
    np.savez_compressed(
        tmp, **{f"A_{k}": v for k, v in A.items()},
        V=V, ENT=ENT, SIDX=SIDX, EXT=EXT, syms=np.array(syms), grid=grid,
        st_stop=st_stop, st_hold=st_hold, st_tgt=st_tgt)
    os.replace(tmp, path)            # atomic publish; no half-written cache


def _load_grid(path):
    z = np.load(path)
    A = {k[2:]: z[k] for k in z.files if k.startswith("A_")}
    return (A, z["V"], z["ENT"], z["SIDX"], z["EXT"],
            [str(s) for s in z["syms"]], z["grid"],
            z["st_stop"], z["st_hold"], z["st_tgt"])


def prepare_grid_cached(strategies, interval="15m",
                        gate_arg="minervini@1d,above_50ma@4h", market="none",
                        months=0, exit_cols=None, gstop=None, ghold=None,
                        gtarget=None, atr_len=14, swing_look=20):
    """prepare_grid with a disk cache: pay the ~grid-build once per
    (strategies, gate, market, interval, window, params, DB-data) tuple, then
    every later sweep loads the 2D grid in ~a second. Returns
    (grid_tuple, cached_bool); does NOT return frames (sweeps don't need them).
    The window is normalized to the day so repeated same-day runs hit cache."""
    cutoff = ((pd.Timestamp.now().normalize() - pd.DateOffset(months=months))
              if months else None)
    path = _grid_cache_path(strategies, interval, gate_arg, market, cutoff,
                            exit_cols, gstop, ghold, gtarget, atr_len,
                            swing_look, int(os.path.getmtime(DB_PATH)))
    if os.path.exists(path):
        return _load_grid(path), True
    grid, _frames = prepare_grid(
        strategies, interval, gate_arg, market, months, exit_cols, gstop,
        ghold, gtarget, atr_len, swing_look, cutoff=cutoff)
    _save_grid(path, grid)
    # drop grids built against older DB data for this interval
    mtoken = os.path.basename(path).rsplit("_", 1)[0]  # grid_<int>_<mtime>
    for old in glob.glob(os.path.join(
            GRID_CACHE_DIR, f"grid_{interval.replace(':', '')}_*.npz")):
        if not os.path.basename(old).startswith(mtoken):
            os.remove(old)
    return grid, False


BOT_FILE = "bot_strategies.json"


def main():
    args = sys.argv[1:]
    interval = arg(args, "interval", "15m")
    gate_arg = arg(args, "gate", "minervini@1d,above_50ma@4h")
    cost_side = float(arg(args, "cost-bps", "0")) / 10000.0 / 2
    max_pos = int(arg(args, "max-positions", "5"))
    capital = float(arg(args, "capital", "100000"))
    months = int(arg(args, "months", "0"))
    gtarget = arg(args, "target", None)
    gtarget = float(gtarget) if gtarget not in (None, "none", "") else None
    gstop = arg(args, "stop", None)
    gstop = float(gstop) if gstop not in (None, "none", "") else None
    ghold = arg(args, "hold", None)
    ghold = int(ghold) if ghold not in (None, "none", "") else None
    # stop placement: pct (fixed %), atr (entry - mult*ATR), swing (below the
    # recent swing low - buffer, i.e. under support). atr/swing avoid the
    # noise-tagging that a fixed 3% stop suffers.
    stop_mode = arg(args, "stop-mode", "pct")
    atr_len = int(arg(args, "atr-len", "14"))
    atr_mult = float(arg(args, "atr-mult", "2.5"))
    swing_look = int(arg(args, "swing-look", "20"))
    swing_buf = float(arg(args, "swing-buf", "0.005"))
    # trailing stop: once a trade is up +trail_act, ratchet the stop to
    # trail_dist below the high-water mark (only ever raises). None = off.
    trail_act = arg(args, "trail-activate", None)
    trail_act = float(trail_act) if trail_act not in (None, "none", "") else None
    trail_dist = float(arg(args, "trail-dist", "0.03"))
    exit_arg = arg(args, "exit", None)          # bearish reversal exit signal(s)
    exit_cols = [] if exit_arg in (None, "none", "") else exit_arg.split(",")
    only = arg(args, "strategies", None)        # comma list of names, or all
    path = arg(args, "file", BOT_FILE)

    if not os.path.exists(path):
        fail(f"no {path}")
    with open(path, encoding="utf-8") as f:
        strategies = json.load(f)
    if only:
        want = set(only.split(","))
        strategies = [s for s in strategies if s["name"] in want]
    if not strategies:
        fail("no strategies selected")

    gates = [tuple(g.split("@", 1)) for g in gate_arg.split(",")
             if "@" in g] if gate_arg != "none" else []

    market = arg(args, "market", "sma100")      # market-regime filter; none to disable
    cutoff = (pd.Timestamp.now() - pd.DateOffset(months=months)
              if months else None)
    frames = load_universe(interval, cutoff)
    if gates:
        frames = apply_gates(frames, gates, None)

    # ── Market-regime filter: when the broad market is in a downtrend, the
    #    bot trades NOTHING (no new entries anywhere) — it waits in cash for
    #    the market to turn back up. The top of the cascade. Uses the daily
    #    equal-weight index of the universe vs its own moving average.
    if market != "none":
        ma_win = int(market.replace("sma", "") or 100)
        didx = benchmark_index(load_universe("1d", None))
        up = (didx > didx.rolling(ma_win, min_periods=ma_win).mean())
        upc = pd.Series(up.to_numpy(bool),
                        index=pd.DatetimeIndex(up.index) + pd.Timedelta(days=1))
        upc = upc[~upc.index.duplicated(keep="last")].sort_index()
        for s, sig in frames.items():
            m = upc.reindex(sig.index, method="ffill").fillna(False).to_numpy(bool)
            base = (sig["xtf_gate"].to_numpy(bool) if "xtf_gate" in sig.columns
                    else np.ones(len(sig), bool))
            sig["xtf_gate"] = base & m

    # numba is the default engine; --engine=loop keeps the original Python
    # loop reachable as a reference implementation to cross-check against.
    engine = arg(args, "engine", "numba")
    if engine == "numba":
        import time as _time
        _t = _time.time()
        A, V, ENT, SIDX, EXT, syms, master, st_stop, st_hold, st_tgt = \
            build_grid(frames, strategies, exit_cols, gstop, ghold, gtarget,
                       atr_len, swing_look)
        res = portsim.simulate(
            A["O"], A["H"], A["L"], A["C"], V, ENT, A["SCORE"], SIDX, EXT,
            A["ATR"], A["SW"], st_stop, st_hold, st_tgt,
            stop_mode=STOP_MODES.get(stop_mode, 0), atr_mult=atr_mult,
            swing_buf=swing_buf, trail_act=trail_act or 0.0,
            trail_dist=trail_dist, cost_side=cost_side, max_pos=max_pos,
            init_cash=capital)
        why = {1: "stop", 2: "trail", 3: "target", 4: "signal", 5: "time",
               6: "eod"}
        trades = [{"symbol": syms[res["sym"][i]],
                   "strat": strategies[res["strat"][i]]["name"],
                   "ret": res["ret"][i], "why": why[res["reason"][i]]}
                  for i in range(len(res["ret"]))]
        eq = pd.Series(res["equity"], index=pd.DatetimeIndex(master))
        inv = res["invested"]
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        cagr = (eq.iloc[-1] / capital) ** (1 / years) - 1
        dd = (eq / eq.cummax() - 1).min()
        r = pd.Series([t["ret"] for t in trades])
        by_strat = pd.Series([t["strat"] for t in trades]).value_counts()
        by_reason = pd.Series([t["why"] for t in trades]).value_counts()
        traded = {t["symbol"] for t in trades}
        bench_frames = {s: frames[s] for s in traded if s in frames} or frames
        bench = benchmark_index(bench_frames)
        bench = bench / bench.iloc[0] * capital
        secs = _time.time() - _t
        lines = [
            f"BOT[numba]: {len(strategies)} strat gate={gate_arg} market={market}"
            f" interval={interval} max_pos={max_pos} "
            f"cost={cost_side * 2 * 10000:.0f}bps stop={stop_mode}"
            + (f" trail={trail_act:.0%}@{trail_dist:.0%}" if trail_act else "")
            + (f" window=last {months}mo" if months else "") + f"  ({secs:.1f}s)",
            f"period: {eq.index[0].date()} -> {eq.index[-1].date()}  "
            f"universe={len(syms)}",
            f"BOT:      {eq.iloc[-1]:>10,.0f}  CAGR={cagr:+.1%}  maxDD={dd:.1%}  "
            f"invested={inv.mean():.0%}  trades/wk={len(r)/(years*52):.1f}",
            f"hold {len(traded)} traded: {bench.iloc[-1]:>10,.0f}  "
            f"CAGR={(bench.iloc[-1]/capital)**(1/years)-1:+.1%}",
            f"trades: n={len(r)}  win={(r > 0).mean():.1%}  avg={r.mean():+.2%}  "
            f"win_avg={r[r > 0].mean():+.2%}  loss_avg={r[r < 0].mean():+.2%}  "
            f"open_end={res['open_end']}",
            "exits by reason: " + ", ".join(f"{k}={v}" for k, v in
                                            by_reason.items()),
            "trades by strategy: " + ", ".join(f"{k}={v}" for k, v in
                                               by_strat.items()),
        ]
        emit(lines)
        return

    # per-symbol arrays + per-strategy entry/score/exit
    data = {}
    for s, sig in frames.items():
        g = (sig["xtf_gate"].to_numpy(bool) if "xtf_gate" in sig.columns
             else np.ones(len(sig), bool))
        strats = []
        for st in strategies:
            cols = [c for c in st["entry_cols"] if c in sig.columns]
            if not cols:
                continue
            w = int(st.get("window", 5))
            ent = combine_signals(sig, cols, int(st.get("min_count", 1)),
                                  w).to_numpy(bool) & g
            score = sum(recent(sig[c], w).astype(int) for c in cols).to_numpy()
            ex = st.get("exit_cols") or exit_cols   # global --exit applies to all
            ext = (sig[[c for c in ex if c in sig.columns]].astype(bool)
                   .any(axis=1).to_numpy(bool) if ex
                   else np.zeros(len(sig), bool))
            strats.append({"name": st["name"], "ent": ent, "score": score,
                           "ext": ext,
                           "stop": gstop or float(st.get("stop_pct", 0.08)),
                           "hold": ghold if ghold is not None
                           else int(st.get("hold", 0)),
                           "target": st.get("target", gtarget)})
        if strats:
            hi = sig["High"].to_numpy(float)
            lo = sig["Low"].to_numpy(float)
            cl = sig["Close"].to_numpy(float)
            pc = np.concatenate([[cl[0]], cl[:-1]])
            tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
            atr = pd.Series(tr).rolling(atr_len, min_periods=1).mean().to_numpy()
            swinglow = pd.Series(lo).rolling(swing_look, min_periods=1).min() \
                         .to_numpy()
            data[s] = {"idx": sig.index.values,
                       "open": sig["Open"].to_numpy(float),
                       "low": lo, "high": hi, "close": cl,
                       "atr": atr, "swinglow": swinglow,
                       "strats": strats, "ptr": 0}
    if not data:
        fail("no usable symbols/strategies")

    master = np.array(sorted(set(np.concatenate(
        [d["idx"] for d in data.values()]))))
    cash = capital
    positions = {}
    trades = []
    eqc = np.empty(len(master))
    inv = np.empty(len(master))

    for ti, t in enumerate(master):
        today = {}
        for s, d in data.items():
            p = d["ptr"]
            if p < len(d["idx"]) and d["idx"][p] == t:
                today[s] = p
                d["ptr"] = p + 1

        for s in list(positions):
            if s not in today:
                continue
            k = today[s]
            pos = positions[s]
            pos["held"] += 1
            d = data[s]
            # trailing stop: hwm reflects highs through the PRIOR bar (no
            # lookahead); once up +trail_act, ratchet stop up under the hwm
            if trail_act is not None and \
                    pos["hwm"] >= pos["entry"] * (1 + trail_act):
                cand = pos["hwm"] * (1 - trail_dist)
                if cand > pos["stop_px"]:
                    pos["stop_px"] = cand
                    pos["trailing"] = True
            xpx = None
            if d["low"][k] <= pos["stop_px"]:
                xpx, why = (min(d["open"][k], pos["stop_px"]),
                            "trail" if pos.get("trailing") else "stop")
            elif d["high"][k] >= pos["tp_px"]:
                xpx, why = max(d["open"][k], pos["tp_px"]), "target"
            elif k > 0 and pos["ext"][k - 1]:
                xpx, why = d["open"][k], "signal"
            elif pos["hold"] > 0 and pos["held"] >= pos["hold"]:
                xpx, why = d["open"][k], "time"
            else:
                pos["hwm"] = max(pos["hwm"], d["high"][k])   # for next bar
            if xpx is not None:
                xpx *= (1 - cost_side)
                cash += pos["shares"] * xpx
                trades.append({"symbol": s, "strat": pos["strat"],
                               "ret": xpx / pos["entry"] - 1, "why": why})
                del positions[s]

        if len(positions) < max_pos:
            cands = []
            for s, k in today.items():
                if s in positions or k == 0:
                    continue
                best = None
                for st in data[s]["strats"]:
                    if st["ent"][k - 1] and (best is None
                                             or st["score"][k - 1] > best[0]):
                        best = (st["score"][k - 1], st)
                if best:
                    cands.append((-best[0], s, k, best[1]))
            cands.sort(key=lambda x: x[0])
            eq_now = cash + sum(
                p["shares"] * data[s2]["close"][today.get(s2, data[s2]["ptr"] - 1)]
                for s2, p in positions.items())
            for _, s, k, st in cands:
                if len(positions) >= max_pos:
                    break
                alloc = min(eq_now / max_pos, cash)
                if alloc <= 0:
                    break
                base = data[s]["open"][k]
                px = base * (1 + cost_side)
                if px <= 0 or not np.isfinite(px):
                    continue
                kk = max(k - 1, 0)          # stop reference: last closed bar
                if stop_mode == "atr":
                    stop_px = base - atr_mult * data[s]["atr"][kk]
                elif stop_mode == "swing":
                    stop_px = data[s]["swinglow"][kk] * (1 - swing_buf)
                else:
                    stop_px = base * (1 - st["stop"])
                if not np.isfinite(stop_px) or stop_px >= base:
                    stop_px = base * (1 - st["stop"])   # fallback
                cash -= (alloc / px) * px
                positions[s] = {"shares": alloc / px, "entry": px,
                                "stop_px": stop_px, "hwm": base,
                                "trailing": False,
                                "tp_px": base * (1 + st["target"])
                                if st["target"] else float("inf"),
                                "hold": st["hold"], "held": 0,
                                "ext": st["ext"], "strat": st["name"]}

        mv = sum(p["shares"] * data[s]["close"][today[s]] if s in today
                 else p["shares"] * data[s]["close"][data[s]["ptr"] - 1]
                 for s, p in positions.items())
        eqc[ti] = cash + mv
        inv[ti] = mv / eqc[ti] if eqc[ti] else 0

    eq = pd.Series(eqc, index=pd.DatetimeIndex(master))
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / capital) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    r = pd.Series([t["ret"] for t in trades])
    by_strat = pd.Series([t["strat"] for t in trades]).value_counts()
    by_reason = pd.Series([t["why"] for t in trades]).value_counts()
    # benchmark = buy-and-hold the STOCKS THE BOT ACTUALLY TRADED over the
    # same period (the honest opportunity cost: did active trading of these
    # names beat just holding them?). Falls back to full universe if empty.
    traded = {t["symbol"] for t in trades}
    bench_frames = {s: frames[s] for s in traded if s in frames} or frames
    bench = benchmark_index(bench_frames)
    bench = bench / bench.iloc[0] * capital
    bench_cagr = (bench.iloc[-1] / capital) ** (1 / years) - 1
    bench_dd = (bench / bench.cummax() - 1).min()

    lines = [
        f"BOT: {len(strategies)} strategies, gate={gate_arg} market={market} "
        f"interval={interval} max_pos={max_pos} cost={cost_side * 2 * 10000:.0f}bps "
        + (f"stop={stop_mode}" + (f"({atr_mult}xATR)" if stop_mode == "atr"
           else f"(swing-{swing_buf:.1%})" if stop_mode == "swing" else ""))
        + (f" target={gtarget:.0%}" if gtarget else "")
        + (f" trail={trail_act:.0%}@{trail_dist:.0%}" if trail_act else "")
        + (f" window=last {months}mo" if months else ""),
        f"period: {eq.index[0].date()} -> {eq.index[-1].date()}  "
        f"universe={len(data)}",
        f"BOT:      {eq.iloc[-1]:>10,.0f}  CAGR={cagr:+.1%}  maxDD={dd:.1%}  "
        f"invested={inv.mean():.0%}  trades/wk={len(r)/(years*52):.1f}",
        f"hold {len(traded)} traded: {bench.iloc[-1]:>10,.0f}  "
        f"CAGR={bench_cagr:+.1%}  maxDD={bench_dd:.1%}  "
        f"(opportunity cost of the names it traded)",
        f"trades: n={len(r)}  win={(r > 0).mean():.1%}  avg={r.mean():+.2%}  "
        f"win_avg={r[r > 0].mean():+.2%}  loss_avg={r[r < 0].mean():+.2%}  "
        f"open_end={len(positions)}",
        "exits by reason: " + ", ".join(f"{k}={v}" for k, v in
                                        by_reason.items()),
        "trades by strategy: " + ", ".join(f"{k}={v}" for k, v in
                                           by_strat.items()),
        f"verdict: bot {'>' if eq.iloc[-1] > bench.iloc[-1] else '<'} holding "
        f"its traded names; {'PROFITABLE' if cagr > 0 else 'UNPROFITABLE'} "
        f"({cagr:+.1%}/yr on active capital, {dd:.1%} maxDD)",
    ]
    emit(lines)


if __name__ == "__main__":
    main()
