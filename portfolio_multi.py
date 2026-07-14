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

from test_strategy import (GATE_DUR, apply_gates, arg, emit, fail,
                           load_universe)
from weisswave import portsim
from weisswave.db import DB_PATH
from weisswave.optimize import benchmark_index
from weisswave.rci import DISPLACEMENT, LAGGING_SPAN2_PERIODS, rci
from weisswave.signals import combine_signals, recent
from weisswave.structure import trend_points

STOP_MODES = {"pct": portsim.PCT, "atr": portsim.ATR, "swing": portsim.SWING,
              "fib": portsim.FIB}
TRAIL_MODES = {"pct": portsim.TRAIL_PCT, "structure": portsim.TRAIL_STRUCT,
               "fib": portsim.TRAIL_FIB}
FIB_ENTRY_MODES = {"off": portsim.FIB_ENTRY_OFF, "zone": portsim.FIB_ENTRY_ZONE,
                   "bounce": portsim.FIB_ENTRY_BOUNCE,
                   "bounce-trend": portsim.FIB_ENTRY_BOUNCE_TREND}

# ── Confluence factors ─────────────────────────────────────────────────────
# Each factor is a per-bar signed strength in ~[-1, 1] (+ = argues for a long,
# - = against). The engine's confluence entry combines them as a WEIGHTED SUM
# (score = sum w_i * factor_i); a --w-<name> knob per factor is auto-exposed,
# and weights are applied at sim time so they sweep cheaply. Add a factor here
# + compute it in build_grid and it's instantly weightable/combinable.
# entry factors (indices 0..5) decide WHEN to enter on the trading TF; the
# htf_* factors (the tail) are the higher-TF SETUP screen (which stocks are
# eligible). HTF_START = count of entry factors; the engine sums entry factors
# for the entry score and htf factors for the screen score, each thresholded.
FACTOR_NAMES = ["signal", "trend", "fib_prox", "range_pos", "vol_dom", "div",
                "rci_trend", "rci_heat",
                "htf_trend", "htf_ema_dist", "htf_fib_prox", "htf_rci_trend"]
HTF_START = 8            # first index of the higher-TF (weekly) factors
# `range_pos` was called `dip_bias` and its comment claimed it was the RCI
# "dangerous range". It is not -- it is price position in a rolling 20-bar
# high/low range. Paul's real RCI (adaptive regression channel + ichimoku) is
# now ported in weisswave/rci.py and enters as rci_trend / rci_heat, with the
# weekly version as htf_rci_trend. The old name is kept alive only as a rename
# so nothing silently claims to be the RCI while computing something else.
SIGNAL_NORM = 3.0        # strategy-confluence count that maps to factor 1.0
FIB_PROX_BAND = 0.05     # within this frac of the leg span from a level -> ~1
RNG_LOOK = 20            # lookback for the range_pos factor
# GOAL #2's third leg: the time-tested textbook strategies Paul's own signals
# must be measured against (alongside buy-and-hold). Carried in the grid as an
# (T, S, len) bool stack so the benchmark runs through the SAME engine, stops
# and costs as any config -- an apples-to-apples comparison, not a quoted stat.
MAINSTREAM_COLUMNS = ["golden_cross", "macd_cross_up", "rsi_oversold_cross",
                      "above_50ma"]
# rci_heat is normalized by this (% above the channel's lower band -> ~1.0).
# Paul's chart thresholds are 25 (heated) / 35 (superheated) on DAILY; on 15m
# the 99th percentile of heat is ~20, so those levels never fire there. 20 maps
# the live 15m range onto the factor scale; the search tunes the rest.
RCI_HEAT_NORM = 20.0
DIV_LOOK = 5             # a WTV divergence stays live for the div factor N bars
WEEKLY_EMAS = (20, 50, 100, 200)   # Paul's EMA ladder, on the weekly TF
HTF_EMA_REF = 50         # reference weekly EMA for htf_ema_dist
HTF_FIB_LR = 10          # weekly pivot window for htf_fib_prox
# only the pivot window is baked into the grid; the fib ratio/buffer/zone are
# applied at sim time so they stay cheaply sweepable. Default 10 = only
# "significant" swings become pivots (matches TradingView Auto Fib depth);
# a bigger right window means more confirmation lag before the anchor updates.
FIB_DEFAULTS = {"left": 10, "right": 10}


def build_grid(frames, strategies, exit_cols, gstop, ghold, gtarget,
               atr_len, swing_look, fib_left=10, fib_right=10,
               gate_mode="hard"):
    """Align all symbols onto a common time grid -> 2D (T x S) arrays for the
    numba engine, plus per-bar best-strategy entry/score/index and stop refs.

    gate_mode decides how the higher-TF trend gate (`--gate`, e.g.
    minervini@1d) reaches the engine -- it is a CHOICE, not a law:
      "hard"   : the gate ANDs the strategy signal (a bar outside the daily
                 uptrend can never be an entry). What the CLI tools expect.
      "factor" : the signal is left ungated and the trend travels ONLY via the
                 `trend` factor (+1 in-trend / -1 out, index 1), so its weight
                 decides how much it matters -- w_trend=0 ignores the trend, a
                 large w_trend reproduces a gate, between is a soft tilt. Lets
                 a search TEST the gate instead of being forced into it.
    Either way `--gate` still selects WHICH criteria the trend means.
    ATR/SW are per-bar stop references; P1/P2/P3 are the three fib-trend
    anchors (leg-start low, swing high, pullback low) the engine forms the fib
    stop/zone/target/extension-trail from at sim time. All lookahead-free —
    computed from bars <= t.
    Returns (arrays dict, symbols, grid, strat_stop/hold/target)."""
    syms = list(frames.keys())
    S = len(syms)
    grid = np.array(sorted(set().union(
        *[set(f.index.values) for f in frames.values()])))
    T = len(grid)
    gpos = {ts: i for i, ts in enumerate(grid)}
    A = {k: np.zeros((T, S)) for k in ("O", "H", "L", "C", "SCORE", "ATR",
                                       "SW", "P1", "P2", "P3", "GATE")}
    A["FACTORS"] = np.zeros((T, S, len(FACTOR_NAMES)))   # (T, S, K) factor stack
    A["MS"] = np.zeros((T, S, len(MAINSTREAM_COLUMNS)))  # textbook benchmarks
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
        A["GATE"][pos, j] = g.astype(float)      # trend mask for BOUNCE_TREND
        hi, lo, cl = (sig["High"].to_numpy(), sig["Low"].to_numpy(),
                      sig["Close"].to_numpy())
        pc = np.concatenate([[cl[0]], cl[:-1]])
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
        A["ATR"][pos, j] = pd.Series(tr).rolling(atr_len, min_periods=1).mean()
        A["SW"][pos, j] = pd.Series(lo).rolling(swing_look, min_periods=1).min()
        # fib anchor: from a higher timeframe when attach_fib_anchor supplied
        # one (the default -- Paul anchors high and those levels persist), else
        # from the trading TF itself. Both stay testable; see attach_fib_anchor.
        if "fib_p1" in sig.columns:
            p1 = sig["fib_p1"]; p2 = sig["fib_p2"]; p3 = sig["fib_p3"]
        else:
            p1, p2, p3 = trend_points(sig["High"], sig["Low"],
                                      fib_left, fib_right)
        A["P1"][pos, j] = p1.to_numpy()          # leg-start swing low
        A["P2"][pos, j] = p2.to_numpy()          # swing high
        A["P3"][pos, j] = p3.to_numpy()          # pullback low (NaN until conf.)

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
                                  w).to_numpy(bool)
            if gate_mode == "hard":
                ent = ent & g        # veto; in "factor" mode w_trend decides
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

        # ── confluence factors (signed strength ~[-1, 1]) ──────────────────
        # signal: strategy-confluence count, normalized, 0..1
        f_signal = np.minimum(1.0, np.where(best_score < 0, 0.0, best_score)
                              / SIGNAL_NORM)
        # trend: +1 in the higher-TF uptrend, -1 otherwise (a soft veto)
        f_trend = np.where(g, 1.0, -1.0)
        # fib_prox: closeness of Close to the nearest retracement level of the
        # P1->P2 up-leg (on a level -> ~1, far -> 0)
        p1v, p2v = p1.to_numpy(), p2.to_numpy()
        legv = p2v - p1v
        okleg = legv > 0
        prox = np.zeros(len(sig))
        for r in (0.382, 0.5, 0.618, 0.786):
            lvl = p2v - r * legv
            d = np.abs(cl - lvl) / np.where(okleg, legv, np.nan)
            prox = np.maximum(prox, np.where(
                okleg, np.maximum(0.0, 1.0 - d / FIB_PROX_BAND), 0.0))
        # range_pos: where price sits in its recent RNG_LOOK-bar high/low range.
        # +1 near the range lows (a dip, good for a long), -1 near the highs
        # (extended). Honest name: this is a plain range-position proxy. It was
        # called `dip_bias` and documented as the RCI "dangerous range" -- it
        # never was; the real RCI is rci_heat below (weisswave/rci.py).
        hh = pd.Series(hi).rolling(RNG_LOOK, min_periods=1).max().to_numpy()
        ll = pd.Series(lo).rolling(RNG_LOOK, min_periods=1).min().to_numpy()
        rng = hh - ll
        rng_pos = np.where(rng > 0, (cl - ll) / np.where(rng > 0, rng, 1.0),
                           0.5)
        range_pos = np.clip(1.0 - 2.0 * rng_pos, -1.0, 1.0)
        # rci_trend: Paul's ichimoku/regression-channel trend, already signed
        # [-1,1] by the study's own precedence (cloud break > plain ichi trend).
        rci_tr = (sig["rci_trend"].to_numpy(float) if "rci_trend" in sig.columns
                  else np.zeros(len(sig)))
        # rci_heat: the REAL "dangerous range" -- how far ohlc4 has stretched
        # above the adaptive channel's lower band. Signed so that overextended
        # is NEGATIVE (a soft "don't buy the top"), never a veto.
        # Scaled by RCI_HEAT_NORM rather than his 25/35 chart thresholds: those
        # are calibrated for DAILY and sit above the 99th percentile on 15m, so
        # a boolean would never fire on the timeframe the bot trades. Feeding
        # the continuous value lets the search find its own threshold.
        heat = (sig["rci_heat"].to_numpy(float) if "rci_heat" in sig.columns
                else np.zeros(len(sig)))
        rci_ht = -np.clip(np.nan_to_num(heat) / RCI_HEAT_NORM, 0.0, 1.0)
        # vol_dom: WTV volume-dominance, graded signed (+ heavy buying already
        # computed by pressure_tiers, - heavy selling). THE volume confirmation
        # -- computed upstream but never wired into what trades until now.
        getb = (lambda name: sig[name].to_numpy(bool) if name in sig.columns
                else np.zeros(len(sig), bool))
        vd = np.zeros(len(sig))
        vd[getb("buy_dominant")] = 0.3
        vd[getb("heavy_buy")] = 0.6
        vd[getb("very_heavy_buy")] = 1.0
        vd[getb("sell_dominant")] = -0.3
        vd[getb("heavy_sell")] = -0.6
        vd[getb("very_heavy_sell")] = -1.0
        # div: WTV divergence confluence, recent bull (+) vs bear (-)
        bull_div = getb("wt_bull_div") | getb("wt_hidden_bull_div")
        bear_div = getb("wt_bear_div") | getb("wt_hidden_bear_div")
        bull_r = recent(pd.Series(bull_div, index=sig.index), DIV_LOOK)
        bear_r = recent(pd.Series(bear_div, index=sig.index), DIV_LOOK)
        dv = np.clip(bull_r.to_numpy(float) - bear_r.to_numpy(float), -1.0, 1.0)
        A["FACTORS"][pos, j, 0] = f_signal
        A["FACTORS"][pos, j, 1] = f_trend
        A["FACTORS"][pos, j, 2] = prox
        A["FACTORS"][pos, j, 3] = range_pos
        A["FACTORS"][pos, j, 4] = vd
        A["FACTORS"][pos, j, 5] = dv
        A["FACTORS"][pos, j, 6] = rci_tr
        A["FACTORS"][pos, j, 7] = rci_ht
        # higher-TF (weekly) setup factors, pre-attached to the frame by
        # attach_htf (0 when the symbol has no weekly data)
        for fk, name in enumerate(FACTOR_NAMES[HTF_START:], start=HTF_START):
            if name in sig.columns:
                A["FACTORS"][pos, j, fk] = sig[name].to_numpy(float)
        # mainstream benchmark entries -- NEVER gated: the textbook strategy is
        # the honest comparison as it is actually taught, not a version we
        # quietly improved with our own trend filter.
        for mk, name in enumerate(MAINSTREAM_COLUMNS):
            if name in sig.columns:
                A["MS"][pos, j, mk] = sig[name].to_numpy(bool).astype(float)
    return A, V, ENT, SIDX, EXT, syms, grid, st_stop, st_hold, st_tgt


def attach_htf(frames):
    """Attach weekly SETUP factors to each intraday frame, reindexed with only
    weeks that have CLOSED (no lookahead -- and it respects "the weekly close
    is the truth"). Weekly bars are resampled from the 25yr daily data:
      htf_trend     : weekly EMA-stack alignment (close>20>50>100>200), -1..1
      htf_ema_dist  : distance above/below the reference weekly EMA, -1..1
      htf_fib_prox  : closeness to the nearest weekly fib retracement level
      htf_rci_trend : Paul's ichimoku/RCI trend read, computed ON THE WEEKLY
                      bars -- a screen candidate to test head-to-head against
                      the EMA stack ("which filter works best IS the test").
    These feed the higher-TF screen (which stocks are eligible)."""
    daily = load_universe("1d", None)
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last",
           "Volume": "sum"}
    names = FACTOR_NAMES[HTF_START:]
    for s, sig in frames.items():
        du = daily.get(s)
        if du is None or len(du) < WEEKLY_EMAS[-1]:
            for n in names:
                sig[n] = 0.0
            continue
        w = du.resample("W").agg(agg).dropna()
        c = w["Close"]
        emas = {n: c.ewm(span=n, adjust=False, min_periods=1).mean()
                for n in WEEKLY_EMAS}
        stack = [c] + [emas[n] for n in WEEKLY_EMAS]
        ups = sum((stack[i] > stack[i + 1]).astype(int)
                  for i in range(len(stack) - 1))
        wt = ups / (len(stack) - 1) * 2.0 - 1.0            # -1..1 EMA alignment
        eref = emas[HTF_EMA_REF]
        wd = ((c - eref) / eref / 0.5).clip(-1.0, 1.0)      # dist vs ref EMA
        p1, p2, _ = trend_points(w["High"], w["Low"], HTF_FIB_LR, HTF_FIB_LR)
        leg = (p2 - p1).where(lambda x: x > 0)
        wp = pd.Series(0.0, index=c.index)
        for r in (0.382, 0.5, 0.618, 0.786):
            d = (c - (p2 - r * leg)).abs() / leg
            wp = np.maximum(wp, (1.0 - d / FIB_PROX_BAND).clip(lower=0.0)
                            .fillna(0.0))
        # weekly RCI (ichimoku + regression channel) -- needs enough weekly
        # bars for the 52-period span plus its 25-bar displacement
        if len(w) >= LAGGING_SPAN2_PERIODS + DISPLACEMENT:
            wr = rci(w, bar_minutes=10080)["rci_trend"]
        else:
            wr = pd.Series(0.0, index=w.index)
        for n, wser in zip(names, (wt, wd, wp, wr)):
            sig[n] = wser.reindex(sig.index, method="ffill").fillna(0.0) \
                .to_numpy(float)
    return frames


def attach_fib_anchor(frames, anchor, fib_left, fib_right):
    """Attach the fib TREND ANCHOR (p1/p2/p3) taken from `anchor`'s timeframe,
    aligned onto each trading-TF frame.

    WHY THIS IS A DIMENSION AND NOT A CONSTANT (Paul, 2026-07-15): he anchors
    on the higher timeframes -- his BTC anchor (53,422 -> 109,028) was drawn
    once and its 0.5/0.618 were still catching price seven months later. The
    trading-TF anchor re-picks itself every ~35 daily bars, so its levels are
    ones he would never have drawn. Higher TF is therefore the default. But
    lower-TF anchors stay reachable because day-trading setups must be testable
    too -- which one wins is a question for the search, not an assumption.

      "self" : anchor on the trading timeframe (the old behaviour)
      "1d" / "1w" / "4h" ... : anchor on that timeframe, forward-filled

    NO LOOKAHEAD: a higher-TF anchor is knowable only at that bar's CLOSE, so
    its index is shifted by the bar's duration before the ffill -- the same rule
    apply_gates uses. Without the shift, a 15m bar at 10:00 would see the whole
    day's pivot.
    """
    if anchor == "self":
        return frames
    if anchor == "1w":
        base = load_universe("1d", None)
        dur = pd.Timedelta(days=7)
    else:
        base = load_universe(anchor, None)
        dur = GATE_DUR.get(anchor, pd.Timedelta(0))
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last",
           "Volume": "sum"}
    for s, sig in frames.items():
        bu = base.get(s)
        if bu is None or len(bu) < (fib_left + fib_right + 2):
            sig["fib_p1"] = np.nan
            sig["fib_p2"] = np.nan
            sig["fib_p3"] = np.nan
            continue
        hb = bu.resample("W").agg(agg).dropna() if anchor == "1w" else bu
        p1, p2, p3 = trend_points(hb["High"], hb["Low"], fib_left, fib_right)
        for nm, ser in (("fib_p1", p1), ("fib_p2", p2), ("fib_p3", p3)):
            shifted = pd.Series(ser.to_numpy(), index=ser.index + dur)
            shifted = shifted[~shifted.index.duplicated(keep="last")].sort_index()
            sig[nm] = shifted.reindex(sig.index, method="ffill").to_numpy(float)
    return frames


def select_universe(frames, universe):
    """Keep only the requested symbols (crypto and stocks live in one table).
    universe: 'stocks' (default; excludes crypto -USD pairs), 'crypto' (only
    -USD pairs), 'all', or an explicit comma list of symbols. Mixing 24/7
    crypto with session-based stocks in one grid is a mess, so pick one."""
    if universe in (None, "", "all"):
        return frames
    if universe == "crypto":
        keep = {s for s in frames if s.endswith("-USD")}
    elif universe == "stocks":
        keep = {s for s in frames if not s.endswith("-USD")}
    else:
        keep = set(universe.split(","))
    return {s: sig for s, sig in frames.items() if s in keep}


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
                 gtarget=None, atr_len=14, swing_look=20, cutoff=None,
                 fib=None, universe="stocks", gate_mode="hard",
                 fib_anchor="1d"):
    """All the expensive, param-independent work: load -> gate -> market ->
    build 2D grid. Do this ONCE, then sweep exit params via portsim.simulate
    on the returned arrays (each config = a numba call = milliseconds).
    Pass an explicit `cutoff` to override the months-derived window (the grid
    cache uses this so a hit is a byte-identical rebuild). `fib` is the
    build-time pivot config (left/right/stop_ratio/buf) for the FIB stop."""
    fib = {**FIB_DEFAULTS, **(fib or {})}
    gates = [tuple(g.split("@", 1)) for g in gate_arg.split(",")
             if "@" in g] if gate_arg != "none" else []
    if cutoff is None and months:
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=months)
    frames = select_universe(load_universe(interval, cutoff), universe)
    if gates:
        frames = apply_gates(frames, gates, None)
    frames = _apply_market(frames, market)
    frames = attach_htf(frames)
    frames = attach_fib_anchor(frames, fib_anchor, fib["left"], fib["right"])
    grid = build_grid(frames, strategies, exit_cols or [], gstop, ghold,
                      gtarget, atr_len, swing_look, fib["left"], fib["right"],
                      gate_mode=gate_mode)
    return grid, frames


GRID_CACHE_DIR = "grid_cache"
# bump when the set of arrays build_grid stores changes, so old-schema cache
# files (which would be missing a newly-added array) can't be loaded.
# v8: factor stack changed -- dip_bias renamed range_pos (honest name), rci_trend
# + rci_heat added as entry factors, htf_rci_trend added to the weekly screen.
# v9: A["MS"] mainstream-benchmark entry stack added (goal #2's third leg).
# v10: fib anchor is a DIMENSION (fib_anchor: self/4h/1d/1w) -- P1/P2/P3 can
#      come from a higher timeframe, so cached grids differ in what the fib
#      levels MEAN.
# Cached grids from v7/v8 have a different layout and MUST NOT be reused.
GRID_SCHEMA_VERSION = 10


def _grid_cache_path(strategies, interval, gate_arg, market, cutoff,
                     exit_cols, gstop, ghold, gtarget, atr_len, swing_look,
                     db_mtime, fib=None, universe="stocks", gate_mode="hard",
                     fib_anchor="1d"):
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
        "fib": {**FIB_DEFAULTS, **(fib or {})}, "schema": GRID_SCHEMA_VERSION,
        "universe": universe, "gate_mode": gate_mode,
        "fib_anchor": fib_anchor,
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
                        gtarget=None, atr_len=14, swing_look=20, fib=None,
                        universe="stocks", gate_mode="hard",
                        fib_anchor="1d"):
    """prepare_grid with a disk cache: pay the ~grid-build once per
    (strategies, gate, market, interval, window, params, DB-data) tuple, then
    every later sweep loads the 2D grid in ~a second. Returns
    (grid_tuple, cached_bool); does NOT return frames (sweeps don't need them).
    The window is normalized to the day so repeated same-day runs hit cache."""
    cutoff = ((pd.Timestamp.now().normalize() - pd.DateOffset(months=months))
              if months else None)
    path = _grid_cache_path(strategies, interval, gate_arg, market, cutoff,
                            exit_cols, gstop, ghold, gtarget, atr_len,
                            swing_look, int(os.path.getmtime(DB_PATH)), fib,
                            universe, gate_mode, fib_anchor)
    if os.path.exists(path):
        return _load_grid(path), True
    grid, _frames = prepare_grid(
        strategies, interval, gate_arg, market, months, exit_cols, gstop,
        ghold, gtarget, atr_len, swing_look, cutoff=cutoff, fib=fib,
        universe=universe, gate_mode=gate_mode, fib_anchor=fib_anchor)
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
    # max-hold time exit: OFF by default (0), overriding any hold baked into
    # the strategy file. Exits should be stops / targets / trailing / reversal
    # -- never a clock. Pass --hold=N only to deliberately test a time exit.
    ghold = int(arg(args, "hold", "0"))
    # stop placement: pct (fixed %), atr (entry - mult*ATR), swing (below the
    # recent swing low - buffer, i.e. under support), fib (below the fib
    # retracement of the last up-leg). atr/swing/fib avoid the noise-tagging
    # that a fixed 3% stop suffers.
    stop_mode = arg(args, "stop-mode", "pct")
    atr_len = int(arg(args, "atr-len", "14"))
    atr_mult = float(arg(args, "atr-mult", "2.5"))
    swing_look = int(arg(args, "swing-look", "20"))
    swing_buf = float(arg(args, "swing-buf", "0.005"))
    # fib stop: auto-fib the last confirmed up-leg (pivot low -> pivot high);
    # stop sits fib-buf below the fib-stop retracement (e.g. 0.786 = below the
    # 78.6% level). --fib-target uses the prior pivot high as the take-profit.
    # left/right (pivot window) are baked into the grid; stop-ratio/buf are
    # applied at sim time (cheaply sweepable).
    fib = {"left": int(arg(args, "fib-left", "10")),
           "right": int(arg(args, "fib-right", "10"))}
    fib_stop_ratio = float(arg(args, "fib-stop", "0.786"))
    fib_buf = float(arg(args, "fib-buf", "0.005"))
    use_fib_target = 1 if arg(args, "fib-target", "0") not in \
        ("0", "no", "false", "none", "") else 0
    # fib entry mode (off/zone/bounce/bounce-trend): filter or trigger entries
    # off the pullback into the [lo, hi] retracement band of the up-leg.
    # bounce needs a confirmed up-close off the band; bounce-trend makes that
    # bounce the entry itself, gated only by the higher-TF trend.
    fib_entry = arg(args, "fib-entry", "off")
    fib_zone_lo = float(arg(args, "fib-zone-lo", "0.5"))
    fib_zone_hi = float(arg(args, "fib-zone-hi", "0.786"))
    fib_bounce_look = int(arg(args, "fib-bounce-look", "3"))
    # confluence entry: enter when the weighted sum of factors clears a
    # threshold. --w-<factor> weights each factor (default 1); nothing is a
    # hard gate -- set a weight to 0 to mute a factor, tune to taste/sweep.
    conf_entry = 1 if ("--conf-entry" in args or arg(args, "conf-entry", "0")
                       not in ("0", "no", "false", "none", "")) else 0
    weights = np.array([float(arg(args, f"w-{n}", "1.0"))
                        for n in FACTOR_NAMES])
    conf_threshold = float(arg(args, "conf-threshold", "1.0"))
    conf_size = 1 if ("--conf-size" in args or arg(args, "conf-size", "0")
                      not in ("0", "no", "false", "none", "")) else 0
    # higher-TF (weekly) SETUP screen: a stock is eligible only if its weighted
    # weekly setup score (the htf_* factors) clears --htf-threshold. A tunable
    # screen -- weights + threshold decide what a "good weekly setup" is.
    htf_screen = 1 if ("--htf-screen" in args or arg(args, "htf-screen", "0")
                       not in ("0", "no", "false", "none", "")) else 0
    htf_threshold = float(arg(args, "htf-threshold", "0.0"))
    # trailing stop: once a trade is up +trail_act, ratchet the stop up. Mode
    # pct = trail_dist below the high-water mark; structure = under the last
    # confirmed swing low; fib = under each cleared fib/extension rung (protect
    # the level, let the winner run). None = off.
    trail_mode = arg(args, "trail-mode", "pct")
    fib_ext = [float(x) for x in arg(args, "fib-ext", "1.0,1.272,1.618,2.0")
               .split(",")]
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
    frames = select_universe(load_universe(interval, cutoff),
                             arg(args, "universe", "stocks"))
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
    if engine == "loop" and (stop_mode == "fib" or trail_mode == "structure"):
        fail("--engine=loop does not implement fib stops / structure trailing "
             "(numba-only); drop --engine=loop or use --stop-mode=pct/atr/swing")
    if engine == "numba":
        import time as _time
        _t = _time.time()
        frames = attach_htf(frames)
        A, V, ENT, SIDX, EXT, syms, master, st_stop, st_hold, st_tgt = \
            build_grid(frames, strategies, exit_cols, gstop, ghold, gtarget,
                       atr_len, swing_look, fib["left"], fib["right"])
        res = portsim.simulate(
            A["O"], A["H"], A["L"], A["C"], V, ENT, A["SCORE"], SIDX, EXT,
            A["ATR"], A["SW"], st_stop, st_hold, st_tgt,
            stop_mode=STOP_MODES.get(stop_mode, 0), atr_mult=atr_mult,
            swing_buf=swing_buf, trail_act=trail_act or 0.0,
            trail_dist=trail_dist, cost_side=cost_side, max_pos=max_pos,
            init_cash=capital, p1=A["P1"], p2=A["P2"], p3=A["P3"],
            fib_stop_ratio=fib_stop_ratio, fib_buf=fib_buf,
            trail_mode=TRAIL_MODES.get(trail_mode, 0), fib_ext=fib_ext,
            use_fib_target=use_fib_target, gate=A["GATE"],
            fib_entry=FIB_ENTRY_MODES.get(fib_entry, 0), fib_zone_lo=fib_zone_lo,
            fib_zone_hi=fib_zone_hi, fib_bounce_look=fib_bounce_look,
            factors=A["FACTORS"], weights=weights, conf_entry=conf_entry,
            conf_threshold=conf_threshold, conf_size=conf_size,
            htf_start=HTF_START, htf_screen=htf_screen,
            htf_threshold=htf_threshold)
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
            + (f"({fib_stop_ratio:.3f})" if stop_mode == "fib" else "")
            + (" tp=fib" if use_fib_target and stop_mode == "fib" else "")
            + (f" entry={fib_entry}[{fib_zone_lo:.3f}-{fib_zone_hi:.3f}]"
               if fib_entry != "off" else "")
            + (" conf[" + ",".join(f"{n}={w:g}" for n, w in
               zip(FACTOR_NAMES, weights)) + f"]>={conf_threshold:g}"
               + (" size~score" if conf_size else "") if conf_entry else "")
            + (f" htf-screen>={htf_threshold:g}" if htf_screen else "")
            + (f" trail={trail_mode}" if trail_mode in ("structure", "fib")
               else f" trail={trail_act:.0%}@{trail_dist:.0%}" if trail_act
               else "")
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
