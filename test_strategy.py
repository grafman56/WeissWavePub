#!/usr/bin/env python3
"""One-command strategy backtest harness.

Built so an orchestrator (human, script, or LLM agent) can test a strategy
idea with a single command — no fetching, no installs, no notebook. Uses
the existing market.duckdb; run fetch_data.py separately when data is
actually stale.

Usage:
    python test_strategy.py tdi_long,adp_bull_div --min-count=2 --filter=in_up_wave
    python test_strategy.py --saved=tdi-adp-confluence
    python test_strategy.py --list-signals          # discover valid names
    python test_strategy.py --list-saved            # strategies.json entries

Options (defaults in brackets):
    --min-count=N   distinct entry signals required in window [1]
    --window=N      confluence window in bars [5]
    --filter=COL    regime gate column, or none [none]
    --exit=COLS     comma-separated exit signal columns, or none [none]
    --stop=F        stop-loss fraction [0.08]
    --hold=N        max bars held [20]
    --interval=I    bar interval [1d]
    --symbols=A,B   restrict universe (quick runs) [all]
    --months=N      test only the last N months (plus indicator warm-up);
                    much faster, but no train/test split and low trade
                    counts - promote survivors with a full-history run [all]
    --gate=COL@IV   cross-timeframe trend gate: entries only allowed when
                    column COL was True on the PREVIOUS bar of interval IV
                    (e.g. --gate=minervini@1d with --interval=1h trades
                    hourly only inside a daily stage-2 uptrend) [none]
    --cost-bps=F    round-trip cost haircut in basis points applied to
                    every trade (spread+slippage; 10 = 0.10%) [0]

Prints train/test stats (70/30 per-symbol time split; test half is the
honesty check) including excess vs the equal-weight buy-and-hold benchmark.
Exit code 0 on success, 2 on bad arguments/config.
"""

import json
import os
import sys
from datetime import datetime

import glob

import numpy as np
import pandas as pd

from weisswave.db import DB_PATH, connect, list_symbols, load_prices
from weisswave.optimize import evaluate_config
from weisswave.signals import (FILTER_COLUMNS, SIGNAL_COLUMNS_BEAR,
                               SIGNAL_COLUMNS_BULL, build_signals)

STRATEGIES_PATH = "strategies.json"
RESULTS_LOG = os.path.join("agent-tasks", "results.log")
CACHE_DIR = "signals_cache"

# BUMP THIS whenever build_signals gains/renames/changes a column. The cache
# was keyed on the DB mtime ALONE, so a signal-code change did not invalidate
# it: a stale cache would quietly serve frames missing the new columns, and
# apply_gates DROPS any symbol whose gate column is absent -- i.e. asking for a
# brand-new gate would silently empty the universe instead of erroring.
# v2 = RCI columns added (rci_bull/bear/breakup/breakdn/heat/...).
# v3 = ema50/ema200 columns added (factor definitions reference them).
SIGNALS_SCHEMA_VERSION = 3


def load_universe(interval: str, cutoff) -> dict:
    """Signal frames for the whole universe via a parquet cache keyed on the DB
    file's mtime AND the signal schema version: first call after a fetch (or a
    schema bump) rebuilds (~2 min), every later call loads in seconds.
    build_signals default parameters only."""
    key = os.path.join(CACHE_DIR,
                       f"signals_{interval.replace(':', '')}_"
                       f"{int(os.path.getmtime(DB_PATH))}_"
                       f"v{SIGNALS_SCHEMA_VERSION}.parquet")
    if not os.path.exists(key):
        try:
            con = connect(read_only=True)
        except Exception:
            # DB is write-locked (backfill/fetch running). Iterate on the
            # newest existing cache rather than blocking the whole workflow.
            # only THIS schema version: an older-schema cache lacks columns the
            # caller may be asking for, and a missing gate column silently
            # drops symbols rather than erroring
            older = glob.glob(os.path.join(
                CACHE_DIR, f"signals_{interval.replace(':', '')}_*_"
                           f"v{SIGNALS_SCHEMA_VERSION}.parquet"))
            if not older:
                fail(f"DB is locked (a fetch/backfill is running?) and no "
                     f"cached {interval} signals exist for schema "
                     f"v{SIGNALS_SCHEMA_VERSION}")
            key = max(older, key=os.path.getmtime)
            print(f"WARNING: DB busy - using STALE signal cache "
                  f"{os.path.basename(key)}; rerun after the fetch for "
                  f"fresh data")
            return _frames_from_parquet(key, cutoff)
        parts = []
        for s in list_symbols(con, interval):
            df = load_prices(con, s, interval)
            if len(df) < 300:
                continue
            try:
                sig = build_signals(df)
            except Exception:
                continue
            sig.index.name = "ts"
            sig["symbol"] = s
            parts.append(sig.reset_index())
        con.close()
        if not parts:
            fail(f"no usable {interval} data in {DB_PATH}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        for old in glob.glob(os.path.join(          # incl. older schemas
                CACHE_DIR, f"signals_{interval.replace(':', '')}_*.parquet")):
            if old != key:
                os.remove(old)
        pd.concat(parts, ignore_index=True).to_parquet(key)
    return _frames_from_parquet(key, cutoff)


def _frames_from_parquet(key: str, cutoff) -> dict:
    pooled = pd.read_parquet(key)
    frames = {}
    for s, g in pooled.groupby("symbol"):
        g = g.drop(columns="symbol").set_index("ts")
        if cutoff is not None:
            g = g[g.index >= cutoff]
            if len(g) < 20:
                continue
        frames[s] = g
    return frames


def emit(lines):
    """Print result lines and append them to the results log, so outcomes
    survive even when an orchestrating agent fumbles the relay."""
    text = "\n".join(lines)
    print(text)
    try:
        os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
        with open(RESULTS_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]\n{text}\n\n")
    except OSError:
        pass                      # logging must never break the run


def arg(args, name, default):
    return next((a.split("=", 1)[1] for a in args
                 if a.startswith(f"--{name}=")), default)


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(2)


def half_line(trades, half):
    t = trades[trades["half"] == half]
    if not len(t):
        return f"{half:5s}  no trades"
    r = t["ret"]
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    return (f"{half:5s}  n={len(t):5d}  win={(r > 0).mean():6.1%}  "
            f"avg={r.mean():+7.2%}  excess={t['excess'].mean():+7.2%}  "
            f"PF={pf:5.2f}")


GATE_DUR = {"1d": pd.Timedelta(days=1), "4h": pd.Timedelta(hours=4),
            "1h": pd.Timedelta(hours=1), "15m": pd.Timedelta(minutes=15),
            "5m": pd.Timedelta(minutes=5), "1m": pd.Timedelta(minutes=1)}


def apply_gates(frames: dict, gates: list, filter_col):
    """Stack of higher-timeframe trend gates (trade with the trend at every
    level of the cascade). gates = [(col, interval), ...]; an entry bar is
    allowed only when EVERY gate is True on the most recent higher-TF bar
    that CLOSED before it (no lookahead). Returns the gated frames; the
    combined gate lands in an 'xtf_gate' column."""
    gframes = {iv: load_universe(iv, None) for iv in {iv for _, iv in gates}}
    out = {}
    for s, sig in frames.items():
        ok = np.ones(len(sig), bool)
        usable = True
        for col, iv in gates:
            gu = gframes[iv].get(s)
            if gu is None or col not in gu.columns:
                usable = False
                break
            gser = gu[col].astype(bool)
            # value is knowable only at the gate bar's CLOSE (start + duration)
            gclose = pd.Series(gser.to_numpy(),
                               index=gser.index + GATE_DUR.get(iv, pd.Timedelta(0)))
            gclose = gclose[~gclose.index.duplicated(keep="last")].sort_index()
            mapped = gclose.reindex(sig.index, method="ffill").fillna(False)
            ok &= mapped.to_numpy(bool)
        if not usable:
            continue
        sig = sig.copy()
        sig["xtf_gate"] = ok
        if filter_col:
            sig["xtf_gate"] &= sig[filter_col].astype(bool)
        out[s] = sig
    return out


def main():
    args = sys.argv[1:]

    if "--list-signals" in args:
        print("BULL:", ", ".join(SIGNAL_COLUMNS_BULL))
        print("BEAR:", ", ".join(SIGNAL_COLUMNS_BEAR))
        print("FILTERS:", ", ".join(FILTER_COLUMNS))
        return
    if "--list-saved" in args:
        if not os.path.exists(STRATEGIES_PATH):
            print("(no strategies.json)")
            return
        with open(STRATEGIES_PATH, encoding="utf-8") as f:
            for s in json.load(f):
                print(f"{s['name']}: {'+'.join(s['entry_cols'])} "
                      f">={s.get('min_count', 1)} in {s.get('window', 5)} "
                      f"filter={s.get('filter') or 'none'}")
        return

    saved = arg(args, "saved", None)
    if saved:
        if not os.path.exists(STRATEGIES_PATH):
            fail(f"--saved given but no {STRATEGIES_PATH}")
        with open(STRATEGIES_PATH, encoding="utf-8") as f:
            cfg = next((s for s in json.load(f) if s["name"] == saved), None)
        if cfg is None:
            fail(f"no saved strategy named '{saved}' (see --list-saved)")
        entry_cols = list(cfg["entry_cols"])
        min_count = int(cfg.get("min_count", 1))
        window = int(cfg.get("window", 5))
        filter_col = cfg.get("filter")
        stop = float(cfg.get("stop_pct", 0.08))
        weights = cfg.get("weights")
    else:
        positional = [a for a in args if not a.startswith("--")]
        if not positional:
            fail("give entry columns (comma-separated), --saved=NAME, "
                 "--list-signals or --list-saved")
        entry_cols = positional[0].split(",")
        min_count = int(arg(args, "min-count", "1"))
        window = int(arg(args, "window", "5"))
        filter_col = arg(args, "filter", None)
        if filter_col in ("none", ""):
            filter_col = None
        stop = float(arg(args, "stop", "0.08"))
        w_arg = arg(args, "weights", None)      # e.g. tdi_long:2,golden:1
        weights = ({p.split(":")[0]: int(p.split(":")[1])
                    for p in w_arg.split(",")} if w_arg else None)

    exit_arg = arg(args, "exit", None)
    exit_cols = [] if exit_arg in (None, "none", "") else exit_arg.split(",")
    hold = int(arg(args, "hold", "20"))
    interval = arg(args, "interval", "1d")
    sym_arg = arg(args, "symbols", None)
    months = int(arg(args, "months", "0"))
    cost = float(arg(args, "cost-bps", "0")) / 10000.0
    # trend gate is ON by default (trade with the trend); --gate=none to disable
    gate_arg = arg(args, "gate", "minervini@1d")
    target = arg(args, "target", None)
    target = float(target) if target not in (None, "none", "") else None
    gates = []                      # [(col, interval), ...]; all ANDed
    if gate_arg and gate_arg != "none":
        for g in gate_arg.split(","):
            if "@" not in g:
                fail("--gate needs COL@INTERVAL[,COL@INTERVAL...], "
                     "e.g. minervini@1d,above_50ma@4h")
            gates.append(tuple(g.split("@", 1)))

    known = set(SIGNAL_COLUMNS_BULL + SIGNAL_COLUMNS_BEAR + FILTER_COLUMNS)
    for c in entry_cols + exit_cols + ([filter_col] if filter_col else []) \
            + [gc for gc, _ in gates] + list(weights or {}):
        if c not in known:
            fail(f"unknown signal column '{c}' (see --list-signals)")

    cutoff = (pd.Timestamp.now() - pd.DateOffset(months=months)
              if months else None)
    if sym_arg:
        # explicit symbol list: small, build directly (no cache round-trip)
        con = connect(read_only=True)
        frames = {}
        for s in [x.strip().upper() for x in sym_arg.split(",")]:
            df = load_prices(con, s, interval)
            if len(df) < 300:
                continue
            try:
                sig = build_signals(df)
            except Exception:
                continue
            if cutoff is not None:
                sig = sig[sig.index >= cutoff]
                if len(sig) < 20:
                    continue
            frames[s] = sig
        con.close()
    else:
        frames = load_universe(interval, cutoff)
    if not frames:
        fail(f"no usable {interval} data for the requested symbols")

    if gates:
        frames = apply_gates(frames, gates, filter_col)
        filter_col = "xtf_gate"
        if not frames:
            fail(f"gate columns {[c for c, _ in gates]} not available")

    trades = evaluate_config(frames, entry_cols, min_count, window,
                             filter_col, exit_cols, stop, hold,
                             weights=weights, take_profit=target)
    wtxt = ("+".join(f"{c}x{weights.get(c, 1)}" for c in entry_cols)
            if weights else "+".join(entry_cols))
    ftxt = gate_arg if gates else (filter_col or "none")
    lines = [f"strategy: {wtxt} (score>={min_count} in {window} "
             f"bars)  filter={ftxt}  "
             f"exit={'+'.join(exit_cols) or 'none'}  stop={stop:.0%}  "
             f"hold={hold}  interval={interval}  universe={len(frames)}"
             + (f"  target={target:.0%}" if target else "")
             + (f"  window=last {months}mo" if months else "")
             + (f"  cost={cost * 10000:.0f}bps" if cost else "")]
    if trades.empty:
        emit(lines + ["no trades triggered"])
        return
    if cost:
        trades["ret"] -= cost
        trades["excess"] -= cost

    if months:
        # short window: no meaningful train/test split — report all trades
        r, xs = trades["ret"], trades["excess"]
        wins = r[r > 0].sum()
        losses = -r[r < 0].sum()
        pf = wins / losses if losses > 0 else float("inf")
        lines.append(f"all    n={len(r):5d}  win={(r > 0).mean():6.1%}  "
                     f"avg={r.mean():+7.2%}  excess={xs.mean():+7.2%}  "
                     f"PF={pf:5.2f}")
        beat = "BEAT" if xs.mean() > 0 else "did NOT beat"
        lines.append(f"verdict: {beat} buy-and-hold over the last {months}mo "
                     f"({xs.mean():+.2%}/trade excess, n={len(r)}) - "
                     f"in-sample regime check only"
                     + ("; LOW N, indicative at best" if len(r) < 100 else ""))
    else:
        lines += [half_line(trades, "train"), half_line(trades, "test")]
        xs = trades.loc[trades["half"] == "test", "excess"]
        if len(xs):
            verdict = ("BEATS buy-and-hold out of sample" if xs.mean() > 0
                       else "does NOT beat buy-and-hold out of sample")
            lines.append(f"verdict: {verdict} ({xs.mean():+.2%}/trade "
                         f"excess, n={len(xs)})")
    emit(lines)


if __name__ == "__main__":
    main()
