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

Prints train/test stats (70/30 per-symbol time split; test half is the
honesty check) including excess vs the equal-weight buy-and-hold benchmark.
Exit code 0 on success, 2 on bad arguments/config.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np

from weisswave.db import connect, list_symbols, load_prices
from weisswave.optimize import evaluate_config
from weisswave.signals import (FILTER_COLUMNS, SIGNAL_COLUMNS_BEAR,
                               SIGNAL_COLUMNS_BULL, build_signals)

STRATEGIES_PATH = "strategies.json"
RESULTS_LOG = os.path.join("agent-tasks", "results.log")


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

    exit_arg = arg(args, "exit", None)
    exit_cols = [] if exit_arg in (None, "none", "") else exit_arg.split(",")
    hold = int(arg(args, "hold", "20"))
    interval = arg(args, "interval", "1d")
    sym_arg = arg(args, "symbols", None)

    known = set(SIGNAL_COLUMNS_BULL + SIGNAL_COLUMNS_BEAR + FILTER_COLUMNS)
    for c in entry_cols + exit_cols + ([filter_col] if filter_col else []):
        if c not in known:
            fail(f"unknown signal column '{c}' (see --list-signals)")

    con = connect(read_only=True)
    symbols = ([s.strip().upper() for s in sym_arg.split(",")] if sym_arg
               else list_symbols(con, interval))
    frames = {}
    for s in symbols:
        df = load_prices(con, s, interval)
        if len(df) < 300:
            continue
        try:
            frames[s] = build_signals(df)
        except Exception:
            pass
    con.close()
    if not frames:
        fail(f"no usable {interval} data for the requested symbols")

    trades = evaluate_config(frames, entry_cols, min_count, window,
                             filter_col, exit_cols, stop, hold)
    lines = [f"strategy: {'+'.join(entry_cols)} (>={min_count} in {window} "
             f"bars)  filter={filter_col or 'none'}  "
             f"exit={'+'.join(exit_cols) or 'none'}  stop={stop:.0%}  "
             f"hold={hold}  interval={interval}  universe={len(frames)}"]
    if trades.empty:
        emit(lines + ["no trades triggered"])
        return
    lines += [half_line(trades, "train"), half_line(trades, "test")]
    xs = trades.loc[trades["half"] == "test", "excess"]
    if len(xs):
        verdict = ("BEATS buy-and-hold out of sample" if xs.mean() > 0
                   else "does NOT beat buy-and-hold out of sample")
        lines.append(f"verdict: {verdict} ({xs.mean():+.2%}/trade excess, "
                     f"n={len(xs)})")
    emit(lines)


if __name__ == "__main__":
    main()
