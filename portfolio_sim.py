#!/usr/bin/env python3
"""Portfolio-level simulator: the reality check between per-trade edge and
what an account actually earns.

Runs a strategy with FINITE capital and a position cap: when more setups
fire than there are free slots, candidates are ranked by their weighted
confluence score and the rest are skipped — exactly the choice a live
account faces. One equity curve, compounding, next-open fills, per-side
cost haircut.

Usage (mirrors test_strategy.py):
    python portfolio_sim.py --saved=tdi-adp-confluence --max-positions=5
    python portfolio_sim.py tdi_long,adp_bull_div --min-count=2 \
        --filter=in_up_wave --stop=0.08 --hold=20 --cost-bps=10

Extra options:
    --max-positions=N   concurrent position cap [5]
    --capital=F         starting capital [100000]
    --gate=COL@IV       cross-timeframe trend gate: entries only allowed
                        when column COL was True on the PREVIOUS bar of
                        interval IV (e.g. --gate=minervini@1d with
                        --interval=1h day-trades hourly inside a daily
                        uptrend). Same no-lookahead shift as the harness.

Exits: stop-loss (low touch, filled at stop or gap open) and time (hold
bars). Ranking: weighted confluence score at the signal bar, ties by
symbol. Benchmark: equal-weight buy-and-hold of the same universe.
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd

from test_strategy import (STRATEGIES_PATH, apply_gates, arg, emit, fail,
                           load_universe)
from weisswave.optimize import benchmark_index
from weisswave.signals import (SIGNAL_COLUMNS_BEAR, SIGNAL_COLUMNS_BULL,
                               FILTER_COLUMNS, combine_signals, recent)


def parse_config(args):
    import json
    import os
    saved = arg(args, "saved", None)
    if saved:
        if not os.path.exists(STRATEGIES_PATH):
            fail(f"--saved given but no {STRATEGIES_PATH}")
        with open(STRATEGIES_PATH, encoding="utf-8") as f:
            cfg = next((s for s in json.load(f) if s["name"] == saved), None)
        if cfg is None:
            fail(f"no saved strategy named '{saved}'")
        return (list(cfg["entry_cols"]), int(cfg.get("min_count", 1)),
                int(cfg.get("window", 5)), cfg.get("filter"),
                float(cfg.get("stop_pct", 0.08)), cfg.get("weights"))
    positional = [a for a in args if not a.startswith("--")]
    if not positional:
        fail("give entry columns or --saved=NAME")
    filter_col = arg(args, "filter", None)
    if filter_col in ("none", ""):
        filter_col = None
    w_arg = arg(args, "weights", None)
    weights = ({p.split(":")[0]: int(p.split(":")[1])
                for p in w_arg.split(",")} if w_arg else None)
    return (positional[0].split(","), int(arg(args, "min-count", "1")),
            int(arg(args, "window", "5")), filter_col,
            float(arg(args, "stop", "0.08")), weights)


def main():
    args = sys.argv[1:]
    entry_cols, min_count, window, filter_col, stop, weights = \
        parse_config(args)
    hold = int(arg(args, "hold", "20"))
    interval = arg(args, "interval", "1d")
    months = int(arg(args, "months", "0"))
    cost_side = float(arg(args, "cost-bps", "0")) / 10000.0 / 2
    max_pos = int(arg(args, "max-positions", "5"))
    capital = float(arg(args, "capital", "100000"))
    gate_arg = arg(args, "gate", "minervini@1d")    # trend gate ON by default
    target = arg(args, "target", None)
    target = float(target) if target not in (None, "none", "") else None
    exit_arg = arg(args, "exit", None)
    exit_cols = [] if exit_arg in (None, "none", "") else exit_arg.split(",")
    gates = []                      # [(col, interval), ...]; all ANDed
    if gate_arg and gate_arg != "none":
        for g in gate_arg.split(","):
            if "@" not in g:
                fail("--gate needs COL@INTERVAL[,COL@INTERVAL...], "
                     "e.g. minervini@1d,above_50ma@4h")
            gates.append(tuple(g.split("@", 1)))

    known = set(SIGNAL_COLUMNS_BULL + SIGNAL_COLUMNS_BEAR + FILTER_COLUMNS)
    for c in entry_cols + exit_cols + ([filter_col] if filter_col else []) \
            + [gc for gc, _ in gates]:
        if c not in known:
            fail(f"unknown signal column '{c}'")

    cutoff = (pd.Timestamp.now() - pd.DateOffset(months=months)
              if months else None)
    frames = load_universe(interval, cutoff)

    if gates:
        frames = apply_gates(frames, gates, filter_col)
        filter_col = "xtf_gate"
        if not frames:
            fail(f"gate columns {[c for c, _ in gates]} not available")

    # per-symbol arrays
    data = {}
    for s, sig in frames.items():
        ent = combine_signals(sig, entry_cols, min_count, window, weights)
        if filter_col:
            ent = ent & sig[filter_col].astype(bool)
        w = {c: int((weights or {}).get(c, 1)) for c in entry_cols
             if c in sig.columns}
        score = sum(recent(sig[c], window).astype(int) * w[c] for c in w)
        ext = (sig[[c for c in exit_cols if c in sig.columns]].astype(bool)
               .any(axis=1).to_numpy(bool) if exit_cols
               else np.zeros(len(sig), bool))
        data[s] = {"idx": sig.index.values,
                   "open": sig["Open"].to_numpy(float),
                   "low": sig["Low"].to_numpy(float),
                   "high": sig["High"].to_numpy(float),
                   "close": sig["Close"].to_numpy(float),
                   "ent": ent.to_numpy(bool), "ext": ext,
                   "score": np.asarray(score, float), "ptr": 0}

    master = np.array(sorted(set(np.concatenate(
        [d["idx"] for d in data.values()]))))

    cash = capital
    positions = {}                 # sym -> dict(shares, stop_px, held, entry)
    trades = []
    equity_curve = np.empty(len(master))
    invested_frac = np.empty(len(master))

    for ti, t in enumerate(master):
        # advance pointers to today's bar where it exists
        today = {}
        for s, d in data.items():
            p = d["ptr"]
            if p < len(d["idx"]) and d["idx"][p] == t:
                today[s] = p
                d["ptr"] = p + 1

        # exits first (stop touch, then time)
        for s in list(positions):
            if s not in today:
                continue
            k = today[s]
            pos = positions[s]
            pos["held"] += 1
            d = data[s]
            exit_px = None
            if d["low"][k] <= pos["stop_px"]:
                exit_px, reason = min(d["open"][k], pos["stop_px"]), "stop"
            elif d["high"][k] >= pos["tp_px"]:
                exit_px, reason = max(d["open"][k], pos["tp_px"]), "target"
            elif k > 0 and d["ext"][k - 1]:      # exit signal prior close -> open
                exit_px, reason = d["open"][k], "signal"
            elif pos["held"] >= hold:
                exit_px, reason = d["open"][k], "time"
            if exit_px is not None:
                exit_px *= (1 - cost_side)
                cash += pos["shares"] * exit_px
                trades.append({"symbol": s, "ret":
                               exit_px / pos["entry"] - 1, "reason": reason})
                del positions[s]

        # entries: signal on the symbol's PREVIOUS bar, ranked by score
        if len(positions) < max_pos:
            cands = []
            for s, k in today.items():
                if s in positions or k == 0:
                    continue
                d = data[s]
                if d["ent"][k - 1]:
                    cands.append((-d["score"][k - 1], s, k))
            cands.sort()
            equity_now = cash + sum(
                p["shares"] * data[s2]["close"][today.get(s2, data[s2]["ptr"] - 1)]
                for s2, p in positions.items())
            for _, s, k in cands:
                if len(positions) >= max_pos:
                    break
                alloc = min(equity_now / max_pos, cash)
                if alloc <= 0:
                    break
                px = data[s]["open"][k] * (1 + cost_side)
                if px <= 0 or not np.isfinite(px):
                    continue
                shares = alloc / px
                cash -= shares * px
                base_px = data[s]["open"][k]
                positions[s] = {"shares": shares, "entry": px,
                                "stop_px": base_px * (1 - stop),
                                "tp_px": base_px * (1 + target) if target
                                else float("inf"),
                                "held": 0}

        mkt_val = sum(p["shares"] * data[s]["close"][today[s]]
                      if s in today else
                      p["shares"] * data[s]["close"][data[s]["ptr"] - 1]
                      for s, p in positions.items())
        equity_curve[ti] = cash + mkt_val
        invested_frac[ti] = mkt_val / equity_curve[ti] if equity_curve[ti] else 0

    eq = pd.Series(equity_curve, index=pd.DatetimeIndex(master))
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / capital) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    bench = benchmark_index(frames)
    bench = bench / bench.iloc[0] * capital
    r = pd.Series([t["ret"] for t in trades])

    ftxt = gate_arg if gates else (filter_col or "none")
    lines = [
        f"portfolio: {'+'.join(entry_cols)} (score>={min_count} in {window}) "
        f"gate={ftxt} stop={stop:.0%}"
        + (f" target={target:.0%}" if target else "")
        + (f" exit={'+'.join(exit_cols)}" if exit_cols else "")
        + f" hold={hold} interval={interval} max_pos={max_pos} "
        f"cost={cost_side * 2 * 10000:.0f}bps"
        + (f" window=last {months}mo" if months else ""),
        f"period: {eq.index[0].date()} -> {eq.index[-1].date()}  "
        f"universe={len(frames)}",
        f"final equity: {eq.iloc[-1]:,.0f}  (start {capital:,.0f})  "
        f"CAGR={cagr:+.1%}  maxDD={dd:.1%}  avg_invested={invested_frac.mean():.0%}",
        f"benchmark (eq-weight B&H): {bench.iloc[-1]:,.0f}  "
        f"strategy vs benchmark: {eq.iloc[-1] - bench.iloc[-1]:+,.0f}",
        f"trades: n={len(r)}  win={(r > 0).mean():.1%}  avg={r.mean():+.2%}  "
        f"open at end: {len(positions)}",
        f"verdict: {'BEATS' if eq.iloc[-1] > bench.iloc[-1] else 'does NOT beat'} "
        f"buy-and-hold as a {max_pos}-slot portfolio",
    ]
    emit(lines)


if __name__ == "__main__":
    main()
