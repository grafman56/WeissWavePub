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
    --hold=N        max bars held; 0 = no time exit [0]. A clock is not an
                    exit strategy -- use --exit=COLS (bearish reversal), --stop,
                    --target. Pass --hold=N only to TEST a time exit as a value.
    --interval=I    bar interval [1d]
    --symbols=A,B   restrict universe (quick runs) [all]
    --months=N      test only the last N months (plus indicator warm-up);
                    much faster, but no train/test split and low trade
                    counts - promote survivors with a full-history run [all]
    --gate=COL@IV   cross-timeframe trend gate: entries only allowed when
                    column COL was True on the PREVIOUS bar of interval IV
                    (e.g. --gate=minervini@1d with --interval=1h trades
                    hourly only inside a daily stage-2 uptrend).
                    [minervini@1d] -- ON BY DEFAULT, pass --gate=none to
                    trade ungated. This said [none] while the code defaulted
                    to minervini@1d, so every run carried a daily trend gate
                    nobody asked for: on heavy_buy 1d/12mo it is the
                    difference between n=4990 (ungated) and n=3017.
    --cost-bps=F    round-trip cost haircut in basis points applied to
                    every trade (spread+slippage; 10 = 0.10%) [0]

Prints train/test stats (70/30 per-symbol time split; test half is the
honesty check) including excess vs the equal-weight buy-and-hold benchmark.
Exit code 0 on success, 2 on bad arguments/config.
"""

import glob
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from weisswave.codesig import package_sig
from weisswave.db import DB_PATH, connect, list_symbols, load_prices
from weisswave.optimize import evaluate_config
from weisswave.signals import (FILTER_COLUMNS, SIGNAL_COLUMNS_BEAR,
                               SIGNAL_COLUMNS_BULL, build_signals)

STRATEGIES_PATH = "strategies.json"
RESULTS_LOG = os.path.join("agent-tasks", "results.log")
CACHE_DIR = "signals_cache"

# A manual escape hatch for invalidations that are NOT visible in our source:
# a duckdb/pandas upgrade that changes float output, a data backfill that
# rewrites history under an unchanged mtime. Bump it when the numbers should
# change and nothing in weisswave/ did.
#
# It no longer carries code changes -- SIGNAL_CODE_SIG does, automatically.
# This constant used to be the only guard, and its contract ("bump when a
# column is added or renamed") missed the failure that mattered: the rising()
# fix changed every volume column's VALUE and no column's NAME, so v3 was still
# honest and a pre-fix cache was still a valid hit. Guards a human has to
# remember are only as good as the last time someone remembered.
# v2 = RCI columns added (rci_bull/bear/breakup/breakdn/heat/...).
# v3 = ema50/ema200 columns added (factor definitions reference them).
SIGNALS_SCHEMA_VERSION = 3

# Hash of build_signals and every weisswave module it imports. Any edit to the
# math -- new column, renamed column, or the same columns with different values
# -- lands here and forces a rebuild.
SIGNAL_CODE_SIG = package_sig("signals")


def sig_params_sig(params):
    """Short signature of a build_signals parameter set. MUST be in the cache
    key: the params change what the columns MEAN, and a cache that ignores them
    serves frames built with different settings while reporting success. That
    is the same silent-wrong-answer failure as the schema bump."""
    if not params:
        return "def"
    return hashlib.sha1(json.dumps(params, sort_keys=True,
                                   default=str).encode()).hexdigest()[:8]


def stale_cache_paths(key, interval):
    """Caches to delete after writing `key`: the ones built against older DATA
    or older CODE. Siblings that differ ONLY by sig_params are kept -- they are
    every bit as valid as `key`, just for different indicator settings.

    This used to be `if old != key: remove`, correct back when exactly one cache
    per (interval, db_mtime, schema) could be legal. sig_params joined the key
    and made several legal at once, so a default run and a --space run each
    deleted the other's 360MB and paid a full ~4min rebuild on every
    alternation. _grid_cache_path's prune already had this right; this is the
    same prefix trick, which is the point -- one rule, not two.
    """
    keep = os.path.basename(key).rsplit("_", 1)[0]   # drop the sig_params token
    return [p for p in glob.glob(os.path.join(
                CACHE_DIR, f"signals_{interval.replace(':', '')}_*.parquet"))
            if not os.path.basename(p).startswith(keep)]


def load_universe(interval: str, cutoff, sig_params: dict = None) -> dict:
    """Signal frames for the whole universe via a parquet cache keyed on the DB
    file's mtime, the signal schema version AND the build_signals parameters:
    first call after a fetch (or a schema/param change) rebuilds (~2 min), every
    later call loads in seconds.

    `sig_params` is what makes the INDICATORS tunable. They were defaults buried
    three calls deep (load_universe -> build_signals -> combined_signals ->
    tdi_signals), so "make what fires the green candle more sensitive" was not
    expressible at all -- and the cache would have hidden any attempt."""
    key = os.path.join(CACHE_DIR,
                       f"signals_{interval.replace(':', '')}_"
                       f"{int(os.path.getmtime(DB_PATH))}_"
                       f"v{SIGNALS_SCHEMA_VERSION}_"
                       f"{SIGNAL_CODE_SIG}_"
                       f"{sig_params_sig(sig_params)}.parquet")
    if not os.path.exists(key):
        try:
            con = connect(read_only=True)
        except Exception:
            # DB is write-locked (backfill/fetch running). Iterate on the
            # newest existing cache rather than blocking the whole workflow.
            # Only THIS schema version AND this code signature: the wildcard is
            # on the DB mtime alone, so we fall back to older DATA built by
            # today's code, never to numbers this code would no longer produce.
            older = glob.glob(os.path.join(
                CACHE_DIR, f"signals_{interval.replace(':', '')}_*_"
                           f"v{SIGNALS_SCHEMA_VERSION}_"
                           f"{SIGNAL_CODE_SIG}_"
                           f"{sig_params_sig(sig_params)}.parquet"))
            if not older:
                fail(f"DB is locked (a fetch/backfill is running?) and no "
                     f"cached {interval} signals exist for schema "
                     f"v{SIGNALS_SCHEMA_VERSION} / code {SIGNAL_CODE_SIG}")
            key = max(older, key=os.path.getmtime)
            print(f"WARNING: DB busy - using STALE signal cache "
                  f"{os.path.basename(key)}; rerun after the fetch for "
                  f"fresh data")
            return _frames_from_parquet(key, cutoff)
        parts = []
        too_short = 0
        raised = 0
        first_err = None
        for s in list_symbols(con, interval):
            df = load_prices(con, s, interval)
            if len(df) < 300:
                too_short += 1
                continue
            try:
                sig = build_signals(df, **(sig_params or {}))
            except Exception as e:              # noqa: BLE001
                # Skipping ONE genuinely bad symbol is right; the universe
                # should not sink because a ticker has a broken bar. But the
                # same error on EVERY symbol is a params/code bug, not bad
                # data -- and this swallow used to report it as "no usable
                # data in market.duckdb", sending you to inspect a database
                # that was never wrong. Keep the first error and account for
                # it below.
                raised += 1
                if first_err is None:
                    first_err = e
                continue
            sig.index.name = "ts"
            sig["symbol"] = s
            parts.append(sig.reset_index())
        con.close()
        if not parts:
            if first_err is not None:
                fail(f"build_signals raised on all {raised} symbols that have "
                     f"usable {interval} data. That is a params/code error, "
                     f"not the DB. First: "
                     f"{type(first_err).__name__}: {first_err}")
            fail(f"no usable {interval} data in {DB_PATH} ({too_short} symbols "
                 f"under the {300}-bar minimum)")
        if raised:
            # a PARTIAL silent failure is the same disease, just survivable
            print(f"WARNING: build_signals raised on {raised}/"
                  f"{raised + len(parts)} symbols; first: "
                  f"{type(first_err).__name__}: {first_err}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        for old in stale_cache_paths(key, interval):
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
    # max-hold time exit: OFF by default. Exits should be stops / targets /
    # trailing / a bearish reversal signal -- never a clock. Selling a winner
    # because days elapsed is not an exit strategy. Pass --hold=N only to
    # deliberately TEST a time exit as a value.
    #
    # This defaulted to 20 while portfolio_multi (fixed 2026-07-13) defaults to
    # 0, so the same strategy scored differently depending on which tool ran it,
    # and this one was quietly amputating the runners: on tdi_long 1d/12mo,
    # hold=20 gives avg +1.51%/trade PF 1.58, hold=0 gives avg +5.41% PF 2.11.
    # The clock raises the WIN RATE (53% vs 39%) by banking small wins early,
    # which is exactly how it hides.
    hold = int(arg(args, "hold", "0"))
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
    # This harness does not expose the indicator params yet; both paths below
    # must therefore build with the SAME (default) settings. Naming it once and
    # passing it to both is what keeps them from drifting -- and line 385 used
    # to reference this without it existing anywhere, which is the bug below.
    sig_params = None

    if sym_arg:
        # explicit symbol list: small, build directly (no cache round-trip)
        con = connect(read_only=True)
        frames = {}
        skipped = []
        first_err = None
        for s in [x.strip().upper() for x in sym_arg.split(",")]:
            df = load_prices(con, s, interval)
            if len(df) < 300:
                skipped.append(f"{s}: only {len(df)} {interval} bars (<300)")
                continue
            try:
                sig = build_signals(df, **(sig_params or {}))
            except Exception as e:                       # noqa: BLE001
                # This swallowed a NameError -- `sig_params` was referenced here
                # and defined nowhere, so --symbols raised on EVERY symbol, for
                # every interval, and reported "no usable data". The flag was
                # dead from the commit that introduced sig_params and nothing
                # said so: a bad request and a broken code path looked
                # identical. You ASKED for these symbols by name; if we cannot
                # build them, say why.
                if first_err is None:
                    first_err = e
                skipped.append(f"{s}: {type(e).__name__}: {e}")
                continue
            if cutoff is not None:
                sig = sig[sig.index >= cutoff]
                if len(sig) < 20:
                    skipped.append(f"{s}: only {len(sig)} bars after the "
                                   f"--months cutoff (<20)")
                    continue
            frames[s] = sig
        con.close()
        if not frames:
            fail(f"none of the requested symbols produced usable {interval} "
                 f"data:\n  " + "\n  ".join(skipped))
        if skipped:
            print("WARNING: skipped " + "; ".join(skipped))
    else:
        frames = load_universe(interval, cutoff)
    if not frames:
        fail(f"no usable {interval} data for the requested symbols")

    req_filter = filter_col          # what was ASKED for; filter_col is about
                                     # to become an internal column name
    if gates:
        # apply_gates ANDs filter_col into xtf_gate, so the request is honoured
        # -- but the name it is honoured under is no longer the user's.
        frames = apply_gates(frames, gates, filter_col)
        filter_col = "xtf_gate"
        if not frames:
            fail(f"gate columns {[c for c, _ in gates]} not available")

    trades = evaluate_config(frames, entry_cols, min_count, window,
                             filter_col, exit_cols, stop, hold,
                             weights=weights, take_profit=target)
    wtxt = ("+".join(f"{c}x{weights.get(c, 1)}" for c in entry_cols)
            if weights else "+".join(entry_cols))
    # Report BOTH, always, under their own names, and report the filter the
    # USER asked for -- not `xtf_gate`, the internal column apply_gates ANDs it
    # into. This printed `filter=<gate_arg> if gates else <filter_col>`: one
    # label for two different things, and since the gate is ON BY DEFAULT an
    # applied --filter was never shown at all. `--filter=above_50ma` cut
    # heavy_buy from 3017 to 2319 trades while the header still read
    # `filter=minervini@1d`. Both were applied; one was invisible. A
    # backtester that misreports what it applied is the same failure as one
    # that reports trades which never fired.
    lines = [f"strategy: {wtxt} (score>={min_count} in {window} "
             f"bars)  gate={gate_arg if gates else 'none'}  "
             f"filter={req_filter or 'none'}  "
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
