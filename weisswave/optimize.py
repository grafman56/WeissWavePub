"""Automated strategy search over the signal suite.

Two-stage design, built to resist curve-fitting:

Stage 1 — entry screening (vectorized, cheap).
    Every entry-signal combination (singles and pairs), optionally gated by
    a regime filter, is scored on the TRAIN portion of the data by mean
    forward return versus the all-bars baseline. Combos with too few events
    are discarded; the rest are ranked by edge.

Stage 2 — full simulation on the survivors.
    The top combos are run through the sequential trade simulator across a
    grid of exits, stop-losses, and max-holding times — separately on the
    train and test halves. Rankings come from TRAIN performance; the test
    columns are the honesty check. A config is only credible when the test
    side is also positive with a reasonable trade count.
"""

import itertools

import numpy as np
import pandas as pd

from .signals import (FILTER_COLUMNS, SIGNAL_COLUMNS_BULL, combine_signals,
                      recent)
from .study import _simulate, backtest_long

# Exit rule menu for stage 2: name -> bear signal columns (empty = exit only
# via stop / max-hold).
EXIT_OPTIONS = {
    "none": [],
    "wt_cross_down": ["wt_cross_down"],
    "wt_cross_down_overbought": ["wt_cross_down_overbought"],
    "wtv_sell": ["wtv_sell"],
    "volume_cross_down": ["volume_cross_down"],
}


def benchmark_index(frames: dict) -> pd.Series:
    """Equal-weight buy-and-hold index of the loaded universe: the average
    per-bar return across symbols, compounded. This is 'what the market
    did' for benchmarking strategies over the same window."""
    closes = pd.DataFrame({s: f["Close"] for s, f in frames.items()})
    rets = closes.pct_change().mean(axis=1)
    return (1 + rets.fillna(0)).cumprod()


def prepare_universe(frames: dict, entry_signals=None, filter_cols=None,
                     window: int = 5, horizons=(5, 10),
                     train_frac: float = 0.7) -> tuple:
    """Precompute per-symbol numpy arrays once so both stages are fast.
    All rolling ops happen per symbol here — nothing leaks across symbols."""
    entry_signals = list(entry_signals or SIGNAL_COLUMNS_BULL)
    filter_cols = list(filter_cols if filter_cols is not None else FILTER_COLUMNS)
    bench = benchmark_index(frames)
    data = {}
    for sym, sig in frames.items():
        n = len(sig)
        if n < 60:
            continue
        entry_px = sig["Open"].shift(-1)
        data[sym] = {
            "fired": sig[entry_signals].astype(bool).to_numpy(),
            "recent": np.column_stack([
                recent(sig[c].astype(bool), window).to_numpy()
                for c in entry_signals]),
            "filters": (sig[filter_cols].astype(bool).to_numpy()
                        if filter_cols else np.zeros((n, 0), bool)),
            "fwd": np.column_stack([
                (sig["Close"].shift(-h) / entry_px - 1).to_numpy()
                for h in horizons]),
            "open": sig["Open"].to_numpy(float),
            "low": sig["Low"].to_numpy(float),
            "close": sig["Close"].to_numpy(float),
            "bench": bench.reindex(sig.index).ffill().to_numpy(float),
            "cut": int(n * train_frac),
        }
    meta = {"entry_signals": entry_signals, "filter_cols": filter_cols,
            "window": window, "horizons": tuple(horizons)}
    return data, meta


def _entry_array(d: dict, combo: tuple, min_count: int, fi: int) -> np.ndarray:
    """Entry rule for one symbol: >=1 combo signal fires now AND >=min_count
    distinct combo signals within the window AND (optional) filter is on."""
    ent = d["fired"][:, combo].any(axis=1) \
        & (d["recent"][:, combo].sum(axis=1) >= min_count)
    if fi >= 0:
        ent = ent & d["filters"][:, fi]
    return ent


def rank_entry_combos(data: dict, meta: dict, max_size: int = 2,
                      min_events: int = 80) -> pd.DataFrame:
    """Stage 1: score every combo x filter x strictness on TRAIN bars."""
    names = meta["entry_signals"]
    fnames = meta["filter_cols"]
    horizons = meta["horizons"]

    F = np.vstack([d["fired"] for d in data.values()])
    R = np.vstack([d["recent"] for d in data.values()])
    FL = np.vstack([d["filters"] for d in data.values()])
    FWD = np.vstack([d["fwd"] for d in data.values()])
    TRAIN = np.concatenate([np.arange(len(d["open"])) < d["cut"]
                            for d in data.values()])
    baseline = np.nanmean(FWD[TRAIN], axis=0)

    k = len(names)
    combos = [(i,) for i in range(k)]
    if max_size >= 2:
        combos += list(itertools.combinations(range(k), 2))

    rows = []
    for combo in combos:
        for fi in [-1] + list(range(len(fnames))):
            for mc in ([1] if len(combo) == 1 else [1, len(combo)]):
                ent = F[:, combo].any(axis=1) & (R[:, combo].sum(axis=1) >= mc)
                if fi >= 0:
                    ent = ent & FL[:, fi]
                m = ent & TRAIN
                n = int(m.sum())
                if n < min_events:
                    continue
                row = {"combo": combo, "min_count": mc, "filter_idx": fi,
                       "entry": " + ".join(names[i] for i in combo)
                                + (f" ({mc} of {len(combo)})" if len(combo) > 1 else ""),
                       "filter": fnames[fi] if fi >= 0 else "-",
                       "n_events": n}
                for hi, h in enumerate(horizons):
                    r = FWD[m, hi]
                    r = r[~np.isnan(r)]
                    row[f"edge_{h}"] = r.mean() - baseline[hi]
                    row[f"win_{h}"] = (r > 0).mean()
                rows.append(row)
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["score"] = res[[f"edge_{h}" for h in horizons]].mean(axis=1)
    return res.sort_values("score", ascending=False).reset_index(drop=True)


def _half_stats(rets: list, excess: list, prefix: str) -> dict:
    r = np.asarray(rets)
    if len(r) == 0:
        return {f"{prefix}_n": 0, f"{prefix}_win": np.nan,
                f"{prefix}_avg": np.nan, f"{prefix}_xs": np.nan,
                f"{prefix}_pf": np.nan}
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return {f"{prefix}_n": len(r),
            f"{prefix}_win": (r > 0).mean(),
            f"{prefix}_avg": r.mean(),
            # avg return per trade MINUS the market's return over the same
            # holding span — positive means the trade beat buy-and-hold
            f"{prefix}_xs": float(np.mean(excess)),
            f"{prefix}_pf": wins / losses if losses > 0 else np.inf}


def find_strategies(frames: dict, entry_signals=None, filter_cols=None,
                    window: int = 5, horizons=(5, 10), train_frac: float = 0.7,
                    max_size: int = 2, min_events: int = 80, top_k: int = 8,
                    exit_names=("none", "wt_cross_down"),
                    stops=(0.08,), holds=(10, 20),
                    progress_cb=None) -> tuple:
    """Full search. Returns (stage1 ranking DataFrame, stage2 results
    DataFrame sorted by train avg return; test columns are the honesty
    check). progress_cb(fraction, text) reports stage-2 progress."""
    data, meta = prepare_universe(frames, entry_signals, filter_cols,
                                  window, horizons, train_frac)
    ranked = rank_entry_combos(data, meta, max_size, min_events)
    if ranked.empty:
        return ranked, pd.DataFrame()

    # Pre-extract exit arrays once per symbol per exit rule.
    exit_arrays = {}
    for name in exit_names:
        cols = EXIT_OPTIONS[name]
        exit_arrays[name] = {
            sym: (frames[sym][cols].astype(bool).any(axis=1).to_numpy()
                  if cols else np.zeros(len(d["open"]), bool))
            for sym, d in data.items()}

    grid = list(itertools.product(range(min(top_k, len(ranked))),
                                  exit_names, stops, holds))
    results = []
    for gi, (ci, exit_name, stop, hold) in enumerate(grid):
        cand = ranked.iloc[ci]
        combo, mc, fi = cand["combo"], cand["min_count"], cand["filter_idx"]
        halves = {"train": ([], []), "test": ([], [])}
        for sym, d in data.items():
            ent = _entry_array(d, combo, mc, fi)
            ext = exit_arrays[exit_name][sym]
            cut = d["cut"]
            for label, sl in (("train", slice(0, cut)), ("test", slice(cut, None))):
                b = d["bench"][sl]
                for ei, xi_, epx, xpx, _r in _simulate(
                        d["open"][sl], d["low"][sl], d["close"][sl],
                        ent[sl], ext[sl], stop, hold):
                    ret = xpx / epx - 1
                    market = b[xi_] / b[ei] - 1 if b[ei] > 0 else 0.0
                    halves[label][0].append(ret)
                    halves[label][1].append(ret - market)
        row = {"entry": cand["entry"], "filter": cand["filter"],
               "exit": exit_name,
               "stop": f"{stop:.0%}" if stop else "-",
               "max_hold": hold,
               # raw config fields, for drill-down / bot export
               "combo": combo, "min_count": mc, "filter_idx": fi,
               "stop_value": stop, "window": window,
               "entry_cols": [meta["entry_signals"][i] for i in combo]}
        row.update(_half_stats(*halves["train"], "train"))
        row.update(_half_stats(*halves["test"], "test"))
        results.append(row)
        if progress_cb:
            progress_cb((gi + 1) / len(grid),
                        f"Simulating config {gi + 1}/{len(grid)}")

    # Rank by TRAIN excess return per trade: a config only rises when it
    # beat simply holding the market over the same spans.
    res = pd.DataFrame(results).sort_values("train_xs", ascending=False) \
        .reset_index(drop=True)
    return ranked, res


def evaluate_config(frames: dict, entry_cols: list, min_count: int,
                    window: int, filter_col: str | None, exit_cols: list,
                    stop: float | None, max_bars: int | None,
                    train_frac: float = 0.7,
                    weights: dict | None = None,
                    take_profit: float | None = None) -> pd.DataFrame:
    """Re-run one configuration and return ALL its trades with the symbol
    and train/test half attached — the per-symbol drill-down behind a
    finder result."""
    parts = []
    for sym, sig in frames.items():
        ent = combine_signals(sig, entry_cols, min_count, window, weights)
        if filter_col:
            ent = ent & sig[filter_col].astype(bool)
        ext = (sig[exit_cols].astype(bool).any(axis=1) if exit_cols
               else pd.Series(False, index=sig.index))
        res = backtest_long(sig, ent, ext, stop_loss=stop, max_bars=max_bars,
                            take_profit=take_profit)
        if res.n_trades:
            t = res.trades.copy()
            t["symbol"] = sym
            cut_ts = sig.index[min(int(len(sig) * train_frac), len(sig) - 1)]
            t["half"] = np.where(t["entry_idx"] < cut_ts, "train", "test")
            parts.append(t)
    if not parts:
        return pd.DataFrame()
    trades = pd.concat(parts, ignore_index=True).sort_values("entry_idx") \
        .reset_index(drop=True)
    bench = benchmark_index(frames)
    b_entry = bench.reindex(trades["entry_idx"], method="ffill").to_numpy()
    b_exit = bench.reindex(trades["exit_idx"], method="ffill").to_numpy()
    trades["market_ret"] = b_exit / b_entry - 1
    trades["excess"] = trades["ret"] - trades["market_ret"]
    return trades


def per_symbol_stats(trades: pd.DataFrame, min_trades: int = 3) -> pd.DataFrame:
    """Aggregate a config's trades by symbol: where does it actually work?"""
    if trades.empty:
        return trades
    g = trades.groupby("symbol")["ret"]
    stats = pd.DataFrame({
        "trades": g.count(),
        "win_rate": g.apply(lambda r: (r > 0).mean()),
        "avg_ret": g.mean(),
        "total_ret": g.sum(),
    })
    if "excess" in trades.columns:
        stats["avg_excess"] = trades.groupby("symbol")["excess"].mean()
    test = trades[trades["half"] == "test"].groupby("symbol")["ret"]
    stats["test_trades"] = test.count().reindex(stats.index).fillna(0).astype(int)
    stats["test_avg"] = test.mean().reindex(stats.index)
    stats = stats[stats["trades"] >= min_trades]
    return stats.sort_values("avg_ret", ascending=False)
