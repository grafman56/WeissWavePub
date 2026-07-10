"""Loss autopsy: why did a strategy's losers lose?

Instead of re-running the backtest once per candidate filter, this looks
up the market context at each trade's DECISION bar (the signal bar, one
bar before the fill) and evaluates every candidate condition across all
trades in one pass:

  * condition_lift() — for each boolean condition (regime filters,
    pressure states, WT zones): average return and win rate when the
    condition was TRUE at entry vs FALSE. A large positive lift on a
    condition the strategy didn't require means adding it as a filter
    would likely have culled losers.
  * numeric_splits() — the same idea for continuous context (WaveTrend
    level, buy/sell volume ratio, distance from the 50MA), split at the
    median.
  * symbol_drag() — symbols whose trades cost the strategy the most.
"""

import numpy as np
import pandas as pd

from .signals import FILTER_COLUMNS

# Boolean conditions worth testing at entry (only those present in the
# signal frame are used, so this works with or without the proprietary suite).
CANDIDATE_CONDITIONS = FILTER_COLUMNS + [
    "wt_oversold", "wt_overbought", "sell_dominant",
    "very_heavy_buy", "heavy_buy", "very_heavy_sell", "heavy_sell",
]

NUMERIC_CONTEXT = ["wt1_at_entry", "vol_ratio", "close_vs_sma50_pct"]


def trade_context(frames: dict, trades: pd.DataFrame) -> pd.DataFrame:
    """One row per trade: outcome + conditions/values at the decision bar."""
    rows = []
    for t in trades.itertuples():
        sig = frames.get(t.symbol)
        if sig is None:
            continue
        try:
            pos = sig.index.get_loc(t.entry_idx)
        except KeyError:
            continue
        dec = sig.iloc[pos - 1] if pos > 0 else sig.iloc[pos]
        row = {
            "symbol": t.symbol, "ret": t.ret, "win": t.ret > 0,
            "bars": t.bars, "exit_reason": t.exit_reason,
            "market_ret": getattr(t, "market_ret", np.nan),
            "excess": getattr(t, "excess", np.nan),
        }
        for c in CANDIDATE_CONDITIONS:
            if c in sig.columns:
                row[c] = bool(dec[c])
        row["wt1_at_entry"] = float(dec["wt1"]) if pd.notna(dec["wt1"]) else np.nan
        row["vol_ratio"] = (float(dec["volumeup"] / dec["volumedn"])
                            if dec["volumedn"] else np.nan)
        row["close_vs_sma50_pct"] = (float(dec["Close"] / dec["sma50"] - 1) * 100
                                     if pd.notna(dec["sma50"]) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def condition_lift(ctx: pd.DataFrame, min_each_side: int = 5) -> pd.DataFrame:
    """Win rate / avg return with each condition TRUE vs FALSE at entry."""
    rows = []
    for c in CANDIDATE_CONDITIONS:
        if c not in ctx.columns:
            continue
        t, f = ctx[ctx[c].astype(bool)], ctx[~ctx[c].astype(bool)]
        if len(t) < min_each_side or len(f) < min_each_side:
            continue
        rows.append({
            "condition": c,
            "n_true": len(t), "win_true": t["win"].mean(),
            "avg_true": t["ret"].mean(),
            "n_false": len(f), "win_false": f["win"].mean(),
            "avg_false": f["ret"].mean(),
            "avg_lift": t["ret"].mean() - f["ret"].mean(),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("avg_lift", ascending=False) if len(out) else out


def numeric_splits(ctx: pd.DataFrame, min_each_side: int = 5) -> pd.DataFrame:
    """Median-split lift for continuous entry context."""
    rows = []
    for c in NUMERIC_CONTEXT:
        if c not in ctx.columns:
            continue
        vals = ctx[c].dropna()
        if len(vals) < 2 * min_each_side:
            continue
        med = vals.median()
        hi = ctx[ctx[c] > med]
        lo = ctx[ctx[c] <= med]
        rows.append({
            "feature": c, "median": med,
            "n_above": len(hi), "win_above": hi["win"].mean(),
            "avg_above": hi["ret"].mean(),
            "n_below": len(lo), "win_below": lo["win"].mean(),
            "avg_below": lo["ret"].mean(),
            "avg_lift_above": hi["ret"].mean() - lo["ret"].mean(),
        })
    out = pd.DataFrame(rows)
    return (out.sort_values("avg_lift_above", ascending=False)
            if len(out) else out)


def symbol_drag(ctx: pd.DataFrame, worst: int = 10) -> pd.DataFrame:
    """Symbols costing the strategy the most, by total return contribution."""
    g = ctx.groupby("symbol")["ret"]
    out = pd.DataFrame({
        "trades": g.count(),
        "win_rate": g.apply(lambda r: (r > 0).mean()),
        "avg_ret": g.mean(),
        "total_ret": g.sum(),
    })
    return out.sort_values("total_ret").head(worst)
