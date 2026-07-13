"""Measurement layer: event studies and a simple long-only simulator.

Methodology
-----------
1. event_study(): for each signal column, measure forward returns over
   several horizons versus the baseline of all bars. This tells you which
   individual signals carry edge BEFORE you combine anything.
2. backtest_long(): given an entry rule and exit rule (boolean Series),
   simulate one-position-at-a-time long trades. Entries and exits execute
   at the NEXT bar's open — no acting on information from the same bar.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def forward_returns(df: pd.DataFrame, horizons=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """Return of buying next bar's open and holding h bars (to close)."""
    entry = df["Open"].shift(-1)
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        out[f"fwd_{h}"] = df["Close"].shift(-h) / entry - 1
    return out


def event_study(df: pd.DataFrame, signal_cols, horizons=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """Per-signal forward-return stats vs. the all-bars baseline."""
    fwd = forward_returns(df, horizons)
    rows = []
    baseline = {f"fwd_{h}": fwd[f"fwd_{h}"].mean() for h in horizons}
    for col in signal_cols:
        if col not in df.columns:
            continue
        mask = df[col].astype(bool)
        n = int(mask.sum())
        row = {"signal": col, "n_events": n}
        for h in horizons:
            r = fwd.loc[mask, f"fwd_{h}"].dropna()
            row[f"mean_{h}"] = r.mean() if len(r) else np.nan
            row[f"win_{h}"] = (r > 0).mean() if len(r) else np.nan
            row[f"edge_{h}"] = (r.mean() - baseline[f"fwd_{h}"]) if len(r) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("signal")


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    n_trades: int = 0
    win_rate: float = np.nan
    avg_return: float = np.nan
    median_return: float = np.nan
    profit_factor: float = np.nan
    avg_bars_held: float = np.nan
    total_return: float = np.nan  # compounded, one unit per trade

    def summary(self) -> str:
        return (f"trades={self.n_trades}  win={self.win_rate:.1%}  "
                f"avg={self.avg_return:.2%}  median={self.median_return:.2%}  "
                f"PF={self.profit_factor:.2f}  bars={self.avg_bars_held:.1f}  "
                f"total={self.total_return:.1%}")


def _simulate(o: np.ndarray, lo: np.ndarray, hi: np.ndarray, c: np.ndarray,
              ent: np.ndarray, ext: np.ndarray,
              stop_loss: float | None, max_bars: int | None,
              take_profit: float | None = None) -> list:
    """Core sequential long-only loop on raw arrays.
    Returns [(entry_i, exit_i, entry_px, exit_px, reason), ...].
    Intrabar exits check stop before target (conservative: if a bar's range
    spans both, assume the adverse fill)."""
    n = len(o)
    trades = []
    i = 0
    while i < n - 1:
        if not ent[i]:
            i += 1
            continue
        entry_i = i + 1                      # signal on close of i → buy open of i+1
        entry_px = o[entry_i]
        stop_px = entry_px * (1 - stop_loss) if stop_loss else -np.inf
        tp_px = entry_px * (1 + take_profit) if take_profit else np.inf
        exit_i, exit_px, reason = None, None, None
        j = entry_i
        while j < n:
            if lo[j] <= stop_px:
                exit_i, exit_px, reason = j, min(o[j], stop_px), "stop"
                break
            if hi[j] >= tp_px:               # target touched intrabar
                exit_i, exit_px, reason = j, max(o[j], tp_px), "target"
                break
            if ext[j] and j + 1 < n:         # exit signal on close → sell next open
                exit_i, exit_px, reason = j + 1, o[j + 1], "signal"
                break
            if max_bars and (j - entry_i) >= max_bars and j + 1 < n:
                exit_i, exit_px, reason = j + 1, o[j + 1], "time"
                break
            j += 1
        if exit_i is None:                   # still open at end of data
            exit_i, exit_px, reason = n - 1, c[n - 1], "eod"
        trades.append((entry_i, exit_i, entry_px, exit_px, reason))
        i = exit_i + 1                       # no overlapping positions
    return trades


def backtest_long(df: pd.DataFrame, entry: pd.Series, exit_: pd.Series,
                  stop_loss: float | None = None,
                  max_bars: int | None = None,
                  take_profit: float | None = None) -> BacktestResult:
    """Sequential long-only simulator, next-open execution.

    entry/exit_ are boolean Series aligned to df. A stop_loss of e.g. 0.05
    exits when the LOW trades 5% below entry (filled at the stop price, or
    the open if it gapped through). take_profit of e.g. 0.10 exits when the
    HIGH trades 10% above entry. max_bars force-exits a stale position.
    """
    raw = _simulate(df["Open"].to_numpy(float), df["Low"].to_numpy(float),
                    df["High"].to_numpy(float), df["Close"].to_numpy(float),
                    entry.fillna(False).to_numpy(bool),
                    exit_.fillna(False).to_numpy(bool),
                    stop_loss, max_bars, take_profit)
    trades = [{
        "entry_idx": df.index[ei], "exit_idx": df.index[xi],
        "entry_px": epx, "exit_px": xpx,
        "ret": xpx / epx - 1, "bars": xi - ei, "exit_reason": reason,
    } for ei, xi, epx, xpx, reason in raw]
    tdf = pd.DataFrame(trades)
    res = BacktestResult(trades=tdf)
    if len(tdf):
        wins = tdf.loc[tdf["ret"] > 0, "ret"].sum()
        losses = -tdf.loc[tdf["ret"] < 0, "ret"].sum()
        res.n_trades = len(tdf)
        res.win_rate = (tdf["ret"] > 0).mean()
        res.avg_return = tdf["ret"].mean()
        res.median_return = tdf["ret"].median()
        res.profit_factor = wins / losses if losses > 0 else np.inf
        res.avg_bars_held = tdf["bars"].mean()
        res.total_return = (1 + tdf["ret"]).prod() - 1
    return res
