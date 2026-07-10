"""Fine-resolution fill verification.

The coarse backtester fills orders at the next coarse bar's open and
models stops off bar lows. That hides what actually happened inside the
bar: gaps through stops, the order price actually hits, whether the stop
was touched before the exit signal. This module replays every trade
against REAL finer bars (15m/5m/1m) stored in the database — no
simulation or assumption about intrabar paths — and reports the modeled
vs verified difference per trade.

Verified fill rules (mirroring the coarse model's order types):
  entry  : market order queued at the coarse entry bar's start -> filled
           at the open of the first fine bar at/after that moment
  stop   : resting stop-market -> triggers at the FIRST fine bar whose low
           touches the stop; fills at min(fine open, stop) so gaps through
           the stop fill at the (worse) gap price
  signal / time exits : market order at the coarse exit bar's start ->
           open of the first fine bar at/after it
  eod    : last fine close in the window

Statuses: ok (fully verified), partial (entry covered, exit beyond fine
coverage), no_data (fine data doesn't cover the entry).
"""

import numpy as np
import pandas as pd

from .db import load_prices

BAR_SPAN = {
    "1d": pd.Timedelta(days=1),
    "1h": pd.Timedelta(hours=1),
    "15m": pd.Timedelta(minutes=15),
    "5m": pd.Timedelta(minutes=5),
    "1m": pd.Timedelta(minutes=1),
}


def _verify_one(fine: pd.DataFrame, t, coarse_span: pd.Timedelta,
                stop_loss: float | None) -> dict:
    entry_ts, exit_ts = t.entry_idx, t.exit_idx
    seg = fine[fine.index >= entry_ts]
    if seg.empty or seg.index[0] >= entry_ts + coarse_span:
        return {"v_status": "no_data"}
    if fine.index[-1] < exit_ts:
        return {"v_status": "partial"}

    v_entry_px = float(seg["Open"].iloc[0])
    stop_px = v_entry_px * (1 - stop_loss) if stop_loss else None
    end = exit_ts + coarse_span
    span = seg[seg.index < end]

    v_exit_px, v_exit_ts, note = None, None, ""
    if stop_px is not None:
        hit = span[span["Low"] <= stop_px]
        if len(hit):
            hit_ts = hit.index[0]
            if t.exit_reason == "stop":
                v_exit_px = min(float(hit["Open"].iloc[0]), stop_px)
                v_exit_ts, note = hit_ts, "stop"
            elif hit_ts < exit_ts:
                # fine bars show the stop was touched BEFORE the modeled
                # exit — the real position would have been stopped out
                v_exit_px = min(float(hit["Open"].iloc[0]), stop_px)
                v_exit_ts = hit_ts
                note = "stop touched before modeled exit"
    if v_exit_px is None:
        if t.exit_reason == "stop":
            # coarse modeled a stop the fine bars never touched (data
            # disagreement between resolutions — worth knowing about)
            after = span[span.index >= exit_ts]
            src = after if len(after) else span
            v_exit_px = float(src["Open"].iloc[0]) if len(after) \
                else float(span["Close"].iloc[-1])
            v_exit_ts = src.index[0] if len(after) else span.index[-1]
            note = "coarse stop never touched at fine resolution"
        elif t.exit_reason == "eod":
            v_exit_px = float(span["Close"].iloc[-1])
            v_exit_ts, note = span.index[-1], "eod"
        else:                                   # signal / time exits
            after = fine[fine.index >= exit_ts]
            if after.empty:
                return {"v_status": "partial"}
            v_exit_px = float(after["Open"].iloc[0])
            v_exit_ts, note = after.index[0], t.exit_reason

    v_ret = v_exit_px / v_entry_px - 1
    return {"v_status": "ok", "v_entry_px": v_entry_px,
            "v_exit_px": v_exit_px, "v_exit_ts": v_exit_ts,
            "v_ret": v_ret, "v_delta": v_ret - t.ret, "v_note": note}


def verify_trades(con, trades: pd.DataFrame, coarse_interval: str,
                  fine_interval: str = "15m",
                  stop_loss: float | None = None) -> pd.DataFrame:
    """Replay each trade against fine bars. Returns `trades` with v_*
    columns appended. Trades outside fine-data coverage get v_status
    'no_data'/'partial' and are excluded from bias stats by the caller."""
    coarse_span = BAR_SPAN[coarse_interval]
    pad = pd.Timedelta(days=2)
    out = []
    for sym, group in trades.groupby("symbol"):
        fine = load_prices(con, sym, fine_interval,
                           group["entry_idx"].min() - pad,
                           group["exit_idx"].max() + pad)
        for t in group.itertuples():
            row = {"_i": t.Index}
            if fine.empty:
                row["v_status"] = "no_data"
            else:
                row.update(_verify_one(fine, t, coarse_span, stop_loss))
            out.append(row)
    ver = pd.DataFrame(out).set_index("_i")
    result = trades.join(ver)
    result["v_status"] = result["v_status"].fillna("no_data")
    return result


def verification_summary(verified: pd.DataFrame) -> dict:
    """Aggregate honesty stats for a verified trade set."""
    ok = verified[verified["v_status"] == "ok"]
    n = len(verified)
    summary = {
        "n_trades": n,
        "n_verified": len(ok),
        "coverage": len(ok) / n if n else 0.0,
        "n_partial": int((verified["v_status"] == "partial").sum()),
        "n_no_data": int((verified["v_status"] == "no_data").sum()),
    }
    if len(ok):
        summary.update({
            "modeled_avg": float(ok["ret"].mean()),
            "verified_avg": float(ok["v_ret"].mean()),
            "bias": float(ok["v_delta"].mean()),
            "max_abs_delta": float(ok["v_delta"].abs().max()),
            "n_changed_exit": int((ok["v_note"].isin(
                ["stop touched before modeled exit",
                 "coarse stop never touched at fine resolution"])).sum()),
        })
    return summary
