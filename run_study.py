#!/usr/bin/env python3
"""
Run the WeissWave signal event study (and an example strategy backtest).

Data source: market.duckdb if present (see fetch_data.py), otherwise
CSVs in stock_data/.

Usage:
    python run_study.py                  # every symbol in the DB (daily)
    python run_study.py AAPL MSFT        # just these tickers
    python run_study.py --interval=1h    # hourly bars instead of daily
    python run_study.py --demo           # synthetic data smoke test
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

from weisswave.db import DB_PATH, connect, list_symbols, load_prices
from weisswave.signals import build_signals, SIGNAL_COLUMNS_BULL, SIGNAL_COLUMNS_BEAR
from weisswave.study import event_study, backtest_long

DATA_DIR = "stock_data"
HORIZONS = (1, 3, 5, 10, 20)


def load_ticker(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "index")), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        df = df.set_index(date_col).sort_index()
    return df


def synthetic_ohlcv(n=600, seed=7) -> pd.DataFrame:
    """Random-walk OHLCV for smoke-testing without market data."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.015, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1e6, 5e6, n).astype(float)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=idx)


def main():
    args = [a for a in sys.argv[1:]]
    demo = "--demo" in args
    interval = next((a.split("=", 1)[1] for a in args
                     if a.startswith("--interval=")), "1d")
    tickers = [a.upper() for a in args if not a.startswith("--")]

    frames = {}
    if demo:
        frames["DEMO"] = synthetic_ohlcv()
    elif os.path.exists(DB_PATH):
        con = connect(read_only=True)
        symbols = tickers or list_symbols(con, interval)
        for s in symbols:
            df = load_prices(con, s, interval)
            if len(df):
                frames[s] = df
        con.close()
        print(f"Loaded {len(frames)} symbols from {DB_PATH} ({interval} bars).")
        if not frames:
            print("Nothing found - run: python fetch_data.py")
            return
    else:
        paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
        if tickers:
            wanted = set(tickers)
            paths = [p for p in paths
                     if os.path.splitext(os.path.basename(p))[0].upper() in wanted]
        if not paths:
            print(f"No {DB_PATH} and no CSVs in {DATA_DIR}/ - run "
                  f"'python fetch_data.py' first, or try: python run_study.py --demo")
            return
        for p in paths:
            frames[os.path.splitext(os.path.basename(p))[0]] = load_ticker(p)

    # == build signals per ticker, pool events across the universe ========─
    all_signals = []
    for ticker, raw in frames.items():
        try:
            sig = build_signals(raw)
        except Exception as e:
            print(f"  {ticker}: skipped ({e})")
            continue
        sig["ticker"] = ticker
        all_signals.append(sig)
    if not all_signals:
        print("No usable data.")
        return
    pooled = pd.concat(all_signals)
    print(f"\nSignals computed for {len(all_signals)} tickers, "
          f"{len(pooled)} bars total.\n")

    # == event study: does each signal carry edge on its own? ==============
    for name, cols in (("BULL", SIGNAL_COLUMNS_BULL), ("BEAR", SIGNAL_COLUMNS_BEAR)):
        stats = event_study(pooled, cols, HORIZONS)
        fmt = stats.copy()
        for c in fmt.columns:
            if c != "n_events":
                fmt[c] = (fmt[c] * 100).round(2)
        print(f"== {name} signals - forward returns (%, next-open entry) ==")
        print(fmt.to_string())
        print()

    # == example strategy: wtv_buy entry, wtv_sell-or-stop exit ============
    print("== Example strategy: enter wtv_buy, exit wtv_sell / 8% stop / 20 bars ==")
    per_ticker = []
    for sig in all_signals:
        res = backtest_long(sig, sig["wtv_buy"], sig["wtv_sell"],
                            stop_loss=0.08, max_bars=20)
        if res.n_trades:
            res.trades["ticker"] = sig["ticker"].iloc[0]
            per_ticker.append(res.trades)
    if per_ticker:
        trades = pd.concat(per_ticker, ignore_index=True)
        wins = trades.loc[trades.ret > 0, "ret"].sum()
        losses = -trades.loc[trades.ret < 0, "ret"].sum()
        pf = wins / losses if losses > 0 else float("inf")
        print(f"trades={len(trades)}  win_rate={(trades.ret > 0).mean():.1%}  "
              f"avg={trades.ret.mean():.2%}  PF={pf:.2f}  "
              f"avg_bars={trades.bars.mean():.1f}")
    else:
        print("No trades triggered.")


if __name__ == "__main__":
    main()
