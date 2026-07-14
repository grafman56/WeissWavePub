"""Signal layer: turn one ticker's OHLCV into a flat table of boolean
signal columns plus the underlying indicator values.

This is the combining methodology's foundation — every idea from the
TradingView scripts becomes an independent, timestamp-aligned boolean
column. Strategies are then just rules over these columns, which makes
each component individually measurable (see study.py) before you commit
to any combination.
"""

import pandas as pd

from .core import crossover, crossunder, ema, normalize_ohlcv, rsi, sma
from .wavetrend import wavetrend

# The experience-driven "Combined v1" suite is proprietary and excluded
# from public distributions; everything here works without it.
try:
    from .combined import (BEAR_COLUMNS as _COMBINED_BEAR,
                           BULL_COLUMNS as _COMBINED_BULL, combined_signals)
except ImportError:
    combined_signals = None
    _COMBINED_BULL, _COMBINED_BEAR = [], []
from .weiswave import weis_wave, pressure_tiers
from .divergence import fractal_divergences
from .rci import rci


def build_signals(df: pd.DataFrame,
                  wt_channel: int = 9,
                  wt_average: int = 12,
                  wt_smooth: int = 3,
                  ob_level: float = 60.0,
                  os_level: float = -60.0,
                  pullback: int = 2,
                  very_heavy_mult: float = 10.0,
                  heavy_mult: float = 4.0,
                  confirm_window: int = 3) -> pd.DataFrame:
    """Compute WaveTrend + Weis Wave + divergences and derive signals.

    All boolean columns are True on the bar the condition is first
    knowable at that bar's close. Backtests should act on the NEXT bar's
    open (study.py does this).
    """
    df = normalize_ohlcv(df)
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    wt = wavetrend(df, wt_channel, wt_average, wt_smooth)
    ww = weis_wave(df, pullback)
    tiers = pressure_tiers(ww["volumeup"], ww["volumedn"],
                           very_heavy_mult, heavy_mult)
    divs = fractal_divergences(wt["wt1"], df["High"], df["Low"], prefix="wt")
    # RCI: Paul's ichimoku/regression-channel trend detection. Its multiplier
    # is timeframe-dependent, inferred from the index so callers need not pass
    # an interval (see rci.infer_bar_minutes).
    rc = rci(df)

    out = out.join([wt, ww, tiers, divs, rc])

    # ── WaveTrend zone signals ────────────────────────────────────────────
    out["wt_oversold"] = out["wt2"] <= os_level
    out["wt_overbought"] = out["wt2"] >= ob_level
    out["wt_exit_oversold"] = crossover(out["wt1"], pd.Series(os_level, index=out.index))
    out["wt_enter_overbought"] = crossunder(out["wt1"], pd.Series(ob_level, index=out.index))
    out["wt_cross_up_oversold"] = out["wt_cross_up"] & (out["wt2"] <= os_level)
    out["wt_cross_down_overbought"] = out["wt_cross_down"] & (out["wt2"] >= ob_level)

    # ── Weis Wave volume dominance crosses ───────────────────────────────
    out["volume_cross_up"] = crossover(out["volumeup"], out["volumedn"])
    out["volume_cross_down"] = crossunder(out["volumeup"], out["volumedn"])

    # ── Example composites (mirror the Combined v1 wtbuy/wtsell idea) ────
    bull_confirm = (out["volume_cross_up"] | out["wt_bull_div"]
                    | out["wt_hidden_bull_div"] | out["up_switch"])
    bear_confirm = (out["volume_cross_down"] | out["wt_bear_div"]
                    | out["wt_hidden_bear_div"] | out["down_switch"])
    out["wtv_buy"] = out["wt_exit_oversold"] & recent(bull_confirm, confirm_window)
    out["wtv_sell"] = out["wt_enter_overbought"] & recent(bear_confirm, confirm_window)

    # ── Standard textbook signals (public benchmark strategies) ──────────
    macd = ema(df["Close"], 12) - ema(df["Close"], 26)
    macd_sig = ema(macd, 9)
    out["macd_cross_up"] = crossover(macd, macd_sig)
    out["macd_cross_down"] = crossunder(macd, macd_sig)
    rsi14 = rsi(df["Close"], 14)
    out["rsi_oversold_cross"] = crossover(rsi14, pd.Series(30.0, index=df.index))
    out["rsi_overbought_cross"] = crossunder(rsi14, pd.Series(70.0, index=df.index))

    # ── Combined v1 Prod signal suite (proprietary; optional) ────────────
    if combined_signals is not None:
        out = out.join(combined_signals(df, out))

    # ── Regime filters (gates for entries, not entry triggers themselves) ─
    out["sma50"] = sma(df["Close"], 50)
    out["sma200"] = sma(df["Close"], 200)
    # EMA references. Paul's charts hang setups off the 200 EMA ("price doesn't
    # hammer below the 200ema, it holds") -- only SMAs existed, and they are not
    # the same line. Exposed as columns so factor definitions can reference them.
    out["ema50"] = ema(df["Close"], 50)
    out["ema200"] = ema(df["Close"], 200)
    out["minervini"] = out["sma50"] > out["sma200"]     # Stage-2 uptrend
    out["above_50ma"] = df["Close"] > out["sma50"]
    out["in_up_wave"] = out["wave"] == 1
    out["golden_cross"] = crossover(out["sma50"], out["sma200"])
    out["death_cross"] = crossunder(out["sma50"], out["sma200"])

    return out


def recent(cond: pd.Series, bars: int) -> pd.Series:
    """True if `cond` fired on this bar or any of the previous `bars` bars."""
    return cond.rolling(bars + 1, min_periods=1).max().astype(bool)


def combine_signals(sig: pd.DataFrame, columns: list, min_count: int = 1,
                    window: int = 3, weights: dict | None = None) -> pd.Series:
    """Confluence rule: True on bars where at least one selected signal fires
    AND the selected signals that fired within the last `window` bars sum to
    at least `min_count`. With no `weights` every signal counts 1 (min_count
    = distinct-signal count, the original behavior); `weights` maps column ->
    small integer so one signal can count double, or 0 to make it advisory.
    Keep weights coarse (0-3): fine-grained weights are a curve-fit magnet."""
    cols = [c for c in columns if c in sig.columns]
    if not cols:
        return pd.Series(False, index=sig.index)
    w = {c: int((weights or {}).get(c, 1)) for c in cols}
    fired_now = sig[[c for c in cols if w[c] > 0]].any(axis=1) \
        if any(w[c] > 0 for c in cols) else sig[cols].any(axis=1)
    score = sum(recent(sig[c], window).astype(int) * w[c] for c in cols)
    return fired_now & (score >= min_count)


SIGNAL_COLUMNS_BULL = [
    "wt_cross_up", "wt_cross_up_oversold", "wt_exit_oversold",
    "wt_bull_div", "wt_hidden_bull_div",
    "volume_cross_up", "up_switch", "up_continue",
    "very_heavy_buy", "heavy_buy",
    "wtv_buy",
    # standard textbook signals
    "macd_cross_up", "rsi_oversold_cross", "golden_cross",
] + _COMBINED_BULL

FILTER_COLUMNS = ["minervini", "above_50ma", "in_up_wave", "buy_dominant"]

SIGNAL_COLUMNS_BEAR = [
    "wt_cross_down", "wt_cross_down_overbought", "wt_enter_overbought",
    "wt_bear_div", "wt_hidden_bear_div",
    "volume_cross_down", "down_switch", "down_continue",
    "very_heavy_sell", "heavy_sell",
    "wtv_sell",
    # standard textbook signals
    "macd_cross_down", "rsi_overbought_cross", "death_cross",
] + _COMBINED_BEAR
