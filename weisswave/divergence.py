"""Fractal-based divergence detection — port of the WTV divergence block.

A 5-bar fractal centered 2 bars back:
    top:    osc[4] < osc[2] and osc[3] < osc[2] and osc[2] > osc[1] and osc[2] > osc[0]
    bottom: mirrored

IMPORTANT timing note for backtesting: a fractal whose peak is at bar i-2
is only *confirmed* at bar i (you need the two bars after the peak).
TradingView draws it two bars back with `offset=-2`, which looks earlier
on the chart than it was knowable. All boolean columns returned here are
True on the CONFIRMATION bar — the first bar you could actually act on.
"""

import numpy as np
import pandas as pd


def pine_pivot_low(s: pd.Series, lbL: int, lbR: int) -> pd.Series:
    """Pine pivotlow(s, lbL, lbR): the value lbR bars back is strictly lower
    than the lbL bars before it and the lbR bars after it. True on the
    CONFIRMATION bar (lbR bars after the pivot)."""
    center = s.shift(lbR)
    left = s.shift(lbR + 1).rolling(lbL, min_periods=lbL).min()
    right = s.rolling(lbR, min_periods=lbR).min()   # the lbR bars after center
    return ((center < left) & (center < right)).fillna(False)


def pine_pivot_high(s: pd.Series, lbL: int, lbR: int) -> pd.Series:
    center = s.shift(lbR)
    left = s.shift(lbR + 1).rolling(lbL, min_periods=lbL).max()
    right = s.rolling(lbR, min_periods=lbR).max()
    return ((center > left) & (center > right)).fillna(False)


def pine_pivot_low_nan(s: pd.Series, lbL: int, lbR: int) -> pd.Series:
    """pivotlow() for na-masked series (`cond ? value : na` in Pine).

    Pine's engine compares with Java NaN semantics: any comparison against
    na is false, so na neighbours can never veto a pivot. The pivot fires
    when the center bar is non-na and no non-na bar in the window beats it.
    """
    center = s.shift(lbR)
    left = s.shift(lbR + 1).rolling(lbL, min_periods=1).min()
    right = s.rolling(lbR, min_periods=1).min()
    left_ok = (center < left) | left.isna()
    right_ok = (center < right) | right.isna()
    return (center.notna() & left_ok & right_ok).fillna(False)


def pine_pivot_high_nan(s: pd.Series, lbL: int, lbR: int) -> pd.Series:
    """pivothigh() with Pine's na-tolerant semantics (see pine_pivot_low_nan)."""
    center = s.shift(lbR)
    left = s.shift(lbR + 1).rolling(lbL, min_periods=1).max()
    right = s.rolling(lbR, min_periods=1).max()
    left_ok = (center > left) | left.isna()
    right_ok = (center > right) | right.isna()
    return (center.notna() & left_ok & right_ok).fillna(False)


def pine_divergences(osc: pd.Series, high: pd.Series, low: pd.Series,
                     lbL: int, lbR: int, range_lower: int, range_upper: int,
                     prefix: str) -> pd.DataFrame:
    """Pivot-based divergences matching the 'Combined v1' Pine blocks:
    compare the current oscillator pivot (and price at that pivot) against
    the previous pivot, requiring the gap between pivot confirmations to be
    within [range_lower, range_upper] bars. True on the confirmation bar."""
    pl = pine_pivot_low(osc, lbL, lbR)
    ph = pine_pivot_high(osc, lbL, lbR)
    idx = pd.Series(np.arange(len(osc), dtype=float), index=osc.index)

    def prev_at(pivot: pd.Series, center: pd.Series) -> pd.Series:
        # value of `center` at the most recent pivot strictly before this bar
        return center.where(pivot).ffill().shift(1)

    def in_range(pivot: pd.Series) -> pd.Series:
        prev_pos = idx.where(pivot).ffill().shift(1)
        bars = idx - prev_pos
        return (bars >= range_lower) & (bars <= range_upper)

    osc_c, low_c, high_c = osc.shift(lbR), low.shift(lbR), high.shift(lbR)
    prev_osc_l, prev_low = prev_at(pl, osc_c), prev_at(pl, low_c)
    prev_osc_h, prev_high = prev_at(ph, osc_c), prev_at(ph, high_c)
    rng_l, rng_h = in_range(pl), in_range(ph)

    out = pd.DataFrame(index=osc.index)
    p = f"{prefix}_"
    out[p + "bull_div"] = pl & (low_c < prev_low) & (osc_c > prev_osc_l) & rng_l
    out[p + "hidden_bull_div"] = pl & (low_c > prev_low) & (osc_c < prev_osc_l) & rng_l
    out[p + "bear_div"] = ph & (high_c > prev_high) & (osc_c < prev_osc_h) & rng_h
    out[p + "hidden_bear_div"] = ph & (high_c < prev_high) & (osc_c > prev_osc_h) & rng_h
    return out.fillna(False)


def _top_fractal(osc: pd.Series) -> pd.Series:
    o = osc
    return (o.shift(4) < o.shift(2)) & (o.shift(3) < o.shift(2)) \
        & (o.shift(2) > o.shift(1)) & (o.shift(2) > o)


def _bottom_fractal(osc: pd.Series) -> pd.Series:
    o = osc
    return (o.shift(4) > o.shift(2)) & (o.shift(3) > o.shift(2)) \
        & (o.shift(2) < o.shift(1)) & (o.shift(2) < o)


def fractal_divergences(osc: pd.Series, high: pd.Series, low: pd.Series,
                        prefix: str = "") -> pd.DataFrame:
    """Regular/hidden bull/bear divergences between `osc` and price.

    Matches the Pine construction: the current fractal (osc and price at
    bar-2) is compared against the previous fractal, obtained via
    `valuewhen(cond, x, 0)[2]` — i.e. the most recent fractal as of two
    bars ago.
    """
    top = _top_fractal(osc)
    bot = _bottom_fractal(osc)

    # Value of osc/price at each fractal, forward-filled, then shifted 2:
    # at a confirmation bar this yields the PREVIOUS fractal's values.
    osc_at_top = osc.shift(2).where(top).ffill()
    price_at_top = high.shift(2).where(top).ffill()
    osc_at_bot = osc.shift(2).where(bot).ffill()
    price_at_bot = low.shift(2).where(bot).ffill()

    prev_osc_top = osc_at_top.shift(2)
    prev_price_top = price_at_top.shift(2)
    prev_osc_bot = osc_at_bot.shift(2)
    prev_price_bot = price_at_bot.shift(2)

    out = pd.DataFrame(index=osc.index)
    p = f"{prefix}_" if prefix else ""
    out[f"{p}bear_div"] = top & (high.shift(2) > prev_price_top) & (osc.shift(2) < prev_osc_top)
    out[f"{p}hidden_bear_div"] = top & (high.shift(2) < prev_price_top) & (osc.shift(2) > prev_osc_top)
    out[f"{p}bull_div"] = bot & (low.shift(2) < prev_price_bot) & (osc.shift(2) > prev_osc_bot)
    out[f"{p}hidden_bull_div"] = bot & (low.shift(2) > prev_price_bot) & (osc.shift(2) < prev_osc_bot)
    return out.fillna(False)
