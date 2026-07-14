"""Structure primitives: confirmed swing pivots and Fibonacci retracement
levels, for structure-based stops / targets / trailing.

No lookahead. A swing pivot centered lbR bars back is only *confirmed* lbR
bars later (you need the bars after it to know it was a pivot). We reuse the
tested Pine pivot functions from `divergence`, which are True on the
confirmation bar, and forward-fill the confirmed price — so the level at bar
t depends only on bars <= t. test_structure.py proves this by recomputing on
truncated prefixes.
"""

import numpy as np
import pandas as pd

from .divergence import pine_pivot_high, pine_pivot_low


def trend_points(high: pd.Series, low: pd.Series, lbL: int = 10,
                 lbR: int = 10):
    """The three anchors of a Fibonacci trend move, lookahead-free:

      point1 = the swing low that STARTED the last up-leg (leg-start low)
      point2 = the last confirmed swing HIGH (leg top)
      point3 = the swing low AFTER that high (the pullback low), or NaN until
               a pullback low has confirmed.

    Retracements/zone/stop anchor point1 -> point2 (the up-leg being retraced).
    Extensions project (point2 - point1) from point3. Each series is NaN until
    the relevant pivots have confirmed, so a bar t depends only on bars <= t
    (test_structure.py proves it via prefix recompute)."""
    ph = pine_pivot_high(high, lbL, lbR)
    pl = pine_pivot_low(low, lbL, lbR)
    ph_price = high.shift(lbR).where(ph).ffill()
    pl_price = low.shift(lbR).where(pl).ffill()
    ipos = pd.Series(np.arange(len(high), dtype=float), index=high.index)
    last_ph_i = ipos.where(ph).ffill()
    last_pl_i = ipos.where(pl).ffill()
    point2 = ph_price                              # last confirmed swing high
    point1 = pl_price.where(ph).ffill()            # the pivot low as of that high
    point3 = pl_price.where(last_pl_i > last_ph_i)  # pullback low after the high
    return point1, point2, point3


def confirmed_pivots(high: pd.Series, low: pd.Series, lbL: int, lbR: int):
    """(ph_price, pl_price): price of the most recent CONFIRMED swing high /
    swing low as of each bar, forward-filled. The pivot value is the price
    lbR bars back, read on the confirmation bar (so it is knowable then)."""
    ph = pine_pivot_high(high, lbL, lbR)
    pl = pine_pivot_low(low, lbL, lbR)
    ph_price = high.shift(lbR).where(ph).ffill()
    pl_price = low.shift(lbR).where(pl).ffill()
    return ph_price, pl_price


def structure_levels(high: pd.Series, low: pd.Series, lbL: int = 5,
                     lbR: int = 5, stop_ratio: float = 0.786,
                     buf: float = 0.005):
    """Per-bar structure levels for a long pullback entry.

    The retraced up-leg is (most recent confirmed pivot low L) -> (most recent
    confirmed pivot high H). Returns (stop, target, pivot_low):
      stop   = buf below the stop_ratio retracement, H - stop_ratio*(H-L),
               e.g. just under the 78.6% (0.786) or 61.8% (0.618) level.
      target = the prior swing high H ("target = prior pivot high").
      pivot_low = the confirmed pivot-low ladder L, for structure trailing
               (ratchet the stop up under each new higher swing low).
    Where no valid up-leg exists yet (H <= L, or not enough history) stop and
    target are NaN and the caller falls back to its default stop / no target.
    """
    H, L = confirmed_pivots(high, low, lbL, lbR)
    span = H - L
    stop = ((H - stop_ratio * span) * (1.0 - buf)).where(span > 0)
    target = H.where(span > 0)
    return stop, target, L
