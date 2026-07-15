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
                 lbR: int = 10, direction: str = "up"):
    """The three anchors of a Fibonacci trend move, lookahead-free.

    A fib trend extension is what it sounds like -- Paul: "if you want to look
    for a 'downtrend' you start high, so the points are swing high, low, high.
    If you want to look for an uptrend you start low, so swing low, high, low."
    The DIRECTION is the whole choice, and it was hardcoded to "up".

      direction="up"   (leg-start LOW -> leg-top HIGH -> pullback LOW)
        point1 = the swing low that STARTED the last up-leg
        point2 = the last confirmed swing HIGH (leg top)
        point3 = the swing low AFTER that high (the pullback low)
        -> (point2 - point1) is POSITIVE, so extensions project UP from point3.

      direction="down" (leg-start HIGH -> leg-bottom LOW -> rally HIGH)
        point1 = the swing high that STARTED the last down-leg
        point2 = the last confirmed swing LOW (leg bottom)
        point3 = the swing high AFTER that low (the rally high)
        -> (point2 - point1) is NEGATIVE, so extensions project DOWN from point3.

    Verified against Paul's charts, same symbol, both directions, same formula:
      TSLA up  : 139.98 -> 489.50 -> 218.36  (p2-p1 = +349.52)
                 0=218.36  0.382=351.88  0.786=493.08     exact
      TSLA down: 272.40 -> 183.77 -> 490.04  (p2-p1 =  -88.63)
                 0=490.04  0.382=456.18  1.272=377.29  4=135.48   exact
      BTC  down: 107,244.8 -> 74,475.0 -> 125,887.2  -> every level to 0.1

    "down" is NOT only for shorting. Those levels answer "where does this
    decline stop", which is a LONG entry question -- BTC spent a year trading
    between its 1.272 and its 2.

    Retracements/zone/stop anchor point1 -> point2 (the leg being retraced).
    Extensions project (point2 - point1) from point3. Each series is NaN until
    the relevant pivots have confirmed, so a bar t depends only on bars <= t
    (test_structure.py proves it via prefix recompute)."""
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    ph = pine_pivot_high(high, lbL, lbR)
    pl = pine_pivot_low(low, lbL, lbR)
    ph_price = high.shift(lbR).where(ph).ffill()
    pl_price = low.shift(lbR).where(pl).ffill()
    ipos = pd.Series(np.arange(len(high), dtype=float), index=high.index)
    last_ph_i = ipos.where(ph).ffill()
    last_pl_i = ipos.where(pl).ffill()
    # The two directions are mirror images: swap which pivot is the leg END
    # (point2) and which supplies the bracketing anchors (point1/point3).
    if direction == "up":
        end, end_px, end_i = ph, ph_price, last_ph_i    # leg TOP
        opp_px, opp_i = pl_price, last_pl_i             # the lows around it
    else:
        end, end_px, end_i = pl, pl_price, last_pl_i    # leg BOTTOM
        opp_px, opp_i = ph_price, last_ph_i             # the highs around it
    point2 = end_px                            # last confirmed leg-end pivot
    point1 = opp_px.where(end).ffill()         # the opposite pivot as of it
    point3 = opp_px.where(opp_i > end_i)       # the reaction pivot after it
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
