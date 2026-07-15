#!/usr/bin/env python3
"""RCI -- "Regression Channel Trend Detection" (Paul's Pine v4 study).

A faithful port of trenddetectionichi.txt. Three things live in that study and
they answer different questions:

  1. ICHIMOKU CLOUD BREAKS -- `breakup` fires when price crosses UP through the
     top of a *bearish* (red) cloud on a green candle. Breaking up through a
     cloud that was supposed to resist is the strong bullish event.
  2. ICHI TREND -- ichibull/ichibear from the conversion/base lines, with a
     Trend Sensitivity deadzone so "barely crossed" is not a trend.
  3. SUPERHEAT -- how far price has stretched above the ADAPTIVE regression
     channel's lower band. This is the "dangerous range" idea: not a signal to
     sell, a warning not to buy the top.

WHY THIS MATTERS (Paul, 2026-07-15): he said his ichimoku trend detection
"should be in there somewhere" -- it was not ported at all. The `dip_bias`
factor in portfolio_multi carried the RCI's *name* while computing something
else entirely (price position in a rolling 20-bar range). This module is the
real thing; see rci_heat for the honest replacement.

TIMEFRAME-DEPENDENT BY DESIGN. The channel's adaptation rate scales with the
chart's timeframe (daily .1 -> 30m .02) -- Paul's own encoding of "higher
timeframes are stronger evidence". The multiplier is inferred from the bar
spacing, so a weekly frame adapts differently from a 15m one automatically.

ASCII only.
"""

import numpy as np
import pandas as pd
from numba import njit

# Ichimoku periods (Pine defaults, unchanged)
CONVERSION_PERIODS = 9
BASE_PERIODS = 26
LAGGING_SPAN2_PERIODS = 52
DISPLACEMENT = 26

TREND_SENSITIVITY = 0.975     # TS: deadzone so a marginal cross is not a trend
HEAT_LEVEL = 25.0             # % above the lower band -> "heated"
SUPERHEAT_LEVEL = 35.0        # -> "superheated"
CHANNEL_LENGTH = 10           # `length`  (inner EMAs)
CHANNEL_LENGTH2 = 10          # `length2` (outer EMA on h/l)

# Paul's TF ladder, keyed by bar minutes. His Pine covers daily..30m only; the
# fall-through is his manual `multiplier` input (0.01). Values NOT in his
# original are marked -- they are extrapolations of his curve and are the kind
# of thing that should be tested, not assumed.
TF_MULTIPLIER = {
    1: 0.01,        # extrapolated (not in Paul's ladder)
    5: 0.01,        # extrapolated (not in Paul's ladder)
    15: 0.01,       # extrapolated (falls through to his `multiplier` default)
    30: 0.02,       # Paul's
    60: 0.02,       # Paul's
    120: 0.04,      # Paul's
    180: 0.06,      # Paul's
    240: 0.07,      # Paul's
    1440: 0.10,     # Paul's (daily)
    10080: 0.10,    # WEEKLY -- extrapolated; his ladder stops at daily
}
DEFAULT_MULTIPLIER = 0.01     # Pine's `multiplier` input default


def infer_bar_minutes(index) -> int:
    """Bar spacing in minutes, from the index itself, so every caller of
    build_signals keeps working without passing an interval."""
    if len(index) < 3:
        return 1440
    d = pd.Series(index[1:]) - pd.Series(index[:-1])
    med = d.median()
    if pd.isna(med):
        return 1440
    mins = int(round(med.total_seconds() / 60.0))
    # snap to the nearest ladder rung (daily bars have weekend gaps; weekly
    # bars land near 10080 but drift with holidays)
    known = sorted(TF_MULTIPLIER)
    return min(known, key=lambda k: abs(k - mins)) if mins > 0 else 1440


def tf_multiplier(bar_minutes: int) -> float:
    return TF_MULTIPLIER.get(bar_minutes, DEFAULT_MULTIPLIER)


def _donchian(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    """Pine: donchian(len) => avg(lowest(len), highest(len)).

    min_periods=n (not 1) on purpose: before n bars exist Pine returns na, and
    fabricating a cloud from partial data would be exactly the kind of quiet
    lie this backtester exists to avoid."""
    return (low.rolling(n, min_periods=n).min()
            + high.rolling(n, min_periods=n).max()) / 2.0


@njit(cache=True)
def _reg_channel(high, low, close, ec, a1, a2, tf):
    """The adaptive regression channel (Pine's recursive h/l/mid).

    Pine:
      h := ema(<expr>, length2) where <expr> depends on h[1], and the inner
      ema(h[1]-close, length) keeps its own state. So two nested recursive
      EMAs per band. Unrolled here bar-by-bar:

        dh[t] = a1*(h[t-1] - close[t]) + (1-a1)*dh[t-1]
        xh[t] = high[t]                        if high[t] >= h[t-1]
              = h[t-1] + dh[t]*tf              if ema(close) >= mid[t-1]
              = h[t-1] - dh[t]*tf              otherwise
        h[t]  = a2*xh[t] + (1-a2)*h[t-1]

    The band snaps instantly to a new extreme (high >= h[1]) but decays only
    gradually toward price otherwise -- which is what makes it a channel rather
    than a rolling max. All inputs are bars <= t: no lookahead.
    """
    n = len(close)
    h = np.empty(n)
    l = np.empty(n)
    mid = np.empty(n)
    dh = 0.0
    dl = 0.0
    for t in range(n):
        if t == 0:                       # Pine: na(h[1]) -> seed with the bar
            h[0] = high[0]
            l[0] = low[0]
            mid[0] = (h[0] + l[0]) / 2.0
            dh = h[0] - close[0]
            dl = close[0] - l[0]
            continue
        hp = h[t - 1]
        lp = l[t - 1]
        mp = mid[t - 1]
        dh = a1 * (hp - close[t]) + (1.0 - a1) * dh
        dl = a1 * (close[t] - lp) + (1.0 - a1) * dl
        if high[t] >= hp:
            xh = high[t]
        elif ec[t] >= mp:
            xh = hp + dh * tf
        else:
            xh = hp - dh * tf
        if low[t] <= lp:
            xl = low[t]
        elif ec[t] <= mp:
            xl = lp - dl * tf
        else:
            xl = lp + dl * tf
        h[t] = a2 * xh + (1.0 - a2) * hp
        l[t] = a2 * xl + (1.0 - a2) * lp
        mid[t] = (h[t] + l[t]) / 2.0
    return h, l, mid


def rci(df: pd.DataFrame,
        bar_minutes: int = None,
        multiplier: float = None,
        trend_sensitivity: float = TREND_SENSITIVITY,
        heat_level: float = HEAT_LEVEL,
        superheat_level: float = SUPERHEAT_LEVEL,
        length: int = CHANNEL_LENGTH,
        length2: int = CHANNEL_LENGTH2,
        conversion_periods: int = CONVERSION_PERIODS,
        base_periods: int = BASE_PERIODS,
        lagging_span2_periods: int = LAGGING_SPAN2_PERIODS,
        displacement: int = DISPLACEMENT) -> pd.DataFrame:
    """Port of Paul's RCI study -> columns for the signal layer.

    EVERY INPUT ON PAUL'S CHART IS AN ARGUMENT HERE. His TradingView header
    reads `RCI 10 10 0.01 35 25 0.975` -- that is length, length2, multiplier,
    superheat_level, heat_level, trend_sensitivity. Those are things he ADJUSTS
    on a chart, and for a while they were module literals here: tunable where he
    reads them, frozen where they get tested. The module constants remain, but
    only as DEFAULTS.

    The four ichimoku periods (9/26/52/26) are Pine's own defaults and were the
    last hardcoded ones. They are parameters now for the same reason as the
    rest: goal #3 is finding which screen actually works, and a period nobody
    can vary is a decision nobody can revisit.

    Returns (all lookahead-free, knowable at each bar's close):
      rci_h/rci_l/rci_mid  : the adaptive regression channel
      rci_heat             : (ohlc4 - lower_band)/lower_band * 100
      rci_heated/_superheated : heat over the two thresholds
      rci_bull/rci_bear    : ichi conversion-vs-base trend, TS deadzone applied
      rci_breakup/_breakdn : cloud breaks (the strong events)
      rci_trend            : the study's own precedence collapsed to [-1, 1]
    """
    o, h_, l_, c = (df["Open"], df["High"], df["Low"], df["Close"])
    if bar_minutes is None:
        bar_minutes = infer_bar_minutes(df.index)
    tf = multiplier if multiplier is not None else tf_multiplier(bar_minutes)

    conversion = _donchian(h_, l_, conversion_periods)
    base = _donchian(h_, l_, base_periods)
    lead1 = (conversion + base) / 2.0
    lead2 = _donchian(h_, l_, lagging_span2_periods)

    # Pine reads leadLine[displacement-1]: the cloud AS DRAWN AT THIS BAR, i.e.
    # computed 25 bars ago. Past data -- shifting FORWARD would be lookahead.
    # max(0, ...) because displacement=1 is a legal input and shift(-0) is fine
    # while a negative shift would read the FUTURE. The guard is the difference
    # between a tunable knob and a lookahead bug.
    lead1_d = lead1.shift(max(0, displacement - 1))
    lead2_d = lead2.shift(max(0, displacement - 1))

    green = c > o
    red = c < o
    x_up = (c > lead2_d) & (c.shift(1) <= lead2_d.shift(1))     # crossover
    x_dn = (c < lead2_d) & (c.shift(1) >= lead2_d.shift(1))     # crossunder

    # lead2 > lead1 == span B above span A == a RED (bearish) cloud. Breaking UP
    # through a cloud that should have resisted is the bullish event.
    breakup = ((lead2_d > lead1_d) & green & x_up).fillna(False)
    breakdn = ((lead2_d < lead1_d) & red & x_dn).fillna(False)

    a1 = 2.0 / (length + 1.0)
    a2 = 2.0 / (length2 + 1.0)
    ec = c.ewm(span=length, adjust=False, min_periods=1).mean().to_numpy(float)
    hh, ll, mm = _reg_channel(h_.to_numpy(float), l_.to_numpy(float),
                              c.to_numpy(float), ec, a1, a2, float(tf))

    ohlc4 = (o + h_ + l_ + c) / 4.0
    with np.errstate(divide="ignore", invalid="ignore"):
        heat = (ohlc4.to_numpy(float) - ll) / np.where(ll == 0, np.nan, ll) * 100.0

    ts = trend_sensitivity
    ichibull = (conversion * ts > base).fillna(False)
    ichibear = (conversion < base * ts).fillna(False)

    out = pd.DataFrame(index=df.index)
    out["rci_h"] = hh
    out["rci_l"] = ll
    out["rci_mid"] = mm
    out["rci_heat"] = heat
    out["rci_heated"] = pd.Series(heat, index=df.index) > heat_level
    out["rci_superheated"] = pd.Series(heat, index=df.index) > superheat_level
    out["rci_bull"] = ichibull.to_numpy(bool)
    out["rci_bear"] = ichibear.to_numpy(bool)
    out["rci_breakup"] = breakup.to_numpy(bool)
    out["rci_breakdn"] = breakdn.to_numpy(bool)

    # The study's own precedence (from its colour ladder): a cloud break
    # outranks a plain ichi trend.
    trend = np.zeros(len(df))
    trend[out["rci_bull"].to_numpy()] = 0.5
    trend[out["rci_bear"].to_numpy()] = -0.5
    trend[out["rci_breakup"].to_numpy()] = 1.0
    trend[out["rci_breakdn"].to_numpy()] = -1.0
    out["rci_trend"] = trend
    return out
