"""LazyBear WaveTrend oscillator — port of the WTV study's core.

Pine source:
    ap  = hlc3
    esa = ema(ap, n1)
    d   = ema(abs(ap - esa), n1)
    ci  = (ap - esa) / (0.015 * d)
    tci = ema(ci, n2)
    wt1 = tci
    wt2 = sma(wt1, smoothing)
"""

import pandas as pd

from .core import ema, sma, hlc3, crossover, crossunder


def wavetrend(df: pd.DataFrame,
              channel_length: int = 9,
              average_length: int = 12,
              smoothing: int = 3) -> pd.DataFrame:
    """Return a DataFrame with wt1, wt2 and cross columns, indexed like df."""
    ap = hlc3(df)
    esa = ema(ap, channel_length)
    d = ema((ap - esa).abs(), channel_length)
    ci = (ap - esa) / (0.015 * d)
    tci = ema(ci, average_length)

    out = pd.DataFrame(index=df.index)
    out["wt1"] = tci
    out["wt2"] = sma(tci, smoothing)
    out["wt_cross_up"] = crossover(out["wt1"], out["wt2"])
    out["wt_cross_down"] = crossunder(out["wt1"], out["wt2"])
    return out
