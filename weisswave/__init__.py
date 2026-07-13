"""
WeissWave â€” Python port of the "WaveTrend with Volume" (WTV) and
"Combined v1 Prod" TradingView studies, plus a signal/backtest layer.

Modules
-------
core        : basic building blocks (ema, sma, rsi, rising/falling, hlc3)
wavetrend   : LazyBear WaveTrend oscillator (wt1/wt2, crosses, OB/OS)
weiswave    : Weis Wave engine (wave, opp counter, volumeup/volumedn, states)
divergence  : 5-bar fractal divergence detection (regular/hidden, bull/bear)
signals     : assembles all boolean signal columns for one ticker
study       : per-signal event study + simple long-only strategy simulator
"""

from . import core, wavetrend, weiswave, divergence, signals, study

__version__ = "0.7.1"

__all__ = ["core", "wavetrend", "weiswave", "divergence", "signals", "study",
           "__version__"]
