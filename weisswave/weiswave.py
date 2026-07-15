"""Weis Wave volume engine — faithful port of the recursive Pine logic
shared by "WaveTrend with Volume" and "Combined v1 Prod".

The Pine `opp` value is a ~46-branch nested ternary; it collapses to a
simple rule once you see the pattern:

  * while in a DOWN wave (wave == -1), `opp` counts consecutive bars where
    close > close[opp_prev + 1]; the chain must start from opp_prev == 0
    and caps at +20 (any break, or exceeding the cap, resets to 0).
  * while in an UP wave (wave == +1), `opp` counts negatively where
    close < close[-opp_prev + 1], down to -23; unlike the down-wave side,
    a fresh chain can start (opp = -1) on any bar where close < close[1],
    regardless of opp_prev. This asymmetry exists in the original Pine and
    is preserved here.
  * any wave flip resets opp to 0.

`tempvolup`/`tempvoldown` accumulate volume while an opposition chain is
alive; when the wave flips, that accumulated counter-volume seeds the new
wave's cumulative volume (volumeup / volumedn).
"""

import numpy as np
import pandas as pd

from .core import rising, falling

OPP_UP_CAP = 20     # max opposition count inside a down wave
OPP_DOWN_CAP = -23  # max (most negative) opposition count inside an up wave


def weis_wave(df: pd.DataFrame, pullback: int = 2,
              pullback1: int = None) -> pd.DataFrame:
    """Compute wave direction, opposition counter, and cumulative wave
    volumes. Returns a DataFrame indexed like df with columns:

    wave        : +1 / -1 current wave direction (0 until established)
    wave_vol    : plain cumulative volume of the current wave (Pine `vol`)
    opp         : opposition counter (see module docstring)
    volumeup    : cumulative up-wave volume, seeded by prior counter-volume
    volumedn    : cumulative down-wave volume, seeded likewise
    volumeup1 / volumedn1 : the SECOND wave engine (below)
    up_continue / down_continue / up_switch / down_switch : wave state flags

    THE SECOND ENGINE. "Combined v1 Prod" runs the wave block TWICE:

        pullback  = input(2, "TDL for Wave")              -> wave  / volumeup
        pullback1 = input(2, "TDL for buy/sell pressure") -> wave1 / volumeup1

    and `crossup = crossover(volumeup1, volumedn1)` -- which feeds `golden` --
    uses the SECOND. Two details make it a hybrid rather than a clean copy, and
    both are reproduced here because the chart is the reference:

      * `tvolup`/`tvoldown` (the counter-volume that seeds a new wave) come
        from the FIRST engine's opp/wave -- the Pine never duplicates them.
      * `volumeup1` accumulates from `nz(volumeup[1])`, i.e. the FIRST
        engine's series, not its own previous value.

    CONSEQUENCE, VERIFIED: at pullback1 == pullback the two engines are
    byte-identical (same `trend`, same `isTrending` => wave1 == wave on every
    bar, and the accumulation branch is literally the same expression). They
    only diverge when the pullbacks differ -- which is exactly why pullback1
    needs to exist here: it is a TUNING dimension ("what if buy/sell pressure
    flipped more eagerly than the wave?"), not a bug fix.
    """
    if pullback1 is None:
        pullback1 = pullback
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    n = len(close)

    # mov / trend / wave (vectorizable parts kept in pandas for clarity)
    mov = np.sign(np.diff(close, prepend=close[:1])).astype(int)
    is_trending = (rising(df["Close"], pullback) | falling(df["Close"], pullback)) \
        .fillna(False).to_numpy()
    is_trending1 = (rising(df["Close"], pullback1)
                    | falling(df["Close"], pullback1)).fillna(False).to_numpy()

    trend = np.zeros(n, dtype=int)
    wave = np.zeros(n, dtype=int)
    wave_vol = np.zeros(n)
    opp = np.zeros(n, dtype=int)
    tempvolup = np.zeros(n)
    tempvoldown = np.zeros(n)
    volumeup = np.zeros(n)
    volumedn = np.zeros(n)
    wave1 = np.zeros(n, dtype=int)          # the buy/sell-pressure engine
    opp1 = np.zeros(n, dtype=int)
    volumeup1 = np.zeros(n)
    volumedn1 = np.zeros(n)

    def _opp_next(cur_wave, k, i):
        """The Pine's ~46-branch opp ladder, shared by both engines."""
        if cur_wave == -1 and 0 <= k < OPP_UP_CAP and i - (k + 1) >= 0 \
                and close[i] > close[i - (k + 1)]:
            return k + 1
        if cur_wave == 1:
            if OPP_DOWN_CAP < k <= -1 and i - (-k + 1) >= 0 \
                    and close[i] < close[i - (-k + 1)]:
                return k - 1
            if i > 0 and close[i] < close[i - 1]:
                return -1   # Pine allows restarting a down-chain any time
        return 0

    for i in range(n):
        prev_trend = trend[i - 1] if i > 0 else 0
        trend[i] = mov[i] if (mov[i] != 0 and mov[i] != (mov[i - 1] if i > 0 else 0)) else prev_trend

        prev_wave = wave[i - 1] if i > 0 else 0
        wave[i] = trend[i] if (trend[i] != prev_wave and is_trending[i]) else prev_wave

        same = wave[i] == prev_wave
        wave_vol[i] = (wave_vol[i - 1] + volume[i]) if (same and i > 0) else volume[i]

        # ── opposition counter ────────────────────────────────────────────
        k = opp[i - 1] if i > 0 else 0
        opp[i] = 0
        if same and i > 0:
            if wave[i] == -1 and 0 <= k < OPP_UP_CAP and i - (k + 1) >= 0 \
                    and close[i] > close[i - (k + 1)]:
                opp[i] = k + 1
            elif wave[i] == 1:
                if OPP_DOWN_CAP < k <= -1 and i - (-k + 1) >= 0 \
                        and close[i] < close[i - (-k + 1)]:
                    opp[i] = k - 1
                elif close[i] < close[i - 1]:
                    opp[i] = -1  # Pine allows restarting a down-chain any time

        # ── counter-volume accumulators ───────────────────────────────────
        if opp[i] == 1:
            tempvolup[i] = volume[i]
        elif opp[i] > 1:
            tempvolup[i] = volume[i] + tempvolup[i - 1]
        if opp[i] == -1:
            tempvoldown[i] = volume[i]
        elif opp[i] < -1:
            tempvoldown[i] = volume[i] + tempvoldown[i - 1]

        # ── cumulative wave volumes, seeded on wave flips ────────────────
        switched = not same
        tvolup = tempvolup[i - 1] if (switched and i > 0) else 0.0
        tvoldown = tempvoldown[i - 1] if (switched and i > 0) else 0.0

        if switched and tvolup > 0:
            volumeup[i] = tvolup + volume[i]
        elif same and wave[i] == 1 and i > 0:
            volumeup[i] = volumeup[i - 1] + volume[i]
        else:
            volumeup[i] = volume[i]

        if switched and tvoldown > 0:
            volumedn[i] = tvoldown + volume[i]
        elif same and wave[i] == -1 and i > 0:
            volumedn[i] = volumedn[i - 1] + volume[i]
        else:
            volumedn[i] = volume[i]

        # ── SECOND ENGINE (buy/sell pressure), pullback1 ──────────────────
        prev_wave1 = wave1[i - 1] if i > 0 else 0
        wave1[i] = trend[i] if (trend[i] != prev_wave1 and is_trending1[i]) \
            else prev_wave1
        same1 = wave1[i] == prev_wave1
        opp1[i] = _opp_next(wave1[i], opp1[i - 1] if i > 0 else 0, i) \
            if (same1 and i > 0) else 0
        # tvolup/tvoldown deliberately come from the FIRST engine (the Pine
        # never duplicates them), and the carry-over reads volumeup[i-1] --
        # also the first engine. Both are faithful, not oversights.
        t1u = tempvolup[i - 1] if (not same1 and i > 0) else 0.0
        t1d = tempvoldown[i - 1] if (not same1 and i > 0) else 0.0
        if not same1 and t1u > 0:
            volumeup1[i] = t1u + volume[i]
        elif same1 and wave1[i] == 1 and i > 0:
            volumeup1[i] = volumeup[i - 1] + volume[i]
        else:
            volumeup1[i] = volume[i]
        if not same1 and t1d > 0:
            volumedn1[i] = t1d + volume[i]
        elif same1 and wave1[i] == -1 and i > 0:
            volumedn1[i] = volumedn[i - 1] + volume[i]
        else:
            volumedn1[i] = volume[i]

    out = pd.DataFrame(index=df.index)
    out["wave"] = wave
    out["wave_vol"] = wave_vol
    out["opp"] = opp
    out["volumeup"] = volumeup
    out["volumedn"] = volumedn
    out["volumeup1"] = volumeup1
    out["volumedn1"] = volumedn1

    # Pine's cd/cu/su/sd simplify to these (the `(wave==wave[1] and wave==x)
    # or wave==x` construction is just `wave==x`):
    cd = (out["wave"] == 1) & (out["opp"] < 0)   # counter-pressure in up wave
    cu = (out["wave"] == -1) & (out["opp"] > 0)  # counter-pressure in down wave
    su = (out["wave"] == 1) & (out["opp"] == 0)
    sd = (out["wave"] == -1) & (out["opp"] == 0)
    s = out["wave"] != out["wave"].shift(1)
    c = ~s

    out["up_continue"] = c & cd.shift(1, fill_value=False) & su
    out["down_continue"] = c & cu.shift(1, fill_value=False) & sd
    out["up_switch"] = s & cu.shift(1, fill_value=False) & su
    out["down_switch"] = s & cd.shift(1, fill_value=False) & sd
    return out


def pressure_tiers(volumeup: pd.Series, volumedn: pd.Series,
                   very_heavy: float = 10.0, heavy: float = 4.0) -> pd.DataFrame:
    """Buy/sell pressure classification — the WTV plot-color logic.

    Tiers are mutually exclusive and mirror the Pine ternary order:
    very heavy buy > heavy buy > buy > very heavy sell > heavy sell > sell.
    """
    out = pd.DataFrame(index=volumeup.index)
    out["very_heavy_buy"] = volumeup > volumedn * very_heavy
    out["heavy_buy"] = ~out["very_heavy_buy"] & (volumeup > volumedn * heavy)
    out["buy_dominant"] = volumeup > volumedn
    out["very_heavy_sell"] = ~out["buy_dominant"] & (volumedn > volumeup * very_heavy)
    out["heavy_sell"] = ~out["buy_dominant"] & ~out["very_heavy_sell"] \
        & (volumedn > volumeup * heavy)
    out["sell_dominant"] = volumedn > volumeup
    return out
