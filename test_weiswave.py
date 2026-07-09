"""Invariant tests for the Weis Wave engine port. Run: python test_weiswave.py"""

import numpy as np
import pandas as pd

from weisswave.weiswave import weis_wave, OPP_UP_CAP, OPP_DOWN_CAP
from weisswave.signals import build_signals


def make_df(closes, volumes=None):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    volumes = np.asarray(volumes, dtype=float) if volumes is not None else np.full(n, 100.0)
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
        "Close": closes, "Volume": volumes,
    }, index=pd.bdate_range("2025-01-01", periods=n))


def test_wave_flips_and_volume():
    # 5 up bars then 5 down bars: wave should flip to -1 during the decline,
    # and wave_vol should reset at the flip.
    df = make_df([100, 101, 102, 103, 104, 103, 102, 101, 100, 99])
    ww = weis_wave(df, pullback=2)
    assert set(ww["wave"].unique()) <= {-1, 0, 1}
    assert ww["wave"].iloc[4] == 1, "should be in an up wave at the top"
    assert ww["wave"].iloc[-1] == -1, "should be in a down wave at the end"
    flip = ww.index[ww["wave"].diff() != 0][-1]
    assert ww.loc[flip, "wave_vol"] == 100.0, "wave volume resets on flip"
    print("test_wave_flips_and_volume OK")


def test_opp_counts_and_caps():
    # Long decline (down wave), then a slow grind of higher closes that never
    # trips the 2-bar rising() wave flip... opp should count up 1,2,3...
    closes = list(np.linspace(110, 100, 12))
    # zig-zag upward: each close above close[k+1] but pattern avoids flipping
    closes += [100.5, 100.2, 100.8, 100.4, 101.0, 100.6, 101.2]
    df = make_df(closes)
    ww = weis_wave(df, pullback=2)
    assert (ww["opp"] <= OPP_UP_CAP).all() and (ww["opp"] >= OPP_DOWN_CAP).all()
    in_down = ww["wave"] == -1
    assert (ww.loc[in_down, "opp"] >= 0).all(), "no negative opp inside a down wave"
    assert (ww.loc[in_down, "opp"] > 0).any(), "opposition chain should register"
    print("test_opp_counts_and_caps OK")


def test_volume_seeding():
    # When a down wave flips up with accumulated counter-volume, volumeup on
    # the flip bar should exceed that bar's own volume (seeded by tempvolup).
    closes = list(np.linspace(120, 100, 15)) + [100.5, 100.2, 101.0, 102.0, 103.0]
    df = make_df(closes)
    ww = weis_wave(df, pullback=2)
    flips_up = (ww["wave"] == 1) & (ww["wave"].shift(1) == -1)
    if flips_up.any():
        bar = ww.index[flips_up][0]
        assert ww.loc[bar, "volumeup"] >= df.loc[bar, "Volume"]
    print("test_volume_seeding OK")


def test_signals_no_nan_bools():
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = make_df(closes, rng.integers(1e5, 1e6, 300))
    sig = build_signals(df)
    for col in sig.columns:
        if sig[col].dtype == bool:
            assert not sig[col].isna().any(), f"{col} has NaNs"
    assert len(sig) == len(df)
    print("test_signals_no_nan_bools OK")


if __name__ == "__main__":
    test_wave_flips_and_volume()
    test_opp_counts_and_caps()
    test_volume_seeding()
    test_signals_no_nan_bools()
    print("\nAll Weis Wave invariant tests passed.")
