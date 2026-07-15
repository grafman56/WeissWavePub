"""Invariant tests for the Weis Wave engine port. Run: python test_weiswave.py"""

import numpy as np
import pandas as pd

from weisswave.core import falling, rising
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


def test_rising_is_monotonic_not_a_breakout():
    # THE regression test. BTC-USD 2025-03-23: closes ran 84,089 -> 83,841 ->
    # 86,082. The last bar is a strong up-bar that clears the prior high, so a
    # "greater than the max of the last N bars" BREAKOUT test calls it rising.
    # Pine's rising() is monotonic -- 83,841 < 84,089 breaks the chain -- so it
    # says NOT rising. The two definitions disagree on exactly this shape, and
    # the wrong one shipped: it flipped the wave and reset volumedn to 1.0x
    # when Paul's chart shows it accumulating at 6.1x. Pine wins.
    s = pd.Series([84500.0, 84089.0, 83841.0, 86082.0])
    assert not bool(rising(s, 2).iloc[-1]), \
        "dip-then-spike is NOT monotonically rising"
    # spell the discredited definition out, so this test fails loudly if anyone
    # ever "fixes" rising() back into a breakout test
    breakout = s > s.shift(1).rolling(2).max()
    assert bool(breakout.iloc[-1]), \
        "the old breakout definition passed this bar -- that was the bug"
    print("test_rising_is_monotonic_not_a_breakout OK")


def test_rising_accepts_a_clean_run():
    s = pd.Series([100.0, 101.0, 102.0, 103.0])
    r = rising(s, 2)
    assert bool(r.iloc[-1]) and bool(r.iloc[2])
    print("test_rising_accepts_a_clean_run OK")


def test_rising_length_counts_comparisons():
    # rising(s, N) is N comparisons spanning N+1 bars -- the same convention as
    # Pine, and what makes pullback=2 mean "close > close[1] > close[2]".
    s = pd.Series([100.0, 99.0, 100.5, 101.0, 102.0])
    assert bool(rising(s, 3).iloc[-1]), "three up-comparisons in a row"
    assert not bool(rising(s, 4).iloc[-1]), \
        "the 4th comparison reaches back to the down-bar and must fail"
    print("test_rising_length_counts_comparisons OK")


def test_rising_rejects_flat_and_warmup_is_false():
    assert not rising(pd.Series([100.0] * 5), 2).any(), "flat is not rising"
    r = rising(pd.Series([100.0, 101.0, 102.0]), 2)
    assert r.dtype == bool, "must be a real bool, never NaN leaking into a gate"
    assert not bool(r.iloc[0]) and not bool(r.iloc[1]), "warm-up is False"
    assert bool(r.iloc[2])
    print("test_rising_rejects_flat_and_warmup_is_false OK")


def test_falling_mirrors_rising():
    s = pd.Series([100.0, 99.0, 98.0, 97.0])
    assert bool(falling(s, 2).iloc[-1]) and not bool(rising(s, 2).iloc[-1])
    # the BTC bar's mirror image: a bounce then a plunge is not monotonic
    s2 = pd.Series([84500.0, 84089.0, 84500.0, 80000.0])
    assert not bool(falling(s2, 2).iloc[-1]), \
        "bounce-then-plunge is NOT monotonically falling"
    print("test_falling_mirrors_rising OK")


def test_dip_then_spike_holds_wave_and_accumulates_volumedn():
    # Why any of the above matters. rising() feeds isTrending, which decides
    # when the wave FLIPS, which decides whether volumeup/volumedn accumulate
    # or reset. This is the 1.0x-vs-6.1x gap Paul's chart caught, in miniature.
    closes = list(np.linspace(120, 100, 10)) + [99.0, 101.5]
    df = make_df(closes)
    ww = weis_wave(df, pullback=2)
    assert ww["wave"].iloc[-1] == -1, \
        "a dip-then-spike must not flip the down wave"
    assert ww["volumedn"].iloc[-1] > df["Volume"].iloc[-1], \
        "volumedn must still be accumulating, not reset to this bar's volume"
    print("test_dip_then_spike_holds_wave_and_accumulates_volumedn OK")


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
    test_rising_is_monotonic_not_a_breakout()
    test_rising_accepts_a_clean_run()
    test_rising_length_counts_comparisons()
    test_rising_rejects_flat_and_warmup_is_false()
    test_falling_mirrors_rising()
    test_dip_then_spike_holds_wave_and_accumulates_volumedn()
    test_wave_flips_and_volume()
    test_opp_counts_and_caps()
    test_volume_seeding()
    test_signals_no_nan_bools()
    print("\nAll Weis Wave invariant tests passed.")
