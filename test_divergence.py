#!/usr/bin/env python3
"""Trust tests for the divergence port -- the signal Paul says is the main one.

    "The main thing seems to be some kind of divergence."  -- Paul, 2026-07-15

It had NO suite. divergence.py was reachable only through test_strategy.py,
which is a harness, not a gate. Nine of the ten bugs found on 2026-07-14 were a
tool reporting something it did not do; the one signal Paul actually trades was
not pinned against anything.

THE LOAD-BEARING TEST IS test_port_matches_a_literal_pine_transcription. Paul's
Pine is the specification, and reading it and reasoning about it is exactly the
method that fails here (his own rule: "Re-derive, do not read"). So the test
transcribes his source MECHANICALLY -- valuewhen, the [2] offsets, the 5-bar
fractal -- and diffs the outputs bar for bar over real data. Passing means the
port and the study agree; it does not mean either is a good idea.

    python test_divergence.py

Needs market.duckdb. ASCII output.
"""

import sys
import unittest

import numpy as np
import pandas as pd

from weisswave.db import connect, list_symbols, load_prices
from weisswave.divergence import (_bottom_fractal, _top_fractal,
                                  fractal_divergences)
from weisswave.signals import build_signals


def pine_literal(wt1, high, low):
    """Paul's Pine, transcribed with no cleverness. Source of truth:

        f_top_fractal(_src)=> _src[4] < _src[2] and _src[3] < _src[2]
                              and _src[2] > _src[1] and _src[2] > _src[0]
        fractal_top1 = f_fractalize(wt1) > 0 ? wt1[2] : na
        high_prev1  = valuewhen(fractal_top1, wt1[2], 0)[2]
        high_price1 = valuewhen(fractal_top1, high[2], 0)[2]
        regular_bearish_div1 = fractal_top1 and high[2] > high_price1
                               and wt1[2] < high_prev1
        hidden_bearish_div1  = fractal_top1 and high[2] < high_price1
                               and wt1[2] > high_prev1
        regular_bullish_div1 = fractal_bot1 and low[2] < low_price1
                               and wt1[2] > low_prev1
        hidden_bullish_div1  = fractal_bot1 and low[2] > low_price1
                               and wt1[2] < low_prev1

    The [2] on each valuewhen is not decoration: it skips a fractal that landed
    within 2 bars, so "previous fractal" cannot mean "the one I just printed".
    """
    s = wt1
    f_top = ((s.shift(4) < s.shift(2)) & (s.shift(3) < s.shift(2))
             & (s.shift(2) > s.shift(1)) & (s.shift(2) > s))
    f_bot = ((s.shift(4) > s.shift(2)) & (s.shift(3) > s.shift(2))
             & (s.shift(2) < s.shift(1)) & (s.shift(2) < s))

    def valuewhen(cond, src):           # occurrence 0 = the most recent true bar
        return src.where(cond).ffill()

    high_prev1 = valuewhen(f_top, s.shift(2)).shift(2)
    high_price1 = valuewhen(f_top, high.shift(2)).shift(2)
    low_prev1 = valuewhen(f_bot, s.shift(2)).shift(2)
    low_price1 = valuewhen(f_bot, low.shift(2)).shift(2)
    return {
        "bear_div": f_top & (high.shift(2) > high_price1)
        & (s.shift(2) < high_prev1),
        "hidden_bear_div": f_top & (high.shift(2) < high_price1)
        & (s.shift(2) > high_prev1),
        "bull_div": f_bot & (low.shift(2) < low_price1)
        & (s.shift(2) > low_prev1),
        "hidden_bull_div": f_bot & (low.shift(2) > low_price1)
        & (s.shift(2) < low_prev1),
    }


def _universe(n=15, interval="1d"):
    con = connect(read_only=True)
    out = []
    for s in list_symbols(con, interval)[:n]:
        df = load_prices(con, s, interval)
        if df is None or len(df) < 400:
            continue
        out.append((s, df, build_signals(df)))
    return out


class TestPortMatchesPine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u = _universe()
        if not cls.u:
            raise unittest.SkipTest("no 1d data in market.duckdb")

    def test_port_matches_a_literal_pine_transcription(self):
        """THE test. Not "does it look right" -- does it produce the same bars
        as Paul's source, on real data, for all four families."""
        checked = bad = 0
        for sym, df, sig in self.u:
            port = fractal_divergences(sig["wt1"], df["High"], df["Low"],
                                       prefix="wt")
            pine = pine_literal(sig["wt1"], df["High"], df["Low"])
            for k, v in pine.items():
                a = port[f"wt_{k}"].fillna(False).to_numpy(bool)
                b = v.fillna(False).to_numpy(bool)
                checked += 1
                bad += int((a != b).sum() > 0)
        self.assertEqual(bad, 0, f"{bad} of {checked} families disagree with "
                                 f"the Pine")
        self.assertGreater(checked, 20, "fixture too small to mean anything")

    def test_the_five_bar_fractal_is_centred_two_back(self):
        """A peak at i-2 needs bars i-1 and i to confirm it. That is a 2-bar
        confirmation lag baked into the definition -- and it is why a divergence
        cannot be known on the bar it happened."""
        v = pd.Series([0, 1, 5, 1, 0, 0, 0], dtype=float)
        top = _top_fractal(v)
        self.assertTrue(bool(top.iloc[4]), "peak at index 2 confirms at index 4")
        self.assertFalse(bool(top.iloc[2]), "must NOT fire on the peak bar")

    def test_bottom_fractal_mirrors(self):
        v = pd.Series([9, 8, 1, 8, 9, 9, 9], dtype=float)
        self.assertTrue(bool(_bottom_fractal(v).iloc[4]))


class TestDivergenceMagnitude(unittest.TestCase):
    """The Pine's own line is `high[2] > high_price1 and wt1[2] < high_prev1`:
    two `>`/`<` comparisons whose DISTANCES are right there and get collapsed to
    one bit. Measured over 25 symbols of 1d, the 2,072 bearish divergences run
    from "price higher by 0.22%, osc lower by 1.3" to "price higher by 23%, osc
    lower by 99.6" -- a ~20x range on both axes, all the same True."""

    @classmethod
    def setUpClass(cls):
        cls.u = _universe(8)
        if not cls.u:
            raise unittest.SkipTest("no 1d data in market.duckdb")

    def test_magnitudes_exist_only_at_their_own_divergence(self):
        """These are only defined AT a pivot pair. NaN everywhere else is the
        point: a rolling slope difference has no pivots in it and is true 57% of
        the time, which is noise, not a signal."""
        for sym, df, sig in self.u:
            for side in ("bear", "bull"):
                d = sig[f"wt_{side}_div"].fillna(False)
                for suf in ("px", "osc"):
                    m = sig[f"wt_{side}_div_{suf}"]
                    self.assertTrue(m[~d].isna().all(),
                                    f"{sym}: {side}_{suf} has a value off a "
                                    f"divergence bar")
                    if d.any():
                        self.assertFalse(m[d].isna().any(),
                                         f"{sym}: {side}_{suf} is NaN ON a "
                                         f"divergence bar")

    def test_magnitudes_are_positive_by_construction(self):
        """Signed so + always argues the divergence's own direction. A bearish
        divergence REQUIRES price higher and osc lower, so both gaps are > 0 --
        if one is ever negative the definition and the magnitude disagree."""
        for sym, df, sig in self.u:
            for col in ("wt_bear_div_px", "wt_bear_div_osc",
                        "wt_bull_div_px", "wt_bull_div_osc"):
                v = sig[col].dropna()
                if not len(v):
                    continue
                self.assertGreaterEqual(float(v.min()), 0.0,
                                        f"{sym}.{col} went negative")

    def test_the_bool_really_does_discard_a_wide_range(self):
        """If the magnitudes were all the same, emitting them would be pointless
        and this whole line of work would be a waste. They are not."""
        v = pd.concat([s["wt_bear_div_osc"].dropna() for _, _, s in self.u])
        self.assertGreater(len(v), 100, "too few divergences to characterise")
        lo, hi = v.quantile(0.10), v.quantile(0.90)
        self.assertGreater(hi / max(lo, 1e-9), 5.0,
                           f"osc gap spans only {lo:.1f}..{hi:.1f} -- if the "
                           f"bool discards nothing, do not build the factor")

    def test_no_lookahead(self):
        """Every divergence column must be knowable at its bar's close. The
        fractal reads _src[0] and _src[1] AFTER the peak, which is confirmation
        lag, not lookahead -- but only if the shifts point the right way."""
        for sym, df, sig in self.u[:3]:
            cut = len(df) - 60
            full = fractal_divergences(sig["wt1"], df["High"], df["Low"],
                                       prefix="wt")
            t = build_signals(df.iloc[:cut])
            trunc = fractal_divergences(t["wt1"], df["High"].iloc[:cut],
                                        df["Low"].iloc[:cut], prefix="wt")
            n = cut - 20
            for col in ("wt_bear_div", "wt_bull_div"):
                a = full[col].fillna(False).to_numpy(bool)[:n]
                b = trunc[col].fillna(False).to_numpy(bool)[:n]
                self.assertTrue((a == b).all(),
                                f"{sym}.{col} changed on PAST bars when future "
                                f"bars were removed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
