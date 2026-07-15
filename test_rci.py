import unittest
import numpy as np
import pandas as pd
from weisswave.rci import rci, infer_bar_minutes, tf_multiplier, TF_MULTIPLIER


def make_ohlcs(n):
    """Create synthetic OHLC data with n bars."""
    index = pd.date_range(start="2024-01-01", periods=n)
    
    # Random walk for open prices
    opens = np.random.randn(n).cumsum() + 100
    
    # Generate high, low, close from random volatility
    volatilities = np.abs(np.random.rand(n)) * 2 + 0.5
    
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    
    for i in range(n):
        change = np.random.randn() * volatilities[i]
        close_val = opens[i] + change
        high_val = max(opens[i], close_val) + np.abs(np.random.randn()) * volatilities[i] / 2
        low_val = min(opens[i], close_val) - np.abs(np.random.randn()) * volatilities[i] / 2
        
        highs[i] = high_val
        lows[i] = low_val
        closes[i] = close_val
    
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": np.random.rand(n)})
    df.index = index
    return df


class TestRCI(unittest.TestCase):
    
    def test_channel_ordering(self):
        """Test that RCI channel ordering is correct: l <= mid <= h."""
        n = 300
        df = make_ohlcs(n)
        
        out = rci(df)
        
        rci_l = out["rci_l"]
        rci_mid = out["rci_mid"]
        rci_h = out["rci_h"]
        
        self.assertTrue((rci_l <= rci_mid + 1e-9).all())
        self.assertTrue((rci_mid <= rci_h + 1e-9).all())
    
    def test_bull_bear_mutually_exclusive(self):
        """Test that bullish and bear signals are mutually exclusive."""
        n = 300
        df = make_ohlcs(n)
        
        out = rci(df)
        
        self.assertFalse((out["rci_bull"] & out["rci_bear"]).any())
    
    def test_no_cloud_break_during_warmup(self):
        """Test that no cloud breaks occur during the warmup period (first 76 bars)."""
        n = 200
        df = make_ohlcs(n)
        
        out = rci(df)
        
        self.assertEqual(out["rci_breakup"].iloc[:76].sum(), 0)
        self.assertEqual(out["rci_breakdn"].iloc[:76].sum(), 0)
    
    def test_rci_trend_in_range(self):
        """Test that RCI trend values are within [-1.0, 1.0]."""
        n = 300
        df = make_ohlcs(n)
        
        out = rci(df)
        
        self.assertTrue(out["rci_trend"].between(-1.0, 1.0).all())
    
    def test_infer_bar_minutes(self):
        """Test that bar minutes are inferred correctly from index."""
        # Test 15min frequency -> 15
        index_15 = pd.date_range(start="2024-01-01", periods=100, freq="15min")
        result_15 = infer_bar_minutes(index_15)
        
        self.assertEqual(result_15, 15)
        
        # Test daily frequency -> 1440
        index_daily = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        result_daily = infer_bar_minutes(index_daily)
        
        self.assertEqual(result_daily, 1440)
    
    def test_tf_multiplier_ladder(self):
        """Test that TF multiplier returns correct values for known timeframes."""
        self.assertEqual(tf_multiplier(1440), 0.10)
        self.assertEqual(tf_multiplier(240), 0.07)
        self.assertEqual(tf_multiplier(60), 0.02)
        self.assertEqual(tf_multiplier(99999), 0.01)
    
    def test_output_columns(self):
        """Test that all expected output columns are present."""
        n = 300
        df = make_ohlcs(n)
        
        out = rci(df)
        
        required_columns = [
            "rci_h", "rci_l", "rci_mid", "rci_heat", "rci_heated", 
            "rci_superheated", "rci_bull", "rci_bear", "rci_breakup", 
            "rci_breakdn", "rci_trend"
        ]
        
        for col in required_columns:
            self.assertIn(col, out.columns)


class TestEveryKnobIsReachable(unittest.TestCase):
    """Paul's chart header reads `RCI 10 10 0.01 35 25 0.975`. Every one of
    those is an input he adjusts while reading. They were module literals, and
    build_signals called `rci(df)` with NO arguments -- rci() had accepted most
    of them all along and nothing could reach them. Tunable where he reads,
    frozen where we test.

    These pin that each knob is READ. A parameter that does not move the number
    is this repo's signature bug: --symbols was dead for every symbol, --exit
    was dead once, --gate-mode was never read by two of three tools. Every one
    printed a perfectly reasonable header describing what it was not doing.
    """

    def frame(self, n=400, seed=3):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1.5, n))
        high = close + np.abs(rng.normal(0, 1, n))
        low = close - np.abs(rng.normal(0, 1, n))
        op = close + rng.normal(0, 0.5, n)
        return pd.DataFrame(
            {"Open": op, "High": high, "Low": low, "Close": close,
             "Volume": rng.uniform(1e5, 1e6, n)},
            index=pd.date_range("2024-01-01", periods=n, freq="D"))

    def setUp(self):
        self.df = self.frame()
        self.base = rci(self.df)

    def assertMoved(self, col, **kw):
        got = rci(self.df, **kw)
        same = got[col].fillna(0).equals(self.base[col].fillna(0))
        k = next(iter(kw))
        self.assertFalse(same, f"{k}={kw[k]} did not move {col}: the knob is "
                               f"accepted and not read")

    def test_trend_sensitivity_is_read(self):
        self.assertMoved("rci_bull", trend_sensitivity=0.90)

    def test_channel_lengths_are_read(self):
        self.assertMoved("rci_h", length=20)
        self.assertMoved("rci_h", length2=20)

    def test_multiplier_is_read_and_bypasses_the_tf_ladder(self):
        """rci.py:46 says the ladder above 30m is EXTRAPOLATED from Paul's curve
        and 'should be tested, not assumed'. `multiplier` is how it is tested."""
        self.assertMoved("rci_h", multiplier=0.05)

    def test_heat_levels_are_read(self):
        """Thresholds come from THIS frame's heat distribution, not from a
        number I liked. First version used superheat_level=80 and "failed":
        the synthetic's heat maxes at 18.9, so 35 and 80 both select zero bars
        and rci_superheated is all-False either way. The knob was fine; the test
        picked a value the data cannot reach. (Real TSLA 1d moves 642 -> 58.)

        This is also why heat is fed CONTINUOUS to the search via RCI_HEAT_NORM:
        25/35 are DAILY numbers and would never fire on a 15m frame."""
        heat = self.base["rci_heat"].dropna()
        lo, hi = heat.quantile(0.5), heat.quantile(0.9)
        self.assertGreater(hi, lo, "degenerate heat distribution in the fixture")
        self.assertMoved("rci_heated", heat_level=float(lo))
        self.assertMoved("rci_superheated", superheat_level=float(hi))

    def test_ichimoku_periods_are_read(self):
        """The last four literals. Pine's own defaults, but goal #3 is finding
        which screen works and a period nobody can vary is a decision nobody can
        revisit."""
        self.assertMoved("rci_bull", conversion_periods=5)
        self.assertMoved("rci_bull", base_periods=40)
        self.assertMoved("rci_breakup", lagging_span2_periods=26)
        self.assertMoved("rci_breakup", displacement=10)

    def test_build_signals_threads_them(self):
        """The actual bug: rci() was parameterised and its ONLY caller passed
        nothing. The plumbing existed one layer down, unconnected."""
        from weisswave.signals import build_signals
        a = build_signals(self.df)
        b = build_signals(self.df, rci_params={"trend_sensitivity": 0.90})
        self.assertNotEqual(int(a["rci_bull"].sum()), int(b["rci_bull"].sum()),
                            "build_signals ignores rci_params")

    def test_displacement_one_does_not_read_the_future(self):
        """displacement=1 is a legal input and shift(displacement-1) would be
        shift(0); anything lower must not become a NEGATIVE shift, which reads
        forward. A knob that can turn into lookahead is worse than a constant."""
        out = rci(self.df, displacement=1)
        self.assertEqual(len(out), len(self.df))
        trunc = rci(self.df.iloc[:-40], displacement=1)
        n = len(trunc) - 30
        self.assertTrue(
            np.allclose(out["rci_h"].to_numpy()[:n],
                        trunc["rci_h"].to_numpy()[:n], equal_nan=True),
            "displacement=1 changed past bars when future bars were removed")


if __name__ == "__main__":
    unittest.main()
