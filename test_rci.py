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


if __name__ == "__main__":
    unittest.main()
