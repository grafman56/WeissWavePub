#!/usr/bin/env python3
"""Trust tests for the data-defined factor compiler.

Two properties matter more than the arithmetic:
  1. NO LOOKAHEAD -- a factor at bar t must use only bars <= t. Same standard
     the rest of the engine is held to.
  2. USER JSON FAILS SAFELY -- a bad definition is a readable message, never a
     traceback, and never executable. A strategy file is untrusted input the
     moment anyone but the author can supply one.
"""

import unittest

import numpy as np
import pandas as pd

from weisswave.factors import (OPS, FactorError, compile_factor, custom_names,
                               validate_spec)


def frame(n=200, seed=0):
    r = np.random.default_rng(seed)
    c = 100 + np.cumsum(r.normal(0, 1, n))
    return pd.DataFrame(
        {"Open": c, "High": c + np.abs(r.normal(0, .5, n)),
         "Low": c - np.abs(r.normal(0, .5, n)), "Close": c,
         "Volume": r.uniform(1e3, 1e4, n),
         "volumeup": np.abs(np.cumsum(r.normal(0, 50, n))) + 100,
         "volumedn": np.abs(np.cumsum(r.normal(0, 50, n))) + 100,
         "ema200": pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy()},
        index=pd.date_range("2024-01-01", periods=n, freq="15min"))


DEFS = {
    "falling": {"op": "falling", "src": "volumedn", "bars": 5},
    "rising": {"op": "rising", "src": "volumeup", "bars": 5},
    "cross_up": {"op": "cross_up", "a": "volumeup", "b": "volumedn",
                 "within": 3},
    "cross_down": {"op": "cross_down", "a": "volumeup", "b": "volumedn"},
    "dist_above": {"op": "dist_above", "src": "Close", "ref": "ema200",
                   "scale": 0.02},
    "prox": {"op": "prox", "src": "Close", "ref": "ema200", "band": 0.05},
    "stall": {"op": "stall", "src": "Close", "bars": 5, "scale": 0.02},
    "hh_hl": {"op": "hh_hl", "bars": 10},
    "fails_to_break": {"op": "fails_to_break", "src": "Low", "ref": "ema200",
                       "bars": 3},
    "pivot_confirm": {"op": "pivot_confirm", "src": "volumedn", "side": "high",
                      "lbL": 6, "cap": 6},
    "column": {"op": "column", "src": "volumeup"},
}


class TestOpsCompile(unittest.TestCase):
    def test_every_op_has_a_definition_under_test(self):
        self.assertEqual(set(DEFS), set(OPS), "an op is untested")

    def test_all_ops_compile_in_range_and_finite(self):
        sig = frame()
        for name, d in DEFS.items():
            v = compile_factor(sig, name, d)
            self.assertEqual(len(v), len(sig), name)
            self.assertTrue(np.isfinite(v).all(), f"{name} produced non-finite")
            self.assertTrue((v >= -1.0).all() and (v <= 1.0).all(),
                            f"{name} escaped [-1, 1]")


class TestNoLookahead(unittest.TestCase):
    """THE property. Recompute on a prefix: the value at bar t must not change
    when future bars are removed."""

    def test_prefix_recompute_matches(self):
        sig = frame(300, seed=3)
        cut = 200
        for name, d in DEFS.items():
            full = compile_factor(sig, name, d)
            pre = compile_factor(sig.iloc[:cut].copy(), name, d)
            # compare away from the warmup edge; ema200 in the fixture is
            # precomputed over the full frame, so only ops reading it directly
            # are exempt from the head -- we compare the tail of the prefix.
            a, b = full[cut - 40:cut], pre[cut - 40:cut]
            self.assertTrue(np.allclose(a, b, atol=1e-9),
                            f"{name}: value at t changed when future bars were "
                            f"removed -> LOOKAHEAD")


class TestSign(unittest.TestCase):
    def test_sign_flips_the_factor(self):
        sig = frame()
        pos = compile_factor(sig, "x", {"op": "falling", "src": "volumedn"})
        neg = compile_factor(sig, "x", {"op": "falling", "src": "volumedn",
                                        "sign": -1})
        self.assertTrue(np.allclose(pos, -neg))

    def test_sign_default_is_positive(self):
        sig = frame()
        a = compile_factor(sig, "x", {"op": "rising", "src": "volumeup"})
        b = compile_factor(sig, "x", {"op": "rising", "src": "volumeup",
                                      "sign": 1})
        self.assertTrue(np.allclose(a, b))


class TestUserJsonFailsSafely(unittest.TestCase):
    """Bad JSON must produce a message a human can act on -- these are the
    errors an end user would hit."""

    def test_unknown_op_named_with_alternatives(self):
        errs = validate_spec({"a": {"op": "nope", "src": "Close"}})
        self.assertEqual(len(errs), 1)
        self.assertIn("nope", errs[0])
        self.assertIn("have:", errs[0])

    def test_missing_required_param(self):
        errs = validate_spec({"a": {"op": "dist_above", "src": "Close"}})
        self.assertTrue(any("ref" in e for e in errs))

    def test_bad_sign_rejected(self):
        self.assertTrue(validate_spec({"a": {"op": "falling", "src": "v",
                                             "sign": 7}}))

    def test_non_object_definition_rejected(self):
        self.assertTrue(validate_spec({"a": "hello"}))

    def test_unknown_column_is_a_factor_error(self):
        sig = frame()
        with self.assertRaises(FactorError) as cm:
            compile_factor(sig, "a", {"op": "prox", "src": "Close",
                                      "ref": "no_such_col"})
        self.assertIn("no_such_col", str(cm.exception))

    def test_valid_spec_has_no_errors(self):
        self.assertEqual(validate_spec(DEFS), [])

    def test_doc_keys_are_skipped(self):
        self.assertEqual(validate_spec({"_doc": "notes",
                                        "a": DEFS["falling"]}), [])
        self.assertEqual(custom_names({"_doc": "x", "a": {}, "b": {}}),
                         ["a", "b"])


class TestSemantics(unittest.TestCase):
    def test_falling_detects_a_decline(self):
        n = 50
        sig = frame(n)
        sig["volumedn"] = np.linspace(1000, 100, n)      # steadily falling
        v = compile_factor(sig, "x", {"op": "falling", "src": "volumedn",
                                      "bars": 5})
        self.assertGreater(v[-1], 0.0, "a falling series must score > 0")

    def test_falling_is_zero_when_rising(self):
        n = 50
        sig = frame(n)
        sig["volumedn"] = np.linspace(100, 1000, n)
        v = compile_factor(sig, "x", {"op": "falling", "src": "volumedn",
                                      "bars": 5})
        self.assertTrue((v[10:] == 0.0).all(), "a rising series must not score")

    def test_dist_above_sign_tracks_the_reference(self):
        n = 60
        sig = frame(n)
        sig["ema200"] = sig["Close"] - 5.0               # price above the ref
        v = compile_factor(sig, "x", {"op": "dist_above", "src": "Close",
                                      "ref": "ema200"})
        self.assertTrue((v > 0).all())
        sig["ema200"] = sig["Close"] + 5.0               # price below it
        v = compile_factor(sig, "x", {"op": "dist_above", "src": "Close",
                                      "ref": "ema200"})
        self.assertTrue((v < 0).all())


class TestPivotConfirm(unittest.TestCase):
    """lbR as a weight instead of a threshold. The op has to earn two things:
    the ramp (evidence accumulating) and the collapse (a candidate beaten on the
    right was never a pivot). Both matter -- a counter that only ever counts up
    would happily weight a peak that price already blew through."""

    def sig(self, vals):
        n = len(vals)
        c = np.full(n, 100.0)
        return pd.DataFrame(
            {"Open": c, "High": c, "Low": c, "Close": c,
             "Volume": np.full(n, 1.0), "volumedn": np.asarray(vals, float)},
            index=pd.date_range("2024-01-01", periods=n, freq="D"))

    def f(self, vals, **over):
        d = {"op": "pivot_confirm", "src": "volumedn", "side": "high",
             "lbL": 3, "cap": 6, **over}
        return compile_factor(self.sig(vals), "x", d)

    def test_evidence_ramps_as_bars_survive(self):
        #                 0  1  2   3   4  5
        v = self.f([1, 2, 3, 10, 4, 5])
        self.assertAlmostEqual(v[3], 0.0)          # the pivot bar itself: nothing yet
        self.assertAlmostEqual(v[4], 1 / 6)        # one bar of right-side evidence
        self.assertAlmostEqual(v[5], 2 / 6)        # two

    def test_a_candidate_beaten_on_the_right_collapses_to_zero(self):
        # 20 prints straight through the 10 peak: it was never a pivot, and the
        # factor has to say so on its own. This is BTC 2025-04-29 in miniature.
        v = self.f([1, 2, 3, 10, 4, 5, 20])
        self.assertAlmostEqual(v[5], 2 / 6)        # still live one bar earlier
        self.assertAlmostEqual(v[6], 0.0)          # beaten -> gone

    def test_never_exceeds_one_at_cap(self):
        v = self.f([1, 2, 3, 10, 4, 5, 6, 7, 8, 9], cap=6)
        self.assertAlmostEqual(v[9], 1.0)          # 6 bars survived, capped
        self.assertLessEqual(float(np.max(v)), 1.0)

    def test_a_candidate_that_never_beat_its_left_is_not_a_pivot(self):
        # 5 is a local high to its right but 10 sits in its left window
        v = self.f([1, 10, 3, 5, 4])
        self.assertAlmostEqual(v[4], 0.0)

    def test_side_low_mirrors(self):
        v = self.f([9, 8, 7, 1, 5, 6], side="low")
        self.assertAlmostEqual(v[4], 1 / 6)
        self.assertAlmostEqual(v[5], 2 / 6)

    def test_bad_side_is_a_readable_error_not_a_traceback(self):
        with self.assertRaises(FactorError):
            self.f([1, 2, 3], side="sideways")

    def test_agrees_with_the_verified_pine_pivot_at_the_frozen_lbR(self):
        """Ties the new weighted op to the binary one that is already verified
        against Paul's chart. Wherever pine_pivot_high(lbL, lbR) declares a
        pivot, the weighted op must read exactly lbR/cap -- same event, same
        bar, evidence reported instead of thresholded."""
        from weisswave.divergence import pine_pivot_high
        rng = np.random.default_rng(7)
        vals = np.abs(np.cumsum(rng.normal(0, 50, 400))) + 100
        sig = self.sig(vals)
        for lbR in (1, 2, 3):
            binary = pine_pivot_high(sig["volumedn"], 3, lbR)
            weighted = pd.Series(
                compile_factor(sig, "x", {"op": "pivot_confirm",
                                          "src": "volumedn", "side": "high",
                                          "lbL": 3, "cap": 6}),
                index=sig.index)
            hits = weighted[binary.astype(bool)]
            self.assertTrue(len(hits) > 0, f"lbR={lbR}: no pivots to compare")
            self.assertTrue(np.allclose(hits, lbR / 6, atol=1e-9),
                            f"lbR={lbR}: weighted op disagrees with the "
                            f"verified pine pivot")


if __name__ == "__main__":
    unittest.main(verbosity=2)
