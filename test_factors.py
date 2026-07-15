#!/usr/bin/env python3
"""Trust tests for the data-defined factor compiler.

Two properties matter more than the arithmetic:
  1. NO LOOKAHEAD -- a factor at bar t must use only bars <= t. Same standard
     the rest of the engine is held to.
  2. USER JSON FAILS SAFELY -- a bad definition is a readable message, never a
     traceback, and never executable. A strategy file is untrusted input the
     moment anyone but the author can supply one.
"""

import inspect
import unittest

import numpy as np
import pandas as pd

from weisswave.factors import (OPS, FactorError, _op_churn, compile_factor,
                               custom_names, validate_spec)


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
    "churn": {"op": "churn", "src": "Close", "bars": 20},
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


class TestChurn(unittest.TestCase):
    """`churn` = 1 - |net| / path. The quantity behind Paul's "builds before
    moves". Every case here is one the op has to get right to be worth having.
    """

    N = 20

    def f(self, vals):
        idx = pd.date_range("2024-01-01", periods=len(vals))
        return pd.DataFrame({"Close": np.asarray(vals, float)}, index=idx)

    def churn(self, vals, bars=None, **kw):
        d = {"op": "churn", "src": "Close", "bars": bars or self.N, **kw}
        return compile_factor(self.f(vals), "x", d)[-1]

    def stall(self, vals, bars=None):
        return compile_factor(self.f(vals), "x",
                              {"op": "stall", "src": "Close",
                               "bars": bars or self.N, "scale": 0.02})[-1]

    # the whole reason this op exists ------------------------------------------
    def test_separates_dead_from_churning_where_stall_cannot(self):
        """THE point. Both series have ZERO net displacement over `bars`, so
        `stall` scores them identically -- but one never moved and the other
        travelled 58 price units to get back. Opposite situations."""
        n = 61
        dead = np.full(n, 100.0)
        # periodic over exactly `N` bars => net displacement is 0 BY CONSTRUCTION
        churned = 100 + 5 * np.sin(2 * np.pi * np.arange(n) / self.N * 3)

        self.assertAlmostEqual(self.stall(dead), self.stall(churned), places=6,
                               msg="premise broken: stall must be blind here")
        self.assertLess(self.churn(dead), 0.01, "a series that never moved has "
                                                "no wasted motion -- not a build")
        self.assertGreater(self.churn(churned), 0.9, "travelled and returned = "
                                                     "maximum churn")

    def test_straight_line_is_zero(self):
        """Every step went somewhere: no waste, no build."""
        self.assertAlmostEqual(self.churn(np.linspace(100, 130, 61)), 0.0,
                               places=6)

    def test_bounded_and_signed_by_the_spec_not_the_op(self):
        rng = np.random.default_rng(0)
        v = 100 + np.cumsum(rng.normal(0, 1, 300))
        self.assertTrue(0.0 <= self.churn(v) <= 1.0)
        neg = compile_factor(self.f(v), "x", {"op": "churn", "src": "Close",
                                              "bars": self.N, "sign": -1})[-1]
        self.assertAlmostEqual(neg, -self.churn(v), places=9,
                               msg="sign belongs to compile_factor, not the op")

    # bars is the search axis, not a setting -----------------------------------
    def test_bars_actually_moves_the_number(self):
        """`bars` is THE parameter -- Paul's builds span 8 to 89 bars. A knob
        that does not move the number is the bug shape this repo is full of."""
        rng = np.random.default_rng(1)
        v = 100 + np.cumsum(rng.normal(0, 1, 400))
        got = {n: self.churn(v, bars=n) for n in (8, 20, 44, 89)}
        self.assertEqual(len(set(np.round(list(got.values()), 6))), len(got),
                         f"bars did not change the result: {got}")

    def test_no_frozen_lookback_constant(self):
        """RNG_LOOK=20 and DIV_LOOK=5 are module literals nothing can reach, and
        they keep biting. `churn` must never grow one: bars comes from the spec.
        """
        src = inspect.getsource(_op_churn)
        self.assertIn('_num(p, "bars"', src, "bars must come from the spec")
        self.assertNotIn("EFF_LOOK", src)
        self.assertNotIn("CHURN_LOOK", src)

    # the degeneracy guard -----------------------------------------------------
    def test_min_path_guard_is_data_and_live(self):
        """With no travel, net/path is 0/0 and the op would score a dead series
        1.0 -- maximum build -- which is `stall`'s bug inverted."""
        v = np.full(61, 100.0)
        v[-1] = 100.05                      # a whisper of movement
        loose = self.churn(v, min_path=0.0)
        tight = self.churn(v, min_path=0.01)
        self.assertGreater(loose, tight,
                           "min_path must gate the degenerate case, and be data")

    def test_real_build_outranks_a_real_trend(self):
        """Synthetics can be tuned to pass. This is Paul's own shape: a churning
        range vs a clean trend of the SAME length and SAME start price."""
        n = 61
        rng = np.random.default_rng(7)
        build = 100 + np.cumsum(rng.normal(0, 2, n))
        build = build - np.linspace(0, build[-1] - build[0], n)   # kill the drift
        trend = np.linspace(100, 140, n)
        self.assertGreater(self.churn(build), self.churn(trend) + 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
