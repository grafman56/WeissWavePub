#!/usr/bin/env python3
"""Regression tests for the tunable search space and the gate-as-factor change.

Each test here pins a bug that actually shipped and silently corrupted results
rather than crashing -- which is the only kind worth a test in this repo.
"""

import unittest

import numpy as np

from portfolio_multi import FACTOR_NAMES, HTF_START
from search_space import (load_space, mutate_cfg, parse_set_args, sample_cfg,
                          space_sig, strip_docs)
from sweep import grid_sig_of

K = len(FACTOR_NAMES)


class TestGridSig(unittest.TestCase):
    """grid_sig names the DATA + SEMANTICS a score was computed under. If two
    incompatible runs share a signature, agent_search reuses one's score for
    the other and the store is silently poisoned."""

    def test_universe_separates(self):
        # THE BUG: universe was missing, and agent_search's default gate is the
        # same for both universes -> a stocks run collided with a crypto run.
        a = grid_sig_of("15m", "sma50_over_200@1d", "none", 0, "crypto", "factor")
        b = grid_sig_of("15m", "sma50_over_200@1d", "none", 0, "stocks", "factor")
        self.assertNotEqual(a, b, "crypto and stocks must not share a grid_sig")

    def test_gate_mode_separates(self):
        # a hard-gated score means nothing for a gate-as-factor config
        a = grid_sig_of("15m", "sma50_over_200@1d", "none", 0, "crypto", "hard")
        b = grid_sig_of("15m", "sma50_over_200@1d", "none", 0, "crypto", "factor")
        self.assertNotEqual(a, b, "gate_mode changes what a score MEANS")

    def test_interval_separates(self):
        self.assertNotEqual(
            grid_sig_of("5m", "sma50_over_200@1d", "none", 0, "crypto", "factor"),
            grid_sig_of("15m", "sma50_over_200@1d", "none", 0, "crypto", "factor"))


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.sp = load_space()
        self.r = np.random.default_rng(0)

    def test_relative_threshold_is_reachable(self):
        """THE BUG: weights and thresholds were drawn independently, so a config
        could mute its weights (capping the achievable score near 0) and then
        draw a threshold that score could never reach -- rejecting every bar
        forever and scoring a deceptively neutral 0.0. 27% of the store is this.
        Relative sampling makes the threshold reachable by construction."""
        for _ in range(500):
            c = sample_cfg(self.r, self.sp, K, HTF_START)
            emax = float(c["w"][:HTF_START].sum())    # factors are ~[-1, 1]
            hmax = float(c["w"][HTF_START:].sum())
            self.assertLessEqual(c["thr"], emax + 1e-6,
                                 "entry threshold above any achievable score")
            self.assertLessEqual(c["htf"], hmax + 1e-6,
                                 "screen threshold above any achievable score")

    def test_weight_bounds_come_from_the_space(self):
        """Widening a bound in the spec must actually widen what is sampled --
        otherwise 'tunable' is a lie and the search looks through a keyhole
        without saying so."""
        sp = load_space(overrides={"space.weights.hi": 12.0,
                                   "space.weights.mute_prob": 0.0})
        hi = max(float(sample_cfg(self.r, sp, K, HTF_START)["w"].max())
                 for _ in range(200))
        self.assertGreater(hi, 3.0, "raising weights.hi did not widen sampling")

    def test_mute_prob_zero_mutes_nothing(self):
        sp = load_space(overrides={"space.weights.mute_prob": 0.0,
                                   "space.weights.lo": 0.5})
        for _ in range(50):
            c = sample_cfg(self.r, sp, K, HTF_START)
            self.assertTrue((c["w"] > 0).all())

    def test_mutate_respects_space(self):
        sp = load_space(overrides={"mutate.td_lo": 0.05, "mutate.td_hi": 0.06,
                                   "mutate.td_prob": 1.0})
        c = sample_cfg(self.r, sp, K, HTF_START)
        for _ in range(50):
            c = mutate_cfg(c, self.r, sp, K)
            self.assertGreaterEqual(c["td"], 0.05 - 1e-9)
            self.assertLessEqual(c["td"], 0.06 + 1e-9)


class TestOverrides(unittest.TestCase):
    def test_parse_set_args_types(self):
        got = parse_set_args(["--set=grid.interval=5m",
                              "--set=space.weights.hi=6",
                              "--set=sim.htf_screen=0",
                              "--ignored=x"])
        self.assertEqual(got["grid.interval"], "5m")
        self.assertEqual(got["space.weights.hi"], 6)      # JSON-typed, not str
        self.assertEqual(got["sim.htf_screen"], 0)
        self.assertNotIn("--ignored", got)

    def test_nested_override_applies(self):
        sp = load_space(overrides={"grid.gate_mode": "hard",
                                   "fitness.min_trades": 25})
        self.assertEqual(sp["grid"]["gate_mode"], "hard")
        self.assertEqual(sp["fitness"]["min_trades"], 25)

    def test_space_sig_ignores_docs_but_tracks_values(self):
        """Every run records the space it used, so a survivor is traceable to
        the bar it had to clear. Doc text must not churn the signature."""
        a = space_sig(load_space())
        b = space_sig(load_space(overrides={"_README": ["different"]}))
        self.assertEqual(a, b, "_doc keys must not affect the signature")
        c = space_sig(load_space(overrides={"curation.min_pos_frac": 0.9}))
        self.assertNotEqual(a, c, "a real curation change must be visible")


class TestUnfitLogic(unittest.TestCase):
    """The no-trade rule, as a pure predicate (mirrors agent_search.score)."""

    @staticmethod
    def _fit(ntr, min_trades, exc, mn, pen=0.5):
        if 0 <= ntr < min_trades:
            return -np.inf
        return round(exc + pen * min(0.0, mn), 1)

    def test_no_trade_is_unfit_not_neutral(self):
        # THE BUG: a config that never trades scores excess 0.0 (flat equity vs
        # an empty traded-names baseline), which outranks every config that
        # actually took risk and lost. Doing nothing is not a tie.
        doing_nothing = self._fit(0, 1, 0.0, 0.0)
        traded_and_lost = self._fit(50, 1, -15.8, -80.6)
        self.assertEqual(doing_nothing, -np.inf)
        self.assertGreater(traded_and_lost, doing_nothing,
                           "a losing config must still outrank doing nothing")

    def test_unknown_trade_count_is_not_unfit(self):
        # legacy rows predate wf_trades; -1 means unknown, which must not be
        # mistaken for "never traded"
        self.assertNotEqual(self._fit(-1, 1, 5.0, -2.0), -np.inf)

    def test_min_trades_threshold_is_honoured(self):
        self.assertEqual(self._fit(9, 10, 99.0, 0.0), -np.inf)
        self.assertNotEqual(self._fit(10, 10, 99.0, 0.0), -np.inf)


class TestStripDocs(unittest.TestCase):
    """The space file carries prose inline with the values it configures.
    agent_search stripped the _doc keys ONE level deep, so
    signals.combined.tdi._oversold_doc rode into tdi_signals() as a keyword
    argument: TypeError on every symbol, swallowed by a bare except, and
    surfaced as "no usable 1d data in market.duckdb". A docstring in a JSON
    file accused the database, and the whole agent search path was dead."""

    def test_strips_at_every_depth(self):
        o = {"a": 1, "_doc": "x",
             "b": {"c": 2, "_doc": "y", "d": {"e": 3, "_doc": "z"}}}
        self.assertEqual(strip_docs(o), {"a": 1, "b": {"c": 2, "d": {"e": 3}}})

    def test_recurses_into_lists(self):
        o = {"a": [{"b": 1, "_doc": "x"}, {"c": 2}]}
        self.assertEqual(strip_docs(o), {"a": [{"b": 1}, {"c": 2}]})

    def test_leaves_real_values_alone(self):
        o = {"oversold": 40.0, "rsi_len": 21, "nested": {"k": [1, 2]}}
        self.assertEqual(strip_docs(o), o)

    def test_real_space_signals_are_accepted_by_build_signals(self):
        """The integration check that would have caught it. Whatever the
        shipped space says under "signals" must be callable as
        build_signals(**that) -- at EVERY nesting depth. No DB needed."""
        import pandas as pd
        from weisswave.signals import build_signals
        params = strip_docs(load_space("search_space.json").get("signals", {}))
        n = 400
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        df = pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99,
             "Close": close, "Volume": np.full(n, 100.0)},
            index=pd.bdate_range("2024-01-01", periods=n))
        build_signals(df, **params)          # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
