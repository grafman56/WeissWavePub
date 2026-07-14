# WeissWave — Results & Methodology

What this framework is, how it stays honest, and what it has found so far.
The deliverable is a **trustworthy, fast, tunable backtesting platform** — the
specific strategies are secondary to being able to test them without fooling
ourselves.

## Methodology (the honesty guarantees)

1. **No lookahead.** Every signal/level uses only bars `<= t`; entries fire on
   `t-1` signals and fill at the next open. Pivots/fib levels are
   confirmation-lagged. Proven by trust suites (`test_engine`, `test_portsim`,
   `test_structure`) and by prefix-recompute checks (recompute on truncated
   data, require identical values).
2. **Beat buy-and-hold, or it doesn't count.** Every config is scored against
   equal-weight buy-and-hold of the names *it* traded over the same window.
   `exc% = active CAGR - hold CAGR`, and the board is **ranked by excess** — a
   strategy that loses to just holding the setups is not a win.
3. **Walk-forward, not one window.** `--wf-folds=N` splits the full history
   into N regimes and scores each config on all of them, ranked by *mean
   per-fold excess*. A single-window number is treated as a fluke until it
   survives across folds. `--oos-split` gives a simpler train/test check.
4. **Exits are stops / targets / trailing / reversal — never a time clock.**
   (A baked-in `hold=78` = 3-day time exit was found closing ~81% of trades and
   guillotining winners; it's now off by default and only testable on request.)

## The model — a tunable cascade

- **Higher-timeframe screen (weekly):** which stocks are set up. Weekly factors
  (`htf_trend` = EMA-stack alignment 20/50/100/200, `htf_ema_dist`,
  `htf_fib_prox`) form a weighted setup score; a stock is eligible only when it
  clears `--htf-threshold`. Reindexed onto the trading grid using only *closed*
  weeks (no lookahead).
- **Lower-timeframe entry (confluence):** when to enter. A **weighted sum of
  factors** (`signal, trend, fib_prox, dip_bias, vol_dom, div`) clears
  `--conf-threshold`. Nothing is a hard gate — every factor has a `--w-<name>`
  knob (0 = mute), so a promising-but-imperfect factor gets *tuned*, not
  discarded. Volume (`vol_dom`) and divergence (`div`) come from the
  WaveTrend-with-Volume port.
- **Exits:** fib / swing / ATR / pct stops, pct / structure / fib-ladder
  trailing, fib or % targets, reversal-signal exit.

All of it runs on a disk-cached 2D numba grid (build once ~40s, then ~30
ms/config), swept in parallel (`--jobs`), callable programmatically
(`run_sweep(spec)`), with every run persisted to `sweep_results/`.

## Key findings (all walk-forward, all vs buy-and-hold)

### Stocks (S&P 500, 15m, 2018–2026): active trading LOSES to holding

Across 6 walk-forward folds, **no exit config beats buy-and-hold.** The best
had −3% mean excess and beat hold in only 2/6 folds; the active strategy makes
money (~+22% CAGR) but *less than just holding the same names* (~+25%). Reason:
S&P names grind up with shallow pullbacks, so the edge is in **selection**, and
the best thing to do with a trending large-cap is **hold it**, not churn it.

### Crypto (13 majors, 15m, 2020–2026): active trading BEATS holding

Across 6 walk-forward folds (bull, the 2022 crash, recovery):

| stop | trail | mean excess vs hold | positive folds | active CAGR | hold CAGR |
|------|-------|--------------------:|:--------------:|------------:|----------:|
| fib  | pct   | **+64.7%**          | **5/6**        | 258.7%      | 194.0%    |
| swing| pct   | +48.4%              | 3/6            | 246.9%      | 198.6%    |
| fib  | fib   | −36.7%              | 1/6            | 158.4%      | 195.1%    |
| swing| structure | −270%           | 0/6            | −96.5%      | 173.9%    |

**The opposite of stocks.** Crypto *crashes*, so buy-and-hold eats −70%
drawdowns; active stops sidestep the crashes and re-enter, beating hold by a
wide, regime-robust margin. Winning recipe: **a stop for the downside + a
*loose* pct trail** to ride the explosive rallies. The fancy structure/fib
trails are catastrophic on crypto — they cut winners far too early and miss the
moon-shots.

### Setup parameters matter more than the exit

On a pure crypto-crash window, a loose confluence entry caught knives (147
trades, 98% invested, −64%). **Requiring a weekly-uptrend setup** (raise
`--htf-threshold`) cut that to 55 trades, **26% invested** (mostly cash through
the crash), and halved the loss to −37%. The engine faithfully traded *less*
when told to — crash re-entries are a tunable setup issue, not a backtester
artifact. The lesson: *don't buy unless the higher timeframe says the trend is
up.*

## The bottom line

The tool did the most valuable thing a backtester can: it said **where the edge
is and isn't** — hold S&P stocks, actively trade crypto — and backed it with
walk-forward evidence against buy-and-hold, on exactly the instruments theory
predicts. Next: a systematic (agent-driven) search of the weight/exit space,
optimizing walk-forward excess, on the crypto universe where a real edge exists.
