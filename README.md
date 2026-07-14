# WeissWave

A backtesting framework for multi-timeframe intraday trading strategies —
grown out of a hand-written VectorBT prototype into a self-contained,
dependency-light engine. It ports the "WaveTrend with Volume" (WTV) and
"Combined v1 Prod" TradingView studies into a tested signal library, then
layers a trustworthy, fast simulation engine for turning those signals into
validated strategies — from single-signal event studies up to a
multi-strategy portfolio bot.

The core idea is a top-down cascade: a higher timeframe (daily/4h trend, or
a market-wide regime filter) decides *which* stocks are worth trading and
*whether* to trade at all, and the actual entries fire on a lower timeframe
(15m/5m). "Trade with the trend" is the default rule; exits are driven by
structure-based stops, trailing stops, and reversal signals — never a fixed
clock.

Two properties are treated as non-negotiable and enforced by test suites:
**no lookahead** (nothing may act on information from the future) and a
**single, tested exit engine** (one numba-compiled simulation loop, so the
fill/stop/trailing logic can't drift between callers).

## Layout

```
weisswave/
  core.py        ema/sma/rma/rsi, Pine rising()/falling(), cross helpers
  wavetrend.py   LazyBear WaveTrend (wt1/wt2, crosses)
  weiswave.py    Weis Wave engine: wave, opp counter, volumeup/volumedn,
                 up/down switch & continue states, pressure tiers
  divergence.py  fractal divergences + Pine pivot divergences
  combined.py    PROPRIETARY (not in this repo): the experience-driven
                 signal suite. The package detects its absence and runs
                 with the open signal set only.
  signals.py     build_signals(): one ticker OHLCV -> boolean signal table
  optimize.py    two-stage automated strategy search + per-symbol drilldown
  study.py       event_study() per-signal edge; backtest_long() + _simulate()
  portsim.py     UNIFIED numba portfolio engine: one @njit loop over a 2D
                 (time x symbol) grid — single tested source of truth for
                 exit logic (stop/target/trailing/signal/time)
  fills.py       fine-resolution fill verification against real intraday bars
  provider.py    data-source abstraction (Yahoo + Alpaca deep intraday/crypto)
  db.py          DuckDB storage: upsert_prices/load_prices/coverage_report
fetch_data.py    incremental S&P 500 fetcher -> market.duckdb
alpaca_backfill.py  deep intraday backfill (15m/5m since 2018; derive 1h/4h)
run_study.py     CLI: event study + example strategy (DB or CSV source)
test_strategy.py one-command backtest harness (any combo/gate/exit/interval)
portfolio_multi.py  multi-strategy portfolio bot sim (stacked trend gates,
                 stop modes, trailing/ratchet, market-regime filter)
sweep.py         fast exit-parameter sweep: build grid once, run many configs
finder_gated.py  discover low-timeframe setups inside a higher-TF trend
scan_today.py    nightly setup scanner (post-close, from strategies.json)
app.py           Streamlit dashboard (fetch, charts, event study, backtests)
test_weiswave.py invariant tests for the wave engine
test_engine.py   trust tests for the backtest engine (fills/no-lookahead/...)
test_portsim.py  trust tests for the unified numba portfolio engine
```

## Dashboard

```
python -m streamlit run app.py
```

Opens at http://localhost:8501. Sidebar controls apply everywhere:
timeframe (1d/1h), symbol filter, WaveTrend lengths and OB/OS levels,
Weis Wave trend length and heavy/very-heavy volume multipliers,
composite confirmation window. Tabs:

- **Database** — coverage/staleness, one-click incremental fetch
- **Chart explorer** — candles with signal markers, WaveTrend and
  Weis Wave volume panels
- **Event study** — per-signal forward-return edge across the universe
- **Strategy finder** — automated search: stage 1 scores every entry
  combo (singles/pairs x regime filters x strictness) by train-set
  forward-return edge; stage 2 fully simulates the survivors across an
  exit/stop/hold grid on train and test halves separately, per selected
  timeframe. Ranked by train performance; the test columns are the
  honesty check. A drill-down expander shows per-symbol results for any
  config and exports it as a bot-ready JSON strategy file.
- **Strategy backtest** — pick entry/exit signal combos, require N-signal
  confluence within a window, regime filter, stop-loss and max-hold;
  trade stats, strategy-vs-market curve, exit-reason breakdown, trade
  list. Finder results load in with one click ("Load into Strategy
  backtest tab" in the finder drill-down).
- **Fine-resolution fill check** (inside the backtest tab) — replays
  every trade against REAL 15m/5m/1m bars stored in the DB (no simulated
  intrabar paths): entry/exit fills and stop touches re-derived from the
  fine bars and compared against the coarse model, with per-trade
  discrepancy tables, an overall bias number, and a trade inspector
  chart showing the fine bars around any trade with modeled vs verified
  fills marked. Yahoo intraday limits: 15m/5m ~55 days, 1m ~7 days —
  older trades report as unverifiable rather than guessed at.
- **Today's signals** — entry screener with named strategy presets
  (textbook: MACD cross, golden cross, RSI bounce; plus the WeissWave
  suite when present), manual rules, or loaded finder results

## Buy-and-hold honesty check

Every backtest and finder result is benchmarked against an equal-weight
buy-and-hold index of the same universe over the same period:

- per trade: `excess = trade return - market return over the same
  holding span`
- the finder ranks configurations by average TRAIN excess (`train_xs`),
  so a strategy only ranks highly when it beat simply holding the market
- the backtest tab shows the market's window return, the average excess
  per trade, the beat-market rate, and warns outright when buy-and-hold
  would have done better

Stage 1 of the finder is inherently market-relative too: signals are
scored by forward-return edge versus the all-bars baseline.

Signal computation over the full universe is cached per parameter set —
the first run takes a minute, tweaks after that are fast.

## CLI usage

```
python fetch_data.py                 # fetch/refresh full S&P 500 into market.duckdb
python fetch_data.py AAPL MSFT       # just these symbols
python fetch_data.py --report        # database coverage summary
python fetch_data.py --full          # force full-lookback refetch

python run_study.py                  # event study over every symbol in the DB
python run_study.py AAPL MSFT        # subset
python run_study.py --interval=1h    # hourly bars
python run_study.py --demo           # synthetic smoke test (no data needed)
```

## Command-line backtesting

```
# one strategy, any timeframe, trend-gated, cost-adjusted, train/test split
python test_strategy.py tdi_long,adp_bull_div --min-count=2 \
    --interval=15m --gate=minervini@1d,above_50ma@4h --cost-bps=10

# multi-strategy portfolio bot (stacked gates, structure stops, trailing)
python portfolio_multi.py --interval=15m --gate=minervini@1d,above_50ma@4h \
    --stop-mode=swing --trail-activate=0.04 --trail-dist=0.03 --engine=numba

# fast sweep: build the grid once, run every exit-parameter combination
python sweep.py --months=12 --stop-mode=swing,atr \
    --trail-activate=0,0.04,0.06 --trail-dist=0.03,0.06,0.10
```

- `--gate=COL@INTERVAL[,COL@INTERVAL...]` stacks trend filters across
  timeframes (e.g. a daily uptrend AND a 4h uptrend). Entries fire on the
  low `--interval` only when every gate was true on its last *closed*
  higher-timeframe bar — no lookahead.
- `--months=N` restricts trading to the last N months for fast iteration;
  promote survivors to a full-history run.
- Exits: fixed-% / ATR / swing-low stops, a profit target, trailing and
  ratchet ("lock a gain once up X%") stops, a reversal exit signal, or a
  max-hold — whichever triggers first.
- Everything is benchmarked against simply *holding the traded names* over
  the same window, so "active trading beat buy-and-hold" is an honest test.

## Trust: the engine is tested

A backtester that silently gets fills or timing wrong is worse than none —
it hands you false confidence. Two suites pin down the simulation core with
hand-built scenarios whose correct answers are known by construction:

- `test_engine.py` — `study._simulate` / `backtest_long`: next-open
  execution, stop/target fills (including gap-through), stop-before-target
  priority, signal/time exits, **no lookahead**, no overlapping positions.
- `test_portsim.py` — the unified numba engine: those properties plus
  trailing ratchets, position-slot caps, score-ranked selection, and
  accounting (equity = cash + positions; no money created or destroyed).

The numba engine (`portsim.py`) reproduces the original Python simulator
byte-for-byte on real data, then runs fast enough to sweep thousands of
configurations: build the signal grid once, fire each exit-parameter
combination through the compiled loop in milliseconds.

## Storage & dedup design

`market.duckdb` has one table, `prices`, with PRIMARY KEY
`(symbol, interval, ts)` and every write is an `INSERT OR REPLACE`:
overlapping fetches can never duplicate rows — they overwrite.

Time-sensitivity is handled deliberately:

- **Daily bars are keyed by trading date.** During market hours Yahoo
  returns the live session as a partial bar with a full timestamp; after
  the close the same day arrives as a plain date. Both normalize to the
  same key, so the fresh complete bar replaces the stale partial one.
- **Every incremental run re-fetches a small overlap window** (5 days
  daily, 2 days hourly) because the newest bars are the least
  trustworthy; the upsert corrects them in place.
- **Intraday timestamps are stored in UTC**; `fetched_at` (UTC) on every
  row records staleness.

Incremental runs only request bars since each symbol's last stored
timestamp, so a daily refresh of the full index takes seconds, and a
failed run never destroys existing data (nothing is ever cleared).

## Methodology

1. **Signal layer** — every idea from the Pine scripts becomes an
   independent boolean column, aligned to bars, True only when knowable
   at that bar's close (divergences flag on the *confirmation* bar, two
   bars after the pivot — unlike TradingView's offset=-2 plotting).
2. **Event study** — `event_study()` measures forward returns (1/3/5/10/20
   bars, next-open entry) for each signal against the all-bars baseline.
   Only signals with real standalone edge earn a place in a strategy.
3. **Combine** — strategies are boolean expressions over signal columns
   (e.g. `wt_exit_oversold & recent(volume_cross_up, 3)`), simulated with
   `backtest_long()` (next-open fills, stop-loss, max-hold).
4. **Validate** — split data in time (train/test), and beware pooled
   results driven by a handful of tickers or one market regime.
5. **Cascade** — the higher timeframe selects the tradable universe: a
   per-stock trend gate (which stocks are in an uptrend) stacked with a
   market-regime filter (whether to trade at all), so lower-timeframe
   entries only fire with the trend. The gate is fully swappable — any
   signal column on any timeframe.
6. **Portfolio** — `portfolio_multi.py` runs one or many strategies on a
   shared pool of capital with a position cap, ranking candidates when
   setups compete, each position managed by its own structure stop /
   target / trailing rule. This is the step that separates a per-trade
   edge from what an account actually earns.

## Porting notes

- The ~46-branch `opp` ternary in Pine reduces to: count consecutive
  closes against the wave direction (`close > close[opp+1]` chains),
  capped at +20 / −23, reset on any break or wave flip. The original's
  asymmetry (up-wave chains may restart mid-stream; down-wave chains
  must start from opp==0) is preserved deliberately.
- Pine v4 `rising(x, y)` = current value above **any** of the previous
  y values (not strictly monotonic); implemented accordingly.
- Requires only pandas + numpy (no pandas_ta; it is broken on numpy 2.x).
