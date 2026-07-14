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
  structure.py   confirmed swing pivots + fib retracement levels (fib stop)
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
test_gridcache.py trust tests for the sweep's grid disk cache
test_structure.py trust tests for swing pivots / fib levels (no lookahead)
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
    --stop-mode=swing --trail-activate=0.04 --trail-dist=0.03

# fast sweep: build the grid once (cached to disk), sweep every exit combo
python sweep.py --months=12 --stop-mode=swing,atr \
    --trail-activate=0,0.04,0.06 --trail-dist=0.03,0.06,0.10
```

- The numba engine is the default. `--engine=loop` runs the original Python
  simulator instead — kept as a reference implementation to cross-check against.
- The sweep's slow step (building the 2D signal grid) is cached to
  `grid_cache/` keyed on the strategies, gates, window, params, and the DB's
  mtime, so the first run pays ~40s and every later sweep loads it in ~0.3s.
  A cache hit is a byte-identical grid, so results never drift.

- `--gate=COL@INTERVAL[,COL@INTERVAL...]` stacks trend filters across
  timeframes (e.g. a daily uptrend AND a 4h uptrend). Entries fire on the
  low `--interval` only when every gate was true on its last *closed*
  higher-timeframe bar — no lookahead.
- `--months=N` restricts trading to the last N months for fast iteration;
  promote survivors to a full-history run.
- Exits: fixed-% / ATR / swing-low / **fib** stops, a profit target, trailing
  and ratchet ("lock a gain once up X%") stops, a reversal exit signal, or a
  max-hold — whichever triggers first.
- `--stop-mode=fib` auto-fibs the up-leg (leg-start swing low -> swing high)
  and stops just below the `--fib-stop` retracement (e.g. 0.786 = under the
  78.6% level). The fib is a three-point construct (point1 leg-start low,
  point2 high, point3 pullback low) — the same anchors a charting tool uses.
- `--trail-mode=fib` trails the stop under each fib level once price clears
  it (the prior high, then the `--fib-ext` extension rungs 1.0/1.272/1.618…),
  so a winner is protected at each level but never taken off "just because" it
  reached one. `--trail-mode=structure` trails under each new higher swing
  low. Both fix winners getting shaken out early. (`--fib-target` — a hard
  take-profit at the prior high — exists but usually caps winners; prefer a
  trail.)
- `--fib-entry` filters or triggers entries off the pullback into the
  `--fib-zone-lo`..`--fib-zone-hi` band (default 0.5-0.786) of the current
  up-leg: `off` (signals only), `zone` (signal + price in the band),
  `bounce` (signal + a confirmed up-close off the band), or `bounce-trend`
  (the bounce **is** the entry, gated only by the higher-TF trend — buying the
  fib bounce in an uptrend, like the charts).
- Pivots come from a confirmation-lagged fractal (`--fib-left/--fib-right`,
  default 10 = only significant swings), so every fib level uses no future
  bars. Only the pivot window rebuilds the grid; the ratio/buffer/target/
  zone/trail-mode are applied at sim time and sweep cheaply.
- Everything is benchmarked against simply *holding the traded names* over
  the same window, so "active trading beat buy-and-hold" is an honest test.

### Confluence entry (weighted factors, nothing hard-coded)

`--conf-entry` switches entries from a single signal to a **weighted sum of
factors** — the philosophy being that no factor should be a hard gate you
discard when a little tuning might have made it work. Each factor is a per-bar
signed strength; the confluence score is `Σ (weight · factor)`, and you enter
when it clears `--conf-threshold`. Every factor exposes a `--w-<name>` knob
(0 = mute it), and the weights are applied at sim time so they **sweep
cheaply** — the whole point is to *discover* good weights by testing, not
decide them up front.

```
python sweep.py --months=12 --stop-mode=fib --conf-entry \
    --w-signal=0,1,2 --w-trend=0,1 --w-fib_prox=0,2 --conf-threshold=1.0,1.5
```

Current factors: `signal` (strategy-confluence count), `trend` (higher-TF
uptrend, a signed soft-veto), `fib_prox` (closeness to a fib retracement
level), `dip_bias` (where price sits in its recent range: +1 near the lows,
-1 extended near the highs — the "don't buy the top" / dangerous-range read,
adapted from the RCI trend indicator). Adding a factor is one array in
`build_grid` + its name in
`FACTOR_NAMES` — it's then instantly weightable and sweepable. `--conf-size`
scales position size by the score (stronger confluence = bigger position).

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
- `test_gridcache.py` — the sweep's grid disk cache: a save/load round-trip
  reproduces every array (dtypes, symbols, timestamps) exactly, and the cache
  key changes iff a build input changes, so a hit can't return a stale grid.
- `test_structure.py` — swing pivots / fib levels: known pivots map to known
  fib stops, and a prefix-recompute check proves the level at bar t uses only
  bars <= t (no lookahead — the same standard the fib stop is held to).

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
