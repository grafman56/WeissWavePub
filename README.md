# WeissWave

Python port of the "WaveTrend with Volume" (WTV) and "Combined v1 Prod"
TradingView studies, plus a measurement layer for turning their signals
into testable trading strategies.

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
  study.py       event_study() per-signal edge; backtest_long() simulator
  db.py          DuckDB storage: upsert_prices/load_prices/coverage_report
fetch_data.py    incremental S&P 500 fetcher -> market.duckdb
run_study.py     CLI: event study + example strategy (DB or CSV source)
app.py           Streamlit dashboard (fetch, charts, event study, backtests)
test_weiswave.py invariant tests for the wave engine
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
  confluence within a window, stop-loss and max-hold; trade stats,
  strategy-vs-market curve, exit-reason breakdown, trade list
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

## Porting notes

- The ~46-branch `opp` ternary in Pine reduces to: count consecutive
  closes against the wave direction (`close > close[opp+1]` chains),
  capped at +20 / −23, reset on any break or wave flip. The original's
  asymmetry (up-wave chains may restart mid-stream; down-wave chains
  must start from opp==0) is preserved deliberately.
- Pine v4 `rising(x, y)` = current value above **any** of the previous
  y values (not strictly monotonic); implemented accordingly.
- Requires only pandas + numpy (no pandas_ta; it is broken on numpy 2.x).
