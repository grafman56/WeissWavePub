# Changelog

All notable changes to WeissWave. Each release maps to git commits on
`main`; run `git log --oneline` for the full trail.

## Unreleased

- **Programmatic sweep API + result store** (toward agent-scale search): the
  sweep is now a callable `run_sweep(spec_dict) -> DataFrame` (spec keys are
  the CLI flags), sharing the exact tested parser with the CLI. Every run's
  full results are persisted to `sweep_results/*.parquet` tagged with a grid
  signature, scoring mode, spec and timestamp; `load_results()` schema-unions
  them so an agent can query "what's been tested, what won" instead of
  re-running. The CLI gains `--save-results`. Fitness is already walk-forward
  robust (`--wf-folds`), so a search optimizes robustness, not one lucky
  window. (Parallel execution is the next piece.)
- **Higher-TF setup screen — the cascade** (`--htf-screen --htf-threshold`):
  weekly setup factors (`htf_trend` = EMA-stack alignment across 20/50/100/200,
  `htf_ema_dist` = distance from the weekly EMA, `htf_fib_prox` = nearest weekly
  fib level) are resampled from the 25-yr daily and reindexed onto the trading
  grid using only *closed* weeks (no lookahead — verified by prefix recompute).
  They form a separate screen score; a stock is eligible only if its weighted
  weekly score clears the threshold, so the weekly picks *which stocks* are set
  up and the entry factors time *when* to trade them. A tunable soft gate (a
  threshold on a weighted score), both weights and threshold sweepable; the
  engine keeps entry factors [0:HTF_START] and screen factors [HTF_START:]
  separate. Grid schema -> v7. Trust: htf-screen scenarios in `test_portsim.py`
  (setup pass/fail gates entry; htf factors don't leak into the entry score).
- **Walk-forward scoring** (`sweep.py --wf-folds=6`): splits the whole history
  into N contiguous folds and scores every config on each, ranking by mean CAGR
  and showing `folds%` (CAGR per fold), `wf_min` (worst fold) and `wf_pos`
  (folds positive). An edge that's green across most folds survived regimes,
  not one lucky window — the robust upgrade over a single split. Reuses the
  sliceable `sim_metrics`; N sims/config, ~28 ms/sim on the cached grid.
- **Out-of-sample scoring** (`sweep.py --oos-split=0.7`): ranks configs on the
  earlier TRAIN slice and scores each on the held-out later TEST slice it never
  saw, showing `tr_CAGR`/`te_CAGR` side by side — a config whose test holds near
  its train is robust; one that craters was overfit. The single split can still
  be regime-biased (walk-forward across many splits is the next step), but it
  turns "best on this window" into "held up on data it didn't see." Two sims per
  config, still ~27 ms/sim.
- **New confluence factors**: `dip_bias` (RCI "dangerous range" — +near range
  lows, -extended near highs), `vol_dom` (WTV graded volume dominance — the
  volume confirmation), `div` (WTV bull/bear divergence confluence). All the
  volume/divergence signals were already computed by the WaveTrend-with-Volume
  port but unused by the live strategies; these surface them as weighted,
  sweepable factors. Grid schema -> v6.

## 0.9.0 — 2026-07-14

Fibonacci structure toolkit + weighted-confluence entry, all sim-time
tunable and swept on a disk-cached grid. numba is now the default engine.

- **Confluence entry — weighted factor stack** (`--conf-entry`): entries can
  fire from a weighted sum of factors (`score = Σ w·factor >= --conf-threshold`)
  instead of a single hard signal, so no factor is a gate you discard when
  tuning might have saved it. `build_grid` stacks each factor into one array;
  the engine combines them by a weight vector at sim time; a `--w-<name>` knob
  per factor is auto-exposed (0 = mute) and is a cheap sweep axis. Adding a
  factor = one array + its name in `FACTOR_NAMES`, then it's instantly
  weightable/combinable — the "combine any/all in any combination" goal.
  Starter factors: `signal`, `trend` (signed soft-veto), `fib_prox`.
  `--conf-size` scales position size by the score. Trust: weighted-sum entry
  scenarios in `test_portsim.py` (threshold, muting, single-factor carry);
  grid schema -> v4.
- **Three-point fib + fib-ladder trailing**: the fib is now a proper
  three-point construct — point1 (leg-start low), point2 (swing high), point3
  (pullback low), via `structure.trend_points` — so retracement/zone/stop
  anchor the true up-leg point1->point2 (previously the "last pivot low" could
  grab the pullback low and mis-anchor). `--trail-mode=fib` trails the stop
  under each fib level once price clears it (the prior high, then `--fib-ext`
  extension rungs 1.0/1.272/1.618/2.0), snapshotting the fib at entry — so a
  winner is protected at each level but never taken off just for tagging one.
  The extension math is verified to the dollar against charted examples.
  Trust: `trend_points` known-anchors + no-lookahead prefix test; fib-ladder
  trail scenario in `test_portsim.py`; grid schema bumped to v3.
- **Fib entry modes + deeper pivots**: `--fib-entry=off|zone|bounce|
  bounce-trend` filters or triggers entries off the pullback into the
  `--fib-zone-lo`..`--fib-zone-hi` band (default 0.5-0.786) of the current
  up-leg. `zone` = price in the band; `bounce` = a confirmed up-close off the
  band (the level held); `bounce-trend` = that bounce IS the entry, gated only
  by the higher-TF trend (the trend gate is now stored in the grid so the
  engine can fire on it). The pivot-window default rose 5 -> 10 so only
  significant swings anchor the fib (matches the TradingView Auto Fib depth).
  Entry mode / zone bounds / bounce lookback are sim-time params (cheap to
  sweep); only the pivot window rebuilds the grid. Trust: zone + bounce +
  bounce-trend scenarios in `test_portsim.py`; grid cache gained a schema
  version so old-schema files can't be mis-loaded.
- **Fib / structure stops** (`--stop-mode=fib`, `weisswave/structure.py`):
  auto-fib the last confirmed up-leg (swing low -> swing high) and stop below
  the `--fib-stop` retracement (0.786 / 0.618); `--fib-target` takes profit at
  the prior swing high; `--trail-mode=structure` trails under each new higher
  swing low instead of a fixed % (fixes winners exiting too early). Pivots are
  confirmation-lagged (reusing the tested Pine pivot functions), so the levels
  use no future bars. The grid stores the raw swing-high/low ladders; the fib
  ratio, buffer, target and trail-mode are applied at sim time, so they sweep
  cheaply — only the pivot window (`--fib-left/--fib-right`) rebuilds the grid.
  `--engine=loop` rejects fib/structure (numba-only) rather than silently
  downgrading. Trust suite: `test_structure.py` (incl. a no-lookahead proof)
  plus new FIB/structure-trail scenarios in `test_portsim.py`.
- **Grid disk cache** (`sweep.py` / `portfolio_multi.prepare_grid_cached`):
  the sweep's slow step — building the 2D signal grid — is now memoized to
  `grid_cache/` keyed on strategies, gates, market, interval, window, params,
  and the DB mtime. First run pays ~40s; every later sweep loads the grid in
  ~0.3s. The window is normalized to the day and folded into the key, so a
  cache hit is a byte-identical grid and results never drift. Atomic writes
  (`tmp` + `os.replace`) prevent half-written caches.
- **numba is now the default engine** for `portfolio_multi.py`. The original
  Python loop stays reachable as `--engine=loop`, a reference implementation
  to cross-check the numba engine against.
- **Trust suite** (`test_gridcache.py`): grid save/load round-trip fidelity
  (arrays, dtypes, symbols, timestamps) and cache-key determinism.

## 0.8.0 — 2026-07-14

- **Unified numba backtest engine** (`weisswave/portsim.py`): one
  numba-compiled loop over a 2D (time x symbol) grid — a single tested
  source of truth for exit logic (stop / target / trailing / ratchet /
  reversal-signal / time), replacing duplicate hand-rolled loops.
  Reproduces the original Python simulator byte-for-byte on real data.
- **Trust suites** (`test_engine.py`, `test_portsim.py`): verify fills,
  no-lookahead, exit priority, trailing, slot caps, score ranking, and
  accounting on hand-built scenarios with known answers.
- **Sweep-mode** (`sweep.py`): build the signal grid once, then run every
  exit-parameter combination through the compiled engine at ~60 ms/config.
- **Multi-strategy portfolio bot** (`portfolio_multi.py`): stacked
  cross-timeframe trend gates, market-regime filter, ATR/swing-low stops,
  trailing/ratchet exits, position caps, benchmarked against holding the
  traded names.
- **Deep intraday data** via Alpaca (`alpaca_backfill.py`): 15m/5m back to
  2018 with real volume; 1h/4h derived session-aligned from 15m.

## 0.7.1 — 2026-07-13

- **Stacked multi-timeframe trend gates.** `--gate` now accepts several
  `COL@INTERVAL` pairs (e.g. `minervini@1d,above_50ma@4h`), all ANDed —
  the higher timeframes select which stocks/bars are worth trading, the
  entry fires on the small `--interval`. Gate values map by the gate
  bar's close time (start + duration, forward-filled), so a 4h or 1h
  gate maps onto 5m bars with no lookahead, not just daily-by-date.

## 0.7.0 — 2026-07-13

- **Trade-with-the-trend is now the default.** `test_strategy.py` and
  `portfolio_sim.py` apply a daily trend gate (`minervini@1d`)
  automatically; pass `--gate=none` to disable or `--gate=COL@IV` to
  override. Trading against the trend is opt-in, not the default.
- **Profit-target exit** (`--target=0.10`) added to the simulator
  (`backtest_long`/`_simulate`), the harness, and the portfolio sim,
  alongside the existing stop, time, and exit-signal exits. Intrabar
  fills check stop before target (conservative).
- Portfolio sim gained exit-signal support (`--exit=col`), so open
  positions can be closed on a weakness signal, matching how the bot
  will watch trades (first of target / stop / signal / time to hit).
- New intraday research tools: `combo_fire_check.py` (combo firing
  counts per timeframe) and `combo_event_study.py` (combo edge, with
  optional trend gate).

## 0.6.1 — 2026-07-13

- `portfolio_sim.py` gains `--gate=COL@INTERVAL`: the portfolio-level
  simulator can now model the core product — a lower-timeframe bot
  trading only inside a higher-timeframe trend — with the same
  no-lookahead one-bar shift the harness uses.

## 0.6.0 — 2026-07-13

- **Alpaca data provider** (`weisswave/provider.py`): SIP-feed historical
  bars with split/dividend adjustment, regular-session filtering,
  Vault-backed credentials (never on disk), rate-limit handling, and
  share-class symbol mapping (BRK-B <-> BRK.B). Yahoo remains the
  default; `WEISSWAVE_PROVIDER=alpaca` switches.
- **Deep intraday backfill** (`alpaca_backfill.py`): resumable 15m
  history since 2018 plus a year of 5m; session-aligned 1h and 4h bars
  derived locally from 15m (Alpaca's clock-aligned hourly bars would
  mix grids and premarket volume).
- **Portfolio simulator** (`portfolio_sim.py`): finite capital,
  position cap, score-ranked candidate selection, compounding equity
  curve, CAGR/drawdown/exposure vs the equal-weight benchmark — the
  bridge between per-trade edge and account-level returns.
- Harness: `--gate=COL@INTERVAL` cross-timeframe trend gate (trade a
  lower timeframe only inside a higher-timeframe trend), and a stale
  signal-cache fallback so backtests keep running while a fetch or
  backfill holds the database write lock.

## 0.5.3 — 2026-07-12

- **Signal cache**: the harness builds full-history signals once per DB
  update (parquet keyed on the DB file's mtime) and every later run —
  any strategy, any window — loads in seconds. Backtests drop from ~90s
  to ~3-5s.
- `test_strategy.py --months=N` tests only the recent window (fast
  regime check; output flags it as in-sample and low-n so it is never
  mistaken for validation) and `--cost-bps=F` haircuts every trade by a
  round-trip cost estimate, so verdicts can be read after fees.

## 0.5.2 — 2026-07-12

- **One-command backtest harness** (`test_strategy.py`): test any entry
  combo / filter / exit / stop / hold / interval with a single CLI call
  against the existing database — train/test stats with excess vs the
  equal-weight benchmark and a plain-language verdict. `--saved=NAME`
  runs a strategies.json entry; `--list-signals` / `--list-saved` let a
  script (or LLM orchestrator) discover valid names; unknown columns
  fail with exit code 2. No fetching or installs — built to be the only
  command an automation loop needs.

## 0.5.1 — 2026-07-12

- Today's signals tab: new **Saved** entry-rule source loads strategies
  from `strategies.json`, and a save panel writes the currently
  configured rule (any source) back to it — one shared config between
  the dashboard, the nightly scanner, and future automation.
- `WEISSWAVE_DB` environment variable overrides the database path
  (defaults to `market.duckdb` in the working directory as before), so
  the project relocates without code edits.

## 0.5.0 — 2026-07-12

- **Nightly setup scanner** (`scan_today.py`): evaluates strategy
  configs from a local `strategies.json` (gitignored; see
  `strategies.example.json`) against the latest bar of every symbol in
  the DB and reports hits with close, stop level, and per-component
  fire distances. `--fetch` runs an incremental fetch first,
  `--lookback=N` widens the reporting window, results are saved as CSV
  under `scans/`. Built to run post-close from Task Scheduler / cron.

## 0.4.1 — 2026-07-12

- **Fixed: pivot detection on na-masked series.** Pine's `pivotlow`/
  `pivothigh` compare with NaN semantics (a comparison against na is
  false), so na neighbours can never veto a pivot — while the Python
  port required a full window of valid bars, meaning any signal built
  on a masked series (`cond ? value : na`) could never fire. New
  `pine_pivot_low_nan` / `pine_pivot_high_nan` in
  `weisswave/divergence.py` implement the faithful semantics; the
  strict variants are unchanged, so existing divergence signals are
  unaffected.

## 0.4.0 — 2026-07-10

- **Loss autopsy** (`weisswave/autopsy.py` + backtest-tab expander): one
  pass over a backtest's trades evaluates every candidate condition at
  the entry bar — regime filters, pressure states, WT zones, plus
  median splits on WaveTrend level / volume ratio / distance from 50MA —
  and reports the win-rate/return lift of each, losses by exit reason,
  market-driven vs stock-specific losses, and worst-dragging symbols.
- Sidebar quick-range presets (3m/6m/1y/2y/5y/10y/max/custom) replace
  manual date typing.
- "How to use" quick guide expander above the tabs.

## 0.3.2 — 2026-07-10

- Backtest chart: click a trade in the Trades table to highlight it as a
  gold star on the strategy-vs-market chart; optional checkbox overlays
  every trade as a green/red dot with hover details (off by default).

## 0.3.1 — 2026-07-10

- Strategy finder results table is row-clickable: clicking a row loads
  the full configuration (timeframe, entries, confluence, filter, exit,
  stop, hold) straight into the Strategy backtest tab.
- Fixed title/legend overlap on the strategy-vs-market chart.
- Added `__version__` (shown in the dashboard sidebar) and this changelog.

## 0.3.0 — 2026-07-10

- **Fine-resolution fill verification** (`weisswave/fills.py`): every
  backtest trade replayed against real 15m/5m/1m bars stored in the DB —
  entry/exit fills and stop touches re-derived from fine bars, per-trade
  deltas, coverage/bias summary, changed-outcome detection, and a trade
  inspector chart.
- **Provider abstraction** (`weisswave/provider.py`): documented
  normalized schema; swapping Yahoo for Polygon/Alpaca is one class.
- Fetcher: 15m/5m/1m intervals, `--intervals=` flag, split-on-failure
  chunk bisection.
- Backtest tab: regime filter, persistent results, fill-check section.
- Finder drill-down: "Load into Strategy backtest tab" handoff
  (including automatic timeframe switch).

## 0.2.0 — 2026-07-10

- **Buy-and-hold honesty check**: equal-weight universe benchmark; every
  trade carries `excess = return − market over the same span`; finder
  ranks by train excess; backtest tab shows market window return,
  beat-market rate, and warns when buy-and-hold won.
- Standard textbook signals in the open package (MACD cross,
  golden/death cross, RSI 30/70 crosses).
- Screener strategy presets (standard + WeissWave when present).

## 0.1.0 — 2026-07-09/10

- Weis Wave + WaveTrend signal engine (Pine v4 ports), divergences,
  Combined-v1 proprietary suite (excluded from public repo).
- DuckDB storage with duplicate-safe upserts, full 25-year daily history,
  incremental fetching with deep-backfill detection.
- Streamlit dashboard: database, chart explorer, event study, two-stage
  strategy finder with train/test split, backtester, screener.
- Date-range scoping with indicator warm-up; per-symbol drill-down;
  bot-config JSON export.
