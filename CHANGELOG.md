# Changelog

All notable changes to WeissWave. Each release maps to git commits on
`main`; run `git log --oneline` for the full trail.

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
