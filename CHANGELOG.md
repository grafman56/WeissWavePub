# Changelog

All notable changes to WeissWave. Each release maps to git commits on
`main`; run `git log --oneline` for the full trail.

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
