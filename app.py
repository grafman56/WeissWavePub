#!/usr/bin/env python3
"""
WeissWave dashboard — run the data fetcher, inspect signals on charts,
run the event study, and build/backtest strategies with adjustable
weights, timeframes, and filters.

Launch:  python -m streamlit run app.py
"""

import contextlib
import io
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from weisswave.db import DB_PATH, connect, coverage_report, list_symbols, load_prices
from weisswave.optimize import (EXIT_OPTIONS, evaluate_config,
                                find_strategies, per_symbol_stats)
from weisswave.signals import (FILTER_COLUMNS, SIGNAL_COLUMNS_BEAR,
                               SIGNAL_COLUMNS_BULL, build_signals,
                               combine_signals)
from weisswave.study import backtest_long, event_study

st.set_page_config(page_title="WeissWave", page_icon="chart_with_upwards_trend",
                   layout="wide")

HORIZONS = (1, 3, 5, 10, 20)


# ── data access (cached; invalidated whenever the DB file changes) ────────────

def db_stamp() -> float:
    return os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0.0


@st.cache_data(show_spinner=False)
def get_symbols(interval: str, stamp: float) -> list:
    con = connect(read_only=True)
    syms = list_symbols(con, interval)
    con.close()
    return syms


@st.cache_data(show_spinner=False, max_entries=6)
def get_universe_signals(interval: str, symbols: tuple, params: dict,
                         start, end, stamp: float) -> dict:
    """{symbol: signal DataFrame} for the selection, scoped to [start, end].

    Indicators need history before `start` to be warmed up (200-bar SMA,
    EMAs, wave state), so extra bars are loaded before the window and
    trimmed after computing — the first visible bar is fully warmed."""
    warmup = pd.Timedelta(days=500 if interval == "1d" else 60)
    lo = pd.Timestamp(start) - warmup
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)   # include end date fully
    con = connect(read_only=True)
    out = {}
    progress = st.progress(0.0, text="Computing signals...")
    for i, s in enumerate(symbols):
        df = load_prices(con, s, interval, lo, hi)
        if len(df) >= 60:
            sig = build_signals(df, **params)
            sig = sig[(sig.index >= pd.Timestamp(start)) & (sig.index < hi)]
            if len(sig) >= 30:
                out[s] = sig
        progress.progress((i + 1) / len(symbols),
                          text=f"Computing signals... {s} ({i + 1}/{len(symbols)})")
    progress.empty()
    con.close()
    return out


# ── sidebar: global settings ──────────────────────────────────────────────────

from datetime import date, timedelta

with st.sidebar:
    st.title("WeissWave")
    interval = st.radio("Timeframe", ["1d", "1h"], horizontal=True)
    all_symbols = get_symbols(interval, db_stamp())
    st.caption(f"{len(all_symbols)} symbols in DB for {interval}")
    chosen = st.multiselect("Symbols (empty = all)", all_symbols)
    symbols = tuple(chosen or all_symbols)

    st.subheader("Period")
    picked = st.date_input(
        "Date range (applies to every tab)",
        (date.today() - timedelta(days=730), date.today()),
        min_value=date(1990, 1, 1), max_value=date.today())
    if isinstance(picked, tuple) and len(picked) == 2:
        start_date, end_date = picked
    else:   # mid-selection: only one endpoint picked so far
        start_date = picked[0] if isinstance(picked, tuple) else picked
        end_date = date.today()
    st.caption(f"{start_date} to {end_date}")

    st.subheader("WaveTrend")
    wt_channel = st.number_input("Channel length", 2, 50, 9)
    wt_average = st.number_input("Average length", 2, 50, 12)
    wt_smooth = st.number_input("Smoothing", 1, 10, 3)
    ob_level = st.number_input("Overbought", 0, 100, 60)
    os_level = st.number_input("Oversold", -100, 0, -60)

    st.subheader("Weis Wave")
    pullback = st.number_input("Trend detection length", 1, 10, 2)
    heavy_mult = st.number_input("Heavy volume multiplier", 1.0, 20.0, 4.0, 0.5)
    very_heavy_mult = st.number_input("Very heavy multiplier", 2.0, 50.0, 10.0, 0.5)

    st.subheader("Composites")
    confirm_window = st.number_input("Confirmation window (bars)", 0, 20, 3)

params = dict(wt_channel=int(wt_channel), wt_average=int(wt_average),
              wt_smooth=int(wt_smooth), ob_level=float(ob_level),
              os_level=float(os_level), pullback=int(pullback),
              very_heavy_mult=float(very_heavy_mult),
              heavy_mult=float(heavy_mult),
              confirm_window=int(confirm_window))

tab_db, tab_chart, tab_study, tab_finder, tab_bt, tab_today = st.tabs(
    ["Database", "Chart explorer", "Event study", "Strategy finder",
     "Strategy backtest", "Today's signals"])


# ── tab 1: database status + fetch ────────────────────────────────────────────

with tab_db:
    st.subheader("Coverage")
    if os.path.exists(DB_PATH):
        con = connect(read_only=True)
        cov = coverage_report(con)
        con.close()
        st.dataframe(cov, width="stretch", hide_index=True)
        if len(cov):
            age = pd.Timestamp.utcnow().tz_localize(None) - cov["last_fetch_utc"].max()
            hours = age.total_seconds() / 3600
            (st.success if hours < 24 else st.warning)(
                f"Freshest data fetched {hours:.1f} hours ago (UTC).")
    else:
        st.info("No database yet — fetch below to create it.")

    st.subheader("Fetch")
    st.caption("Incremental and duplicate-safe: existing bars are overwritten "
               "in place, never appended. Progress prints below.")
    col1, col2 = st.columns(2)
    do_sel = col1.button("Fetch selected symbols",
                         disabled=not chosen, width="stretch")
    do_all = col2.button("Fetch full S&P 500 (takes a few minutes)",
                         width="stretch")
    if do_sel or do_all:
        from fetch_data import get_sp500_tickers, run
        with st.spinner("Fetching..."):
            targets = list(chosen) if do_sel else get_sp500_tickers()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run(targets)
        st.code(buf.getvalue() or "(no output)")
        st.cache_data.clear()
        st.rerun()


# ── tab 2: chart explorer ─────────────────────────────────────────────────────

with tab_chart:
    if not all_symbols:
        st.info("Fetch some data first (Database tab).")
    else:
        c1, c2 = st.columns([1, 3])
        sym = c1.selectbox("Symbol", symbols)
        marks = c2.multiselect(
            "Signals to mark on the chart",
            SIGNAL_COLUMNS_BULL + SIGNAL_COLUMNS_BEAR,
            default=["wtv_buy", "wtv_sell"])
        sig = get_universe_signals(interval, (sym,), params, start_date,
                                   end_date, db_stamp()).get(sym)
        if sig is None:
            st.warning(f"Not enough {interval} data for {sym}.")
        else:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03,
                                subplot_titles=(sym, "WaveTrend", "Weis Wave volume"))
            fig.add_trace(go.Candlestick(
                x=sig.index, open=sig["Open"], high=sig["High"],
                low=sig["Low"], close=sig["Close"], name=sym,
                showlegend=False), row=1, col=1)
            for m in marks:
                pts = sig.index[sig[m].astype(bool)]
                if not len(pts):
                    continue
                bull = m in SIGNAL_COLUMNS_BULL
                y = sig.loc[pts, "Low"] * 0.995 if bull else sig.loc[pts, "High"] * 1.005
                fig.add_trace(go.Scatter(
                    x=pts, y=y, mode="markers", name=m,
                    marker=dict(symbol="triangle-up" if bull else "triangle-down",
                                size=10)), row=1, col=1)
            fig.add_trace(go.Scatter(x=sig.index, y=sig["wt1"], name="wt1",
                                     line=dict(width=1.5)), row=2, col=1)
            fig.add_trace(go.Scatter(x=sig.index, y=sig["wt2"], name="wt2",
                                     line=dict(width=1)), row=2, col=1)
            fig.add_hline(y=params["ob_level"], line_dash="dot", row=2, col=1)
            fig.add_hline(y=params["os_level"], line_dash="dot", row=2, col=1)
            fig.add_trace(go.Scatter(x=sig.index, y=sig["volumeup"], name="volumeup",
                                     line=dict(color="green", width=1)), row=3, col=1)
            fig.add_trace(go.Scatter(x=sig.index, y=sig["volumedn"], name="volumedn",
                                     line=dict(color="red", width=1)), row=3, col=1)
            fig.update_layout(height=750, xaxis_rangeslider_visible=False,
                              margin=dict(t=40, b=20),
                              legend=dict(orientation="h", y=1.06))
            st.plotly_chart(fig, width="stretch")


# ── tab 3: event study ────────────────────────────────────────────────────────

with tab_study:
    st.caption("Forward returns (next-open entry) after each signal, versus the "
               "all-bars baseline. 'edge' columns are the difference — a signal "
               "only earns a spot in a strategy if its edge holds up across "
               "horizons and has a decent event count.")
    if st.button("Run event study", type="primary"):
        frames = get_universe_signals(interval, symbols, params, start_date,
                                      end_date, db_stamp())
        if not frames:
            st.warning("No data — fetch first.")
        else:
            pooled = pd.concat(frames.values())
            st.write(f"{len(frames)} symbols, {len(pooled):,} bars")
            for label, cols in (("Bull signals", SIGNAL_COLUMNS_BULL),
                                ("Bear signals", SIGNAL_COLUMNS_BEAR)):
                stats = event_study(pooled, cols, HORIZONS)
                pct_cols = [c for c in stats.columns if c != "n_events"]
                styled = (stats.style
                          .format("{:.2%}", subset=pct_cols, na_rep="-")
                          .map(lambda v: "color: #2e7d32" if isinstance(v, float) and v > 0
                               else "color: #c62828" if isinstance(v, float) and v < 0
                               else "",
                               subset=[c for c in pct_cols if c.startswith("edge")]))
                st.subheader(label)
                st.dataframe(styled, width="stretch")


# ── tab 4: strategy backtest ──────────────────────────────────────────────────

with tab_bt:
    st.caption("Entry fires when at least one selected entry signal fires AND "
               "the required number of distinct entry signals have fired within "
               "the confluence window. Exits on any selected exit signal, the "
               "stop, or max holding time. Fills at next bar's open.")
    c1, c2 = st.columns(2)
    entry_cols = c1.multiselect("Entry signals (bull)", SIGNAL_COLUMNS_BULL,
                                default=["wt_cross_up_oversold"])
    exit_cols = c2.multiselect("Exit signals (bear)", SIGNAL_COLUMNS_BEAR,
                               default=["wt_cross_down_overbought"])
    c3, c4, c5, c6 = st.columns(4)
    min_count = c3.number_input("Min entry signals in confluence", 1,
                                max(1, len(entry_cols)), 1)
    conf_win = c4.number_input("Confluence window (bars)", 0, 30, 5)
    stop_pct = c5.number_input("Stop loss %", 0.0, 50.0, 8.0, 0.5)
    max_bars = c6.number_input("Max holding (bars, 0 = none)", 0, 250, 20)

    if st.button("Run backtest", type="primary", disabled=not entry_cols):
        frames = get_universe_signals(interval, symbols, params, start_date,
                                      end_date, db_stamp())
        all_trades = []
        for s, sig in frames.items():
            entry = combine_signals(sig, entry_cols, int(min_count), int(conf_win))
            exit_ = (sig[exit_cols].any(axis=1) if exit_cols
                     else pd.Series(False, index=sig.index))
            res = backtest_long(sig, entry, exit_,
                                stop_loss=stop_pct / 100 if stop_pct else None,
                                max_bars=int(max_bars) or None)
            if res.n_trades:
                res.trades["symbol"] = s
                all_trades.append(res.trades)
        if not all_trades:
            st.warning("No trades triggered with these settings.")
        else:
            trades = pd.concat(all_trades, ignore_index=True) \
                .sort_values("exit_idx").reset_index(drop=True)
            wins = trades.loc[trades.ret > 0, "ret"].sum()
            losses = -trades.loc[trades.ret < 0, "ret"].sum()
            m = st.columns(6)
            m[0].metric("Trades", len(trades))
            m[1].metric("Win rate", f"{(trades.ret > 0).mean():.1%}")
            m[2].metric("Avg return", f"{trades.ret.mean():.2%}")
            m[3].metric("Median", f"{trades.ret.median():.2%}")
            m[4].metric("Profit factor",
                        f"{wins / losses:.2f}" if losses > 0 else "inf")
            m[5].metric("Avg bars held", f"{trades.bars.mean():.1f}")

            # Trades from different symbols overlap in time, so compounding
            # them sequentially would be meaningless; sum fixed-stake returns.
            eq = trades.set_index("exit_idx")["ret"].cumsum() * 100
            fig = go.Figure(go.Scatter(x=eq.index, y=eq, mode="lines",
                                       name="equity"))
            fig.update_layout(title="Cumulative return, % (fixed 1-unit stake "
                                    "per trade, ordered by exit date)",
                              yaxis_ticksuffix="%",
                              height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig, width="stretch")

            st.subheader("Exit reasons")
            st.dataframe(trades.groupby("exit_reason")["ret"]
                         .agg(n="count", avg="mean", win_rate=lambda r: (r > 0).mean())
                         .style.format({"avg": "{:.2%}", "win_rate": "{:.1%}"}),
                         width="stretch")
            st.subheader("Trades")
            st.dataframe(trades[["symbol", "entry_idx", "exit_idx", "entry_px",
                                 "exit_px", "ret", "bars", "exit_reason"]]
                         .style.format({"entry_px": "{:.2f}", "exit_px": "{:.2f}",
                                        "ret": "{:.2%}"}),
                         width="stretch", height=400)


# ── tab 5: strategy finder (automated search) ─────────────────────────────────

STOP_CHOICES = {"none": None, "5%": 0.05, "8%": 0.08, "12%": 0.12}
FINDER_CACHE = "finder_results.pkl"


def _pct_style(df, pct_cols, signed_cols):
    return (df.style
            .format({c: "{:.2%}" for c in pct_cols} |
                    {c: "{:.2f}" for c in df.columns if c.endswith("_pf")},
                    na_rep="-")
            .map(lambda v: "color: #2e7d32" if isinstance(v, float) and v > 0
                 else "color: #c62828" if isinstance(v, float) and v < 0
                 else "", subset=signed_cols))


with tab_finder:
    st.caption(
        "Automated search: stage 1 scores every entry-signal combination "
        "(singles and pairs, optionally gated by a regime filter) by forward-"
        "return edge on the TRAIN portion of history. Stage 2 fully simulates "
        "the best combos across the exit/stop/hold grid on train and test "
        "separately. Rankings use train performance only — trust a row when "
        "its TEST columns are also green with a decent trade count.")
    f1, f2, f3, f4, f5 = st.columns(5)
    intervals_avail = [iv for iv in ("1d", "1h") if get_symbols(iv, db_stamp())]
    search_ivs = f1.multiselect("Timeframes", intervals_avail,
                                default=intervals_avail[:1])
    top_k = f2.number_input("Combos to simulate (stage 2)", 3, 25, 10)
    min_events = f3.number_input("Min train events (stage 1)", 20, 500, 80)
    conf_window = f4.number_input("Confluence window", 1, 20, 5)
    train_frac = f5.slider("Train fraction", 0.5, 0.9, 0.7, 0.05)
    g1, g2, g3 = st.columns(3)
    exit_sel = g1.multiselect("Exit rules to try", list(EXIT_OPTIONS),
                              default=["none", "wt_cross_down"])
    stop_sel = g2.multiselect("Stops to try", list(STOP_CHOICES),
                              default=["8%"])
    hold_sel = g3.multiselect("Max holds to try", [5, 10, 20, 40],
                              default=[10, 20])

    if st.button("Find strategies", type="primary",
                 disabled=not (search_ivs and exit_sel and stop_sel and hold_sel)):
        all_res = []
        for iv in search_ivs:
            syms_iv = tuple(chosen or get_symbols(iv, db_stamp()))
            frames = get_universe_signals(iv, syms_iv, params, start_date,
                                          end_date, db_stamp())
            bar = st.progress(0.0, text=f"Searching {iv}...")
            _ranked, res = find_strategies(
                frames, window=int(conf_window), train_frac=float(train_frac),
                min_events=int(min_events), top_k=int(top_k),
                exit_names=tuple(exit_sel),
                stops=tuple(STOP_CHOICES[s] for s in stop_sel),
                holds=tuple(int(h) for h in hold_sel),
                progress_cb=lambda f, t: bar.progress(f, text=f"[{iv}] {t}"))
            bar.empty()
            if not res.empty:
                res.insert(0, "interval", iv)
                all_res.append(res)
        if not all_res:
            st.warning("Nothing passed the minimum-events bar — lower it or "
                       "widen the symbol selection.")
        else:
            res = pd.concat(all_res, ignore_index=True) \
                .sort_values("train_avg", ascending=False).reset_index(drop=True)
            st.session_state["finder_results"] = res
            st.session_state["finder_window"] = int(conf_window)
            st.session_state["finder_train_frac"] = float(train_frac)
            # Persist so results survive page reloads / app restarts.
            pd.to_pickle({"results": res, "window": int(conf_window),
                          "train_frac": float(train_frac)}, FINDER_CACHE)

    if "finder_results" not in st.session_state and os.path.exists(FINDER_CACHE):
        saved = pd.read_pickle(FINDER_CACHE)
        if "entry_cols" in saved["results"].columns:  # ignore stale schemas
            st.session_state["finder_results"] = saved["results"]
            st.session_state["finder_window"] = saved["window"]
            st.session_state["finder_train_frac"] = saved.get("train_frac", 0.7)
            st.caption("Showing results from the last saved search.")

    if "finder_results" in st.session_state:
        res = st.session_state["finder_results"]
        cols = (["interval"] if "interval" in res.columns else []) + \
            ["entry", "filter", "exit", "stop", "max_hold",
             "train_n", "train_win", "train_avg", "train_pf",
             "test_n", "test_win", "test_avg", "test_pf"]
        show = res[cols].head(25)
        st.subheader("Best configurations (ranked by train avg return)")
        st.dataframe(_pct_style(
            show,
            [c for c in show.columns if c.endswith(("_win", "_avg"))],
            ["train_avg", "test_avg"]),
            width="stretch", height=520)
        robust = res[(res.test_avg > 0) & (res.test_n >= 20)]
        st.caption(f"{len(robust)} of {len(res)} configs are also positive "
                   f"out-of-sample with >=20 test trades. Click a column "
                   f"header to re-sort (e.g. by test_avg).")

        # ── drill-down: where does a config actually work? ────────────────
        with st.expander("Drill into a configuration (per-symbol results, "
                         "bot export)"):
            labels = [f"#{i + 1}  [{r.get('interval', interval)}]  "
                      f"{r['entry']}  |  {r['filter']}  |  exit {r['exit']}, "
                      f"stop {r['stop']}, hold {r['max_hold']}"
                      for i, r in res.head(15).iterrows()]
            pick = st.selectbox("Configuration", labels, key="drill_pick")
            row = res.iloc[labels.index(pick)]
            row_iv = row.get("interval", interval)
            filter_col = None if row["filter"] == "-" else row["filter"]
            if st.button("Analyze per symbol"):
                syms_iv = tuple(chosen or get_symbols(row_iv, db_stamp()))
                frames = get_universe_signals(row_iv, syms_iv, params,
                                              start_date, end_date, db_stamp())
                trades = evaluate_config(
                    frames, list(row["entry_cols"]), int(row["min_count"]),
                    int(row["window"]), filter_col,
                    EXIT_OPTIONS[row["exit"]], row["stop_value"],
                    int(row["max_hold"]),
                    st.session_state.get("finder_train_frac", 0.7))
                if trades.empty:
                    st.info("No trades for this configuration.")
                else:
                    stats = per_symbol_stats(trades, min_trades=1)
                    c1, c2 = st.columns(2)
                    c1.metric("Symbols traded", trades["symbol"].nunique())
                    c2.metric("Symbols with positive avg return",
                              int((stats["avg_ret"] > 0).sum()))
                    st.dataframe(stats.style.format(
                        {"win_rate": "{:.0%}", "avg_ret": "{:.2%}",
                         "total_ret": "{:.1%}", "test_avg": "{:.2%}"},
                        na_rep="-"), width="stretch", height=380)
            bot_config = {
                "name": f"{row['entry']} | {row['filter']}",
                "interval": row_iv,
                "indicator_params": params,
                "entry": {"signals": list(row["entry_cols"]),
                          "min_count": int(row["min_count"]),
                          "confluence_window": int(row["window"]),
                          "regime_filter": filter_col},
                "exit": {"signals": EXIT_OPTIONS[row["exit"]],
                         "stop_loss": row["stop_value"],
                         "max_hold_bars": int(row["max_hold"])},
                "execution": {"fill": "next_bar_open"},
                "backtest_stats": {
                    "train": {"n": int(row["train_n"]),
                              "win_rate": round(float(row["train_win"]), 4),
                              "avg_return": round(float(row["train_avg"]), 5)},
                    "test": {"n": int(row["test_n"]),
                             "win_rate": round(float(row["test_win"]), 4)
                             if pd.notna(row["test_win"]) else None,
                             "avg_return": round(float(row["test_avg"]), 5)
                             if pd.notna(row["test_avg"]) else None}},
            }
            st.download_button("Download bot config (JSON)",
                               json.dumps(bot_config, indent=2),
                               file_name="strategy_config.json",
                               mime="application/json")


# ── tab 6: today's signals (manual-entry screener) ────────────────────────────

with tab_today:
    st.caption("Which symbols fire a chosen entry rule at the END of the "
               "selected date range — with the range ending today this is "
               "your live manual-entry shortlist; with a historical end date "
               "it shows what the screen would have flagged then. Refresh "
               "data on the Database tab first; the latest daily bar is "
               "partial if fetched while the market is open.")
    have_finder = "finder_results" in st.session_state
    source = st.radio("Entry rule", ["Manual"] + (["From finder"] if have_finder else []),
                      horizontal=True)
    screen_interval = interval
    if source == "From finder":
        res = st.session_state["finder_results"]
        labels = [f"#{i + 1}  [{r.get('interval', interval)}]  {r['entry']}  |  "
                  f"filter: {r['filter']}  (test avg "
                  f"{r['test_avg']:.2%})" if pd.notna(r['test_avg'])
                  else f"#{i + 1}  {r['entry']}  |  filter: {r['filter']}"
                  for i, r in res.head(10).iterrows()]
        pick = st.selectbox("Finder result", labels)
        row = res.iloc[labels.index(pick)]
        entry_cols = list(row["entry_cols"])
        min_count = int(row["min_count"])
        filter_col = None if row["filter"] == "-" else row["filter"]
        window = int(row.get("window", st.session_state.get("finder_window", 5)))
        screen_interval = row.get("interval", interval)
        st.write(f"Entry [{screen_interval}]: **{' + '.join(entry_cols)}** "
                 f"(>= {min_count} in {window} bars), "
                 f"filter: **{filter_col or 'none'}**")
    else:
        s1, s2, s3, s4 = st.columns([3, 1, 1, 2])
        entry_cols = s1.multiselect("Entry signals", SIGNAL_COLUMNS_BULL,
                                    default=["wtv_buy"], key="today_entries")
        min_count = int(s2.number_input("Min count", 1,
                                        max(1, len(entry_cols)), 1,
                                        key="today_mc"))
        window = int(s3.number_input("Window", 0, 20, 5, key="today_win"))
        filter_pick = s4.selectbox("Regime filter", ["none"] + FILTER_COLUMNS)
        filter_col = None if filter_pick == "none" else filter_pick

    lookback = int(st.number_input("Fired within the last N bars", 1, 10, 2))
    if st.button("Screen universe", type="primary", disabled=not entry_cols):
        syms_scr = tuple(chosen or get_symbols(screen_interval, db_stamp()))
        frames = get_universe_signals(screen_interval, syms_scr, params,
                                      start_date, end_date, db_stamp())
        rows = []
        for s, sig in frames.items():
            entry = combine_signals(sig, entry_cols, min_count, window)
            if filter_col:
                entry = entry & sig[filter_col].astype(bool)
            tail = entry.iloc[-lookback:]
            if not tail.any():
                continue
            fired_at = tail.index[tail][-1]
            comps = [c for c in entry_cols if sig[c].iloc[-(lookback + window):].any()]
            rows.append({"symbol": s, "signal_bar": fired_at,
                         "close": sig["Close"].iloc[-1],
                         "components": ", ".join(comps),
                         "filter_ok": bool(sig[filter_col].iloc[-1]) if filter_col else True})
        if not rows:
            st.info("No symbols fire this rule in the lookback window.")
        else:
            hits = pd.DataFrame(rows).sort_values("signal_bar", ascending=False)
            st.write(f"**{len(hits)} candidates**")
            st.dataframe(hits.style.format({"close": "{:.2f}"}),
                         width="stretch", hide_index=True)
