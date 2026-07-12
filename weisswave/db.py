"""DuckDB storage layer for OHLCV bars.

Design notes
------------
- One row per (symbol, interval, ts) — enforced by the PRIMARY KEY, so
  re-fetching an overlapping window can never duplicate data: incoming
  rows for existing keys REPLACE the old row. That is deliberate for
  time-sensitive data: a bar fetched while the session was still open is
  partial, and the next fetch silently corrects it.
- Timestamps are stored naive:
    * daily/weekly/monthly bars -> midnight of the TRADING DATE
    * intraday bars             -> UTC
  Normalization happens in fetch_data.py before rows reach this module.
- `fetched_at` (UTC) records when each row was last written, so you can
  always tell how stale a bar is.
"""

import os
from datetime import datetime, timezone

import duckdb
import pandas as pd

DB_PATH = os.environ.get("WEISSWAVE_DB", "market.duckdb")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol     VARCHAR   NOT NULL,
    interval   VARCHAR   NOT NULL,
    ts         TIMESTAMP NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    volume     DOUBLE,
    adjclose   DOUBLE,
    fetched_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, interval, ts)
);
"""

PRICE_COLS = ["symbol", "interval", "ts", "open", "high", "low",
              "close", "volume", "adjclose"]


def connect(path: str = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path, read_only=read_only)
    if not read_only:
        con.execute(_SCHEMA)
    return con


def upsert_prices(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Insert-or-replace price rows. `df` must have PRICE_COLS. Returns the
    number of incoming rows (inserts + replacements)."""
    if df.empty:
        return 0
    df = df[PRICE_COLS].copy()
    df["fetched_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    con.register("incoming", df)
    con.execute(f"""
        INSERT OR REPLACE INTO prices ({", ".join(PRICE_COLS)}, fetched_at)
        SELECT {", ".join(PRICE_COLS)}, fetched_at FROM incoming
    """)
    con.unregister("incoming")
    return len(df)


def last_timestamps(con: duckdb.DuckDBPyConnection, interval: str) -> dict:
    """{symbol: latest ts} for one interval — drives incremental fetching."""
    rows = con.execute(
        "SELECT symbol, max(ts) FROM prices WHERE interval = ? GROUP BY symbol",
        [interval]).fetchall()
    return {sym: ts for sym, ts in rows}


def first_timestamps(con: duckdb.DuckDBPyConnection, interval: str) -> dict:
    """{symbol: earliest ts} — detects symbols whose history needs a
    deeper backfill when the configured lookback grows."""
    rows = con.execute(
        "SELECT symbol, min(ts) FROM prices WHERE interval = ? GROUP BY symbol",
        [interval]).fetchall()
    return {sym: ts for sym, ts in rows}


def list_symbols(con: duckdb.DuckDBPyConnection, interval: str = "1d") -> list:
    rows = con.execute(
        "SELECT DISTINCT symbol FROM prices WHERE interval = ? ORDER BY symbol",
        [interval]).fetchall()
    return [r[0] for r in rows]


def load_prices(con: duckdb.DuckDBPyConnection, symbol: str,
                interval: str = "1d", start=None, end=None) -> pd.DataFrame:
    """One symbol's bars as an OHLCV DataFrame (capitalized columns,
    ts index) — the shape weisswave.signals expects. Optional start/end
    (inclusive) restrict the range in SQL."""
    q = """
        SELECT ts, open AS "Open", high AS "High", low AS "Low",
               close AS "Close", volume AS "Volume", adjclose AS "AdjClose"
        FROM prices WHERE symbol = ? AND interval = ?
    """
    args = [symbol, interval]
    if start is not None:
        q += " AND ts >= ?"
        args.append(pd.Timestamp(start).to_pydatetime())
    if end is not None:
        q += " AND ts <= ?"
        args.append(pd.Timestamp(end).to_pydatetime())
    df = con.execute(q + " ORDER BY ts", args).df()
    return df.set_index("ts")


def coverage_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-interval summary: symbols, rows, date range, freshest fetch."""
    return con.execute("""
        SELECT interval,
               count(DISTINCT symbol) AS symbols,
               count(*)               AS rows,
               min(ts)                AS first_bar,
               max(ts)                AS last_bar,
               max(fetched_at)        AS last_fetch_utc
        FROM prices GROUP BY interval ORDER BY interval
    """).df()
