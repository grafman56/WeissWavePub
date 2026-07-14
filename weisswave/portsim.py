"""Unified portfolio simulator — one numba-jitted loop over a 2D (time x
symbol) grid, modeled on VectorBT's architecture but custom (no dependency).

Single source of truth for exit logic (stop / target / trailing / signal /
time) shared by every caller, so bugs can't hide in a duplicate loop. Fast
enough to sweep: the inner loop is @njit-compiled.

Design
------
Callers build aligned 2D numpy arrays (all symbols on a common sorted time
grid, missing bars masked) plus per-bar "best strategy" entry/score/params,
then call `simulate()`. Trades come back as flat numpy arrays (records).

No lookahead: an entry fires on bar t only if its signal was true on t-1
(passed in already shifted), and the stop reference uses t-1. Exits are
checked intrabar (stop before target, conservative); signal/time exit at
the open of the bar after the trigger.
"""

import numpy as np
from numba import njit

# exit reason codes
STOP, TRAIL, TARGET, SIGNAL, TIME, EOD = 1, 2, 3, 4, 5, 6
# stop-placement modes (how the initial stop is set at entry)
PCT, ATR, SWING, FIB = 0, 1, 2, 3
# trailing modes (how the stop ratchets once in profit)
#   PCT    - a fixed % under the high-water mark
#   STRUCT - under the latest confirmed swing low (pullback low)
#   FIB    - under each fib/extension level once price closes above it (so a
#            winner is protected at each rung but never taken off "just because")
TRAIL_PCT, TRAIL_STRUCT, TRAIL_FIB = 0, 1, 2
# fib entry modes (extra entry filter/trigger off the fib retracement zone)
#   OFF          - strategy signals only (fib unused for entry)
#   ZONE         - signal AND price currently in the [lo, hi] retr. band
#   BOUNCE       - signal AND a confirmed bounce off that band
#   BOUNCE_TREND - the bounce IS the entry, gated only by the trend (no signal)
FIB_ENTRY_OFF, FIB_ENTRY_ZONE, FIB_ENTRY_BOUNCE, FIB_ENTRY_BOUNCE_TREND = \
    0, 1, 2, 3


@njit(cache=True)
def _simulate(open2d, high2d, low2d, close2d, valid,
              ent, score, sidx, ext,
              atr, swing, p1, p2, p3, gate,
              strat_stop, strat_hold, strat_target,
              stop_mode, atr_mult, swing_buf,
              fib_stop_ratio, fib_buf,
              trail_act, trail_dist, trail_mode, use_fib_target, fib_ext,
              fib_entry, fib_zone_lo, fib_zone_hi, fib_bounce_look,
              cost_side, max_pos, init_cash):
    T, S = open2d.shape
    cash = init_cash
    held = np.zeros(S, np.bool_)
    p_shares = np.zeros(S)
    p_entry = np.zeros(S)
    p_stop = np.zeros(S)
    p_tp = np.zeros(S)
    p_hold = np.zeros(S, np.int64)
    p_held = np.zeros(S, np.int64)
    p_hwm = np.zeros(S)
    p_p1 = np.zeros(S)          # fib anchors snapshotted at entry, for the
    p_p2 = np.zeros(S)          # fib-ladder trail (the drawn fib is fixed for
    p_p3 = np.zeros(S)          # the life of the trade)
    last_close = np.zeros(S)
    n_open = 0

    p_sidx = np.zeros(S, np.int64)
    cap = T * max_pos + S + 16
    r_sym = np.empty(cap, np.int64)
    r_ret = np.empty(cap)
    r_reason = np.empty(cap, np.int64)
    r_bars = np.empty(cap, np.int64)
    r_strat = np.empty(cap, np.int64)
    ntr = 0
    equity = np.empty(T)
    invested = np.empty(T)

    for t in range(T):
        # update last_close for marking
        for s in range(S):
            if valid[t, s]:
                last_close[s] = close2d[t, s]

        # ---- exits ---------------------------------------------------------
        for s in range(S):
            if not held[s] or not valid[t, s]:
                continue
            p_held[s] += 1
            # trailing: ratchet the stop up once the trade is in profit.
            #   pct       -> under the high-water mark (fixed distance)
            #   structure -> under the latest confirmed swing low (p3), which
            #                fixes winners getting shaken out by a fixed %.
            #   fib       -> under each fib/extension rung once price has
            #                cleared it (protect the level, let the rest run).
            # structure/fib are on whenever selected; pct needs trail_act>0
            # (its original gate, unchanged).
            do_trail = (trail_mode != TRAIL_PCT) or (trail_act > 0.0)
            if do_trail and p_hwm[s] >= p_entry[s] * (1.0 + trail_act):
                cand = -1.0
                if trail_mode == TRAIL_STRUCT:
                    cand = p3[t - 1, s] * (1.0 - swing_buf)
                elif trail_mode == TRAIL_FIB:
                    # trail just under the highest fib level the high-water mark
                    # has cleared: first the prior swing high P2, then each
                    # extension rung P3 + r*(P2-P1). Protect the level, let the
                    # rest of the move run.
                    span = p_p2[s] - p_p1[s]
                    if span > 0.0:
                        if p_hwm[s] >= p_p2[s]:
                            lvl = p_p2[s] * (1.0 - swing_buf)
                            if lvl > cand:
                                cand = lvl
                        if np.isfinite(p_p3[s]):
                            for k in range(fib_ext.shape[0]):
                                rung = p_p3[s] + fib_ext[k] * span
                                if p_hwm[s] >= rung:
                                    lvl = rung * (1.0 - swing_buf)
                                    if lvl > cand:
                                        cand = lvl
                else:
                    cand = p_hwm[s] * (1.0 - trail_dist)
                if np.isfinite(cand) and cand > p_stop[s]:
                    p_stop[s] = cand
            xpx = -1.0
            reason = 0
            if low2d[t, s] <= p_stop[s]:
                xpx = open2d[t, s] if open2d[t, s] < p_stop[s] else p_stop[s]
                reason = TRAIL if p_stop[s] > p_entry[s] else STOP
            elif p_tp[s] > 0.0 and high2d[t, s] >= p_tp[s]:
                xpx = open2d[t, s] if open2d[t, s] > p_tp[s] else p_tp[s]
                reason = TARGET
            elif t > 0 and ext[t - 1, s]:
                xpx = open2d[t, s]
                reason = SIGNAL
            elif p_hold[s] > 0 and p_held[s] >= p_hold[s]:
                xpx = open2d[t, s]
                reason = TIME
            if xpx > 0.0:
                xpx *= (1.0 - cost_side)
                cash += p_shares[s] * xpx
                r_sym[ntr] = s
                r_ret[ntr] = xpx / p_entry[s] - 1.0
                r_reason[ntr] = reason
                r_bars[ntr] = p_held[s]
                r_strat[ntr] = p_sidx[s]
                ntr += 1
                held[s] = False
                n_open -= 1
            else:
                if high2d[t, s] > p_hwm[s]:
                    p_hwm[s] = high2d[t, s]

        # ---- entries: rank candidates by score, fill free slots -----------
        if n_open < max_pos and t > 0:
            # base entry signal: the strategy signal, except BOUNCE_TREND
            # mode where the trend gate alone qualifies (the fib bounce below
            # is then the actual trigger).
            trend_only = fib_entry == FIB_ENTRY_BOUNCE_TREND
            ncand = 0
            for s in range(S):
                base_ok = gate[t - 1, s] > 0.5 if trend_only else ent[t - 1, s]
                if valid[t, s] and (not held[s]) and base_ok:
                    ncand += 1
            if ncand > 0:
                cs = np.empty(ncand, np.int64)
                sc = np.empty(ncand)
                k = 0
                for s in range(S):
                    base_ok = (gate[t - 1, s] > 0.5 if trend_only
                               else ent[t - 1, s])
                    if valid[t, s] and (not held[s]) and base_ok:
                        cs[k] = s
                        sc[k] = score[t - 1, s]
                        k += 1
                order = np.argsort(-sc)          # highest score first
                mkt = 0.0
                for s in range(S):
                    if held[s]:
                        mkt += p_shares[s] * last_close[s]
                eq_now = cash + mkt
                for oi in range(ncand):
                    if n_open >= max_pos:
                        break
                    s = cs[order[oi]]
                    # fib entry filter on the [lo, hi] retracement band of the
                    # current up-leg L->H. All refs are bars <= t-1 (the entry
                    # fills at open[t]), so no lookahead.
                    if fib_entry != FIB_ENTRY_OFF:
                        H = p2[t - 1, s]             # swing high
                        L = p1[t - 1, s]             # leg-start low
                        if not (H > L):
                            continue                 # no valid up-leg = no setup
                        span = H - L
                        z_hi = H - fib_zone_lo * span   # shallow retr (higher px)
                        z_lo = H - fib_zone_hi * span   # deep retr (lower px)
                        if fib_entry == FIB_ENTRY_ZONE:
                            # price is simply sitting in the band
                            pxref = close2d[t - 1, s]
                            if pxref < z_lo or pxref > z_hi:
                                continue
                        else:
                            # BOUNCE / BOUNCE_TREND: the pullback low reached
                            # the band and the deep bound held intrabar, and
                            # the confirmation bar closed UP (the level held
                            # and price is turning) -- not a knife mid-fall.
                            mlow = np.inf
                            u0 = t - fib_bounce_look
                            if u0 < 0:
                                u0 = 0
                            for u in range(u0, t):
                                if valid[u, s] and low2d[u, s] < mlow:
                                    mlow = low2d[u, s]
                            if mlow < z_lo or mlow > z_hi:
                                continue             # didn't tag the band / broke it
                            if close2d[t - 1, s] <= open2d[t - 1, s]:
                                continue             # no up-close confirmation
                    base = open2d[t, s]
                    px = base * (1.0 + cost_side)
                    if px <= 0.0 or not np.isfinite(px):
                        continue
                    alloc = eq_now / max_pos
                    if alloc > cash:
                        alloc = cash
                    if alloc <= 0.0:
                        break
                    si = sidx[t - 1, s]
                    if stop_mode == ATR:
                        stop_px = base - atr_mult * atr[t - 1, s]
                    elif stop_mode == SWING:
                        stop_px = swing[t - 1, s] * (1.0 - swing_buf)
                    elif stop_mode == FIB:
                        # below the fib retracement of the up-leg P1->P2:
                        # P2 - ratio*(P2-P1), then a small buffer under it. A
                        # valid up-leg needs P2>P1; otherwise NaN -> pct fallback.
                        H = p2[t - 1, s]
                        L = p1[t - 1, s]
                        if H > L:
                            stop_px = (H - fib_stop_ratio * (H - L)) \
                                * (1.0 - fib_buf)
                        else:
                            stop_px = np.nan
                    else:
                        stop_px = base * (1.0 - strat_stop[si])
                    # any invalid/degenerate placement (incl. FIB with no
                    # confirmed up-leg yet -> NaN, or P2<=P1) falls back to pct
                    if (not np.isfinite(stop_px)) or stop_px >= base:
                        stop_px = base * (1.0 - strat_stop[si])
                    shares = alloc / px
                    cash -= shares * px
                    held[s] = True
                    p_shares[s] = shares
                    p_entry[s] = px
                    p_stop[s] = stop_px
                    # snapshot the fib anchors so the fib-ladder trail draws
                    # off a fixed fib for the life of the trade.
                    p_p1[s] = p1[t - 1, s]
                    p_p2[s] = p2[t - 1, s]
                    p_p3[s] = p3[t - 1, s]
                    # hard target: fib target = swing high P2 (an absolute
                    # price) when enabled and above entry; else per-strategy %.
                    # (usually left off -- fib levels are better trailed under.)
                    tgt = strat_target[si]
                    if use_fib_target and stop_mode == FIB:
                        H = p2[t - 1, s]
                        p_tp[s] = H if (np.isfinite(H) and H > base) else 0.0
                    else:
                        p_tp[s] = base * (1.0 + tgt) if tgt > 0.0 else 0.0
                    p_hold[s] = strat_hold[si]
                    p_held[s] = 0
                    p_hwm[s] = base
                    p_sidx[s] = si
                    n_open += 1

        # ---- mark equity ---------------------------------------------------
        mkt = 0.0
        for s in range(S):
            if held[s]:
                mkt += p_shares[s] * last_close[s]
        equity[t] = cash + mkt
        invested[t] = mkt / equity[t] if equity[t] > 0.0 else 0.0

    return (r_sym[:ntr], r_ret[:ntr], r_reason[:ntr], r_bars[:ntr],
            r_strat[:ntr], equity, invested, n_open)


def simulate(open2d, high2d, low2d, close2d, valid, ent, score, sidx, ext,
             atr, swing, strat_stop, strat_hold, strat_target,
             stop_mode=PCT, atr_mult=2.5, swing_buf=0.005,
             trail_act=0.0, trail_dist=0.03,
             cost_side=0.0, max_pos=5, init_cash=100000.0,
             p1=None, p2=None, p3=None, fib_stop_ratio=0.786, fib_buf=0.005,
             trail_mode=TRAIL_PCT, use_fib_target=0, fib_ext=(1.0, 1.272, 1.618,
             2.0), gate=None, fib_entry=FIB_ENTRY_OFF, fib_zone_lo=0.5,
             fib_zone_hi=0.786, fib_bounce_look=3):
    """Python wrapper: ensures dtypes/contiguity, runs the njit core.
    Returns dict with sym, ret, reason, bars (per-trade) and equity/invested
    (per-bar). All 2D inputs are (T, S) float64/bool; strat_* are 1D per
    strategy; sidx/ent/score/ext/atr/swing are (T, S). fib_hi/fib_lo are the
    (T, S) confirmed swing-high / swing-low ladders; the engine forms the fib
    stop (P2 - ratio*(P2-P1), buffered) and zone/bounce entry from the up-leg
    P1->P2, the extension trail (under each cleared rung P3 + r*(P2-P1)) from
    all three (default zeros = fib unused). fib_ext is the ladder of extension
    ratios for TRAIL_FIB. gate is the (T, S) trend-gate mask (1/0) the
    BOUNCE_TREND entry mode fires inside."""
    f = lambda a: np.ascontiguousarray(a, np.float64)
    b = lambda a: np.ascontiguousarray(a, np.bool_)
    i = lambda a: np.ascontiguousarray(a, np.int64)
    z = np.zeros_like(f(open2d))
    a1 = z if p1 is None else f(p1)
    a2 = z if p2 is None else f(p2)
    a3 = z if p3 is None else f(p3)
    gt = z if gate is None else f(gate)
    out = _simulate(f(open2d), f(high2d), f(low2d), f(close2d), b(valid),
                    b(ent), f(score), i(sidx), b(ext), f(atr), f(swing),
                    a1, a2, a3, gt,
                    f(strat_stop), i(strat_hold), f(strat_target),
                    int(stop_mode), float(atr_mult), float(swing_buf),
                    float(fib_stop_ratio), float(fib_buf),
                    float(trail_act), float(trail_dist),
                    int(trail_mode), int(use_fib_target),
                    f(np.asarray(fib_ext)),
                    int(fib_entry), float(fib_zone_lo), float(fib_zone_hi),
                    int(fib_bounce_look),
                    float(cost_side), int(max_pos), float(init_cash))
    sym, ret, reason, bars, strat, equity, invested, n_open = out
    return {"sym": sym, "ret": ret, "reason": reason, "bars": bars,
            "strat": strat, "equity": equity, "invested": invested,
            "open_end": int(n_open)}
