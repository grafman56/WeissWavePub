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
TRAIL_PCT, TRAIL_STRUCT = 0, 1


@njit(cache=True)
def _simulate(open2d, high2d, low2d, close2d, valid,
              ent, score, sidx, ext,
              atr, swing, fib_hi, fib_lo,
              strat_stop, strat_hold, strat_target,
              stop_mode, atr_mult, swing_buf,
              fib_stop_ratio, fib_buf,
              trail_act, trail_dist, trail_mode, use_fib_target,
              fib_zone_gate, fib_zone_lo, fib_zone_hi,
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
            #   structure -> under the last confirmed swing low (fib_lo), which
            #                fixes winners getting shaken out by a fixed %.
            # structure-trail is on whenever trail_mode==STRUCT; pct-trail
            # needs trail_act>0 (its original gate, unchanged).
            do_trail = (trail_mode == TRAIL_STRUCT) or (trail_act > 0.0)
            if do_trail and p_hwm[s] >= p_entry[s] * (1.0 + trail_act):
                if trail_mode == TRAIL_STRUCT:
                    cand = fib_lo[t - 1, s] * (1.0 - swing_buf)
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
            ncand = 0
            for s in range(S):
                if valid[t, s] and (not held[s]) and ent[t - 1, s]:
                    ncand += 1
            if ncand > 0:
                cs = np.empty(ncand, np.int64)
                sc = np.empty(ncand)
                k = 0
                for s in range(S):
                    if valid[t, s] and (not held[s]) and ent[t - 1, s]:
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
                    # fib zone gate: only enter if price has pulled back into
                    # the [lo, hi] retracement band of the current up-leg L->H
                    # (buying the 0.5-0.786 zone, like the charts). Uses the
                    # confirmed prior close, so no lookahead.
                    if fib_zone_gate:
                        H = fib_hi[t - 1, s]
                        L = fib_lo[t - 1, s]
                        if not (H > L):
                            continue                 # no valid up-leg = no setup
                        span = H - L
                        z_hi = H - fib_zone_lo * span   # shallow retr (higher px)
                        z_lo = H - fib_zone_hi * span   # deep retr (lower px)
                        pxref = close2d[t - 1, s]
                        if pxref < z_lo or pxref > z_hi:
                            continue                 # not in the pullback zone
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
                        # below the fib retracement of the last up-leg L->H:
                        # H - ratio*(H-L), then a small buffer under it. A
                        # valid up-leg needs H>L; otherwise NaN -> pct fallback.
                        H = fib_hi[t - 1, s]
                        L = fib_lo[t - 1, s]
                        if H > L:
                            stop_px = (H - fib_stop_ratio * (H - L)) \
                                * (1.0 - fib_buf)
                        else:
                            stop_px = np.nan
                    else:
                        stop_px = base * (1.0 - strat_stop[si])
                    # any invalid/degenerate placement (incl. FIB with no
                    # confirmed up-leg yet -> NaN, or H<=L) falls back to pct
                    if (not np.isfinite(stop_px)) or stop_px >= base:
                        stop_px = base * (1.0 - strat_stop[si])
                    shares = alloc / px
                    cash -= shares * px
                    held[s] = True
                    p_shares[s] = shares
                    p_entry[s] = px
                    p_stop[s] = stop_px
                    # target: fib target = prior pivot high H (an absolute
                    # price) when enabled and above entry; else per-strategy %.
                    tgt = strat_target[si]
                    if use_fib_target and stop_mode == FIB:
                        H = fib_hi[t - 1, s]
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
             fib_hi=None, fib_lo=None, fib_stop_ratio=0.786, fib_buf=0.005,
             trail_mode=TRAIL_PCT, use_fib_target=0,
             fib_zone_gate=0, fib_zone_lo=0.5, fib_zone_hi=0.786):
    """Python wrapper: ensures dtypes/contiguity, runs the njit core.
    Returns dict with sym, ret, reason, bars (per-trade) and equity/invested
    (per-bar). All 2D inputs are (T, S) float64/bool; strat_* are 1D per
    strategy; sidx/ent/score/ext/atr/swing are (T, S). fib_hi/fib_lo are the
    (T, S) confirmed swing-high / swing-low ladders; the engine forms the fib
    stop (H - ratio*(H-L), buffered), target (H) and structure trail (under L)
    from them at sim time (default zeros = FIB unused)."""
    f = lambda a: np.ascontiguousarray(a, np.float64)
    b = lambda a: np.ascontiguousarray(a, np.bool_)
    i = lambda a: np.ascontiguousarray(a, np.int64)
    z = np.zeros_like(f(open2d))
    fhi = z if fib_hi is None else f(fib_hi)
    flo = z if fib_lo is None else f(fib_lo)
    out = _simulate(f(open2d), f(high2d), f(low2d), f(close2d), b(valid),
                    b(ent), f(score), i(sidx), b(ext), f(atr), f(swing),
                    fhi, flo,
                    f(strat_stop), i(strat_hold), f(strat_target),
                    int(stop_mode), float(atr_mult), float(swing_buf),
                    float(fib_stop_ratio), float(fib_buf),
                    float(trail_act), float(trail_dist),
                    int(trail_mode), int(use_fib_target),
                    int(fib_zone_gate), float(fib_zone_lo), float(fib_zone_hi),
                    float(cost_side), int(max_pos), float(init_cash))
    sym, ret, reason, bars, strat, equity, invested, n_open = out
    return {"sym": sym, "ret": ret, "reason": reason, "bars": bars,
            "strat": strat, "equity": equity, "invested": invested,
            "open_end": int(n_open)}
