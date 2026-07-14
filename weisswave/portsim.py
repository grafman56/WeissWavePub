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
# stop modes
PCT, ATR, SWING = 0, 1, 2


@njit(cache=True)
def _simulate(open2d, high2d, low2d, close2d, valid,
              ent, score, sidx, ext,
              atr, swing,
              strat_stop, strat_hold, strat_target,
              stop_mode, atr_mult, swing_buf,
              trail_act, trail_dist,
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

    cap = T * max_pos + S + 16
    r_sym = np.empty(cap, np.int64)
    r_ret = np.empty(cap)
    r_reason = np.empty(cap, np.int64)
    r_bars = np.empty(cap, np.int64)
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
            # trailing: ratchet stop up under the high-water mark
            if trail_act > 0.0 and p_hwm[s] >= p_entry[s] * (1.0 + trail_act):
                cand = p_hwm[s] * (1.0 - trail_dist)
                if cand > p_stop[s]:
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
                    else:
                        stop_px = base * (1.0 - strat_stop[si])
                    if (not np.isfinite(stop_px)) or stop_px >= base:
                        stop_px = base * (1.0 - strat_stop[si])
                    shares = alloc / px
                    cash -= shares * px
                    held[s] = True
                    p_shares[s] = shares
                    p_entry[s] = px
                    p_stop[s] = stop_px
                    tgt = strat_target[si]
                    p_tp[s] = base * (1.0 + tgt) if tgt > 0.0 else 0.0
                    p_hold[s] = strat_hold[si]
                    p_held[s] = 0
                    p_hwm[s] = base
                    n_open += 1

        # ---- mark equity ---------------------------------------------------
        mkt = 0.0
        for s in range(S):
            if held[s]:
                mkt += p_shares[s] * last_close[s]
        equity[t] = cash + mkt
        invested[t] = mkt / equity[t] if equity[t] > 0.0 else 0.0

    return (r_sym[:ntr], r_ret[:ntr], r_reason[:ntr], r_bars[:ntr],
            equity, invested)


def simulate(open2d, high2d, low2d, close2d, valid, ent, score, sidx, ext,
             atr, swing, strat_stop, strat_hold, strat_target,
             stop_mode=PCT, atr_mult=2.5, swing_buf=0.005,
             trail_act=0.0, trail_dist=0.03,
             cost_side=0.0, max_pos=5, init_cash=100000.0):
    """Python wrapper: ensures dtypes/contiguity, runs the njit core.
    Returns dict with sym, ret, reason, bars (per-trade) and equity/invested
    (per-bar). All 2D inputs are (T, S) float64/bool; strat_* are 1D per
    strategy; sidx/ent/score/ext/atr/swing are (T, S)."""
    f = lambda a: np.ascontiguousarray(a, np.float64)
    b = lambda a: np.ascontiguousarray(a, np.bool_)
    i = lambda a: np.ascontiguousarray(a, np.int64)
    out = _simulate(f(open2d), f(high2d), f(low2d), f(close2d), b(valid),
                    b(ent), f(score), i(sidx), b(ext), f(atr), f(swing),
                    f(strat_stop), i(strat_hold), f(strat_target),
                    int(stop_mode), float(atr_mult), float(swing_buf),
                    float(trail_act), float(trail_dist),
                    float(cost_side), int(max_pos), float(init_cash))
    sym, ret, reason, bars, equity, invested = out
    return {"sym": sym, "ret": ret, "reason": reason, "bars": bars,
            "equity": equity, "invested": invested}
