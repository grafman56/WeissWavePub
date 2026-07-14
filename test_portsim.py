#!/usr/bin/env python3
"""Trust tests for the unified numba portfolio engine (weisswave.portsim).
Hand-built 2D scenarios with KNOWN answers: entry fills, stop/target fills,
trailing ratchet, no-lookahead, accounting (no money created), slot cap and
score-ranked selection. Run: python test_portsim.py (exit 0 = all pass)."""

import sys

import numpy as np

from weisswave.portsim import (simulate, STOP, TARGET, TIME, TRAIL, FIB,
                               TRAIL_STRUCT)

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def grid(rows):
    """rows = list of per-bar [ (o,h,l,c) per symbol ]; returns 2D arrays."""
    T = len(rows); S = len(rows[0])
    o = np.array([[r[s][0] for s in range(S)] for r in rows], float)
    h = np.array([[r[s][1] for s in range(S)] for r in rows], float)
    l = np.array([[r[s][2] for s in range(S)] for r in rows], float)
    c = np.array([[r[s][3] for s in range(S)] for r in rows], float)
    return o, h, l, c


def run(o, h, l, c, ent, score=None, sidx=None, ext=None,
        stop=0.5, hold=0, target=0.0, valid=None, **kw):
    T, S = o.shape
    ent = np.asarray(ent, bool)
    score = np.ones((T, S)) if score is None else np.asarray(score, float)
    sidx = np.zeros((T, S), np.int64) if sidx is None else np.asarray(sidx)
    ext = np.zeros((T, S), bool) if ext is None else np.asarray(ext, bool)
    valid = np.ones((T, S), bool) if valid is None else valid
    z = np.zeros((T, S))
    return simulate(o, h, l, c, valid, ent, score, sidx, ext, z, z,
                    np.array([stop]), np.array([hold]), np.array([target]),
                    cost_side=0.0, init_cash=100000.0, **kw)


# 1. entry next-open + time exit + accounting ---------------------------------
o, h, l, c = grid([[(10, 10, 10, 10)], [(11, 11, 11, 11)], [(12, 12, 12, 12)],
                   [(13, 13, 13, 13)], [(14, 14, 14, 14)]])
ent = [[True], [False], [False], [False], [False]]
r = run(o, h, l, c, ent, hold=2, max_pos=1)
# enter open[1]=11, time-exit open[3]=13 -> +18.18%; equity 100000*1.1818
check("entry+time exit return", len(r["ret"]) == 1
      and abs(r["ret"][0] - (13/11 - 1)) < 1e-9, f"ret={r['ret']}")
check("time exit reason", len(r["reason"]) and r["reason"][0] == TIME)
check("accounting (equity matches P&L)",
      abs(r["equity"][-1] - 100000 * (13/11)) < 1e-3, f"eq={r['equity'][-1]:.1f}")

# 2. stop fill ----------------------------------------------------------------
o, h, l, c = grid([[(10, 10, 10, 10)], [(11, 11, 11, 11)], [(11, 11, 11, 11)],
                   [(11, 11, 9, 11)]])
ent = [[True], [False], [False], [False]]
r = run(o, h, l, c, ent, stop=0.10)   # stop 9.9; bar3 low 9 -> fill 9.9
check("stop fills at stop price",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (9.9/11 - 1)) < 1e-9
      and r["reason"][0] == STOP, f"ret={r['ret']}")

# 3. target fill --------------------------------------------------------------
o, h, l, c = grid([[(10, 10, 10, 10)], [(11, 11, 11, 11)], [(11, 11, 11, 11)],
                   [(11, 13, 11, 11)]])
ent = [[True], [False], [False], [False]]
r = run(o, h, l, c, ent, target=0.10)  # tp 12.1; bar3 high 13 -> fill 12.1
check("target fills at target price",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (12.1/11 - 1)) < 1e-9
      and r["reason"][0] == TARGET, f"ret={r['ret']}")

# 4. trailing ratchet locks a gain -------------------------------------------
# rise to +20% (activate at +5%), then drop; trail 5% below hwm
o, h, l, c = grid([[(10, 10, 10, 10)], [(10, 10, 10, 10)], [(10, 12, 10, 12)],
                   [(12, 12, 10.5, 11)], [(11, 11, 10.5, 11)]])
ent = [[True], [False], [False], [False], [False]]
r = run(o, h, l, c, ent, stop=0.20, trail_act=0.05, trail_dist=0.05)
# entry open[1]=10; hwm hits 12 on bar2 (+20%); trail = 12*0.95=11.4;
# bar3 low 10.5 <= 11.4 -> exit ~11.4 (trail), locking a gain
check("trailing exits above entry (locks gain)",
      len(r["ret"]) == 1 and r["ret"][0] > 0 and r["reason"][0] == TRAIL,
      f"ret={r['ret']} reason={r['reason']}")

# 5. no lookahead: bars after exit don't change the trade --------------------
o, h, l, c = grid([[(10, 10, 10, 10)], [(11, 11, 11, 11)], [(11, 11, 9, 11)],
                   [(11, 11, 11, 11)]])
ent = [[True], [False], [False], [False]]
r1 = run(o, h, l, c, ent, stop=0.10)
o2 = o.copy(); o2[3, 0] = 999; h2 = h.copy(); h2[3, 0] = 999
r2 = run(o2, h2, l, c, ent, stop=0.10)
check("no lookahead", len(r1["ret"]) == len(r2["ret"]) == 1
      and abs(r1["ret"][0] - r2["ret"][0]) < 1e-12)

# 6. slot cap + score ranking: 2 symbols, 1 slot, higher score wins ----------
o, h, l, c = grid([[(10, 10, 10, 10), (20, 20, 20, 20)],
                   [(11, 11, 11, 11), (21, 21, 21, 21)],
                   [(12, 12, 12, 12), (22, 22, 22, 22)],
                   [(13, 13, 13, 13), (23, 23, 23, 23)]])
ent = np.array([[True, True], [False, False], [False, False], [False, False]])
score = np.array([[1.0, 2.0]] * 4)     # symbol 1 has the higher score
r = run(o, h, l, c, ent, score=score, hold=2, max_pos=1)
check("slot cap respected (1 position)", len(r["ret"]) == 1)
check("higher score wins the slot", len(r["sym"]) and r["sym"][0] == 1,
      f"winner sym={r['sym']}")
check("max invested never exceeds 100%", r["invested"].max() <= 1.0 + 1e-9,
      f"max_inv={r['invested'].max():.3f}")


# 7. FIB stop = H - ratio*(H-L) below entry, from the H/L ladders ------------
o, h, l, c = grid([[(10, 10, 10, 10)], [(12, 12, 12, 12)], [(12, 12, 12, 12)],
                   [(12, 12, 9, 12)]])
ent = [[True], [False], [False], [False]]
fib_hi = np.full((4, 1), np.nan); fib_hi[0, 0] = 11.0   # H, read at entry
fib_lo = np.full((4, 1), np.nan); fib_lo[0, 0] = 8.0    # L
# entry 12; stop = (11 - 0.5*(11-8))*(1-0) = 9.5; bar3 low 9 -> fill 9.5
r = run(o, h, l, c, ent, stop_mode=FIB, fib_hi=fib_hi, fib_lo=fib_lo,
        fib_stop_ratio=0.5, fib_buf=0.0)
check("FIB stop = H - ratio*(H-L), fills at 9.5",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (9.5/12 - 1)) < 1e-9
      and r["reason"][0] == STOP, f"ret={r['ret']} reason={r['reason']}")

# 8. FIB with no confirmed up-leg (H/L NaN) falls back to the pct stop --------
nan1 = np.full((4, 1), np.nan)
r = run(o, h, l, c, ent, stop_mode=FIB, fib_hi=nan1, fib_lo=nan1, stop=0.10)
# fallback = 12*(1-0.10) = 10.8; bar3 low 9 -> fill 10.8
check("FIB NaN -> pct-stop fallback (10.8)",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (10.8/12 - 1)) < 1e-9
      and r["reason"][0] == STOP, f"ret={r['ret']}")

# 9. fib target = prior pivot high H (absolute price above entry) -------------
o, h, l, c = grid([[(10, 10, 10, 10)], [(11, 11, 11, 11)], [(11, 11, 11, 11)],
                   [(11, 13, 11, 11)]])
ent = [[True], [False], [False], [False]]
fh = np.full((4, 1), np.nan); fh[0, 0] = 13.0    # H > entry 11
r = run(o, h, l, c, ent, stop_mode=FIB, fib_hi=fh, fib_lo=nan1,
        use_fib_target=1)   # fib_lo NaN -> stop falls back (won't trigger)
check("fib target fills at prior pivot high (13)",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (13/11 - 1)) < 1e-9
      and r["reason"][0] == TARGET, f"ret={r['ret']} reason={r['reason']}")

# 10. fib target H BELOW entry is ignored (no backwards target) --------------
fh_lo = np.full((4, 1), np.nan); fh_lo[0, 0] = 10.0    # H below entry 11
r = run(o, h, l, c, ent, stop_mode=FIB, fib_hi=fh_lo, fib_lo=nan1,
        use_fib_target=1, hold=2)
check("fib target H <= entry is dropped (exits by time, not target)",
      len(r["ret"]) == 1 and r["reason"][0] == TIME, f"reason={r['reason']}")

# 11. structure trailing ratchets under rising swing lows (fib_lo) -----------
o, h, l, c = grid([[(10, 10, 10, 10)], [(10, 10, 10, 10)], [(10, 12, 10, 12)],
                   [(12, 12, 11, 11)], [(11, 11, 10.7, 11)]])
ent = [[True], [False], [False], [False], [False]]
# fib_lo[t-1] is the swing low the stop trails under at bar t; rises 9 -> 10.8
flo = np.array([[np.nan], [9.0], [9.0], [10.8], [10.8]])
r = run(o, h, l, c, ent, stop=0.20, trail_mode=TRAIL_STRUCT, fib_lo=flo,
        swing_buf=0.0)
# stop ratchets to fib_lo[3]=10.8 by bar4; low 10.7 <= 10.8 -> exit 10.8 (+8%)
check("structure trail exits under the swing low, locking a gain",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (10.8/10 - 1)) < 1e-9
      and r["reason"][0] == TRAIL, f"ret={r['ret']} reason={r['reason']}")


# 12. fib zone gate: enter only when prior close is in the retracement band --
# H=20, L=10; zone 0.5-0.786 -> price band [20-0.786*10, 20-0.5*10] = [12.14, 15]
def zbars(pxref):
    return grid([[(pxref, pxref, pxref, pxref)], [(13, 13, 13, 13)],
                 [(13, 13, 13, 13)]])
zent = [[True], [False], [False]]
zh = np.full((3, 1), 20.0); zl = np.full((3, 1), 10.0)


def entered(pxref, gate=1, hi=zh, lo=zl):
    o, h, l, c = zbars(pxref)
    return len(run(o, h, l, c, zent, hold=1, fib_zone_gate=gate,
                   fib_hi=hi, fib_lo=lo)["ret"])


check("zone gate: in-band pullback (13) enters", entered(13.0) == 1)
check("zone gate: too-shallow pullback (16) is skipped", entered(16.0) == 0)
check("zone gate: too-deep pullback (11) is skipped", entered(11.0) == 0)
check("zone gate: no valid up-leg (NaN H/L) is skipped",
      entered(13.0, hi=np.full((3, 1), np.nan),
              lo=np.full((3, 1), np.nan)) == 0)
check("zone gate OFF: shallow pullback (16) still enters (gate is the cause)",
      entered(16.0, gate=0) == 1)


if __name__ == "__main__":
    print("\n" + ("ALL PORTSIM TRUST TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
