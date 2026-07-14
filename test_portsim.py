#!/usr/bin/env python3
"""Trust tests for the unified numba portfolio engine (weisswave.portsim).
Hand-built 2D scenarios with KNOWN answers: entry fills, stop/target fills,
trailing ratchet, no-lookahead, accounting (no money created), slot cap and
score-ranked selection. Run: python test_portsim.py (exit 0 = all pass)."""

import sys

import numpy as np

from weisswave.portsim import (simulate, STOP, TARGET, TIME, TRAIL, FIB,
                               TRAIL_STRUCT, FIB_ENTRY_ZONE, FIB_ENTRY_BOUNCE,
                               FIB_ENTRY_BOUNCE_TREND)

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
r = run(o, h, l, c, ent, stop_mode=FIB, p2=fib_hi, p1=fib_lo,
        fib_stop_ratio=0.5, fib_buf=0.0)
check("FIB stop = H - ratio*(H-L), fills at 9.5",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (9.5/12 - 1)) < 1e-9
      and r["reason"][0] == STOP, f"ret={r['ret']} reason={r['reason']}")

# 8. FIB with no confirmed up-leg (H/L NaN) falls back to the pct stop --------
nan1 = np.full((4, 1), np.nan)
r = run(o, h, l, c, ent, stop_mode=FIB, p2=nan1, p1=nan1, stop=0.10)
# fallback = 12*(1-0.10) = 10.8; bar3 low 9 -> fill 10.8
check("FIB NaN -> pct-stop fallback (10.8)",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (10.8/12 - 1)) < 1e-9
      and r["reason"][0] == STOP, f"ret={r['ret']}")

# 9. fib target = prior pivot high H (absolute price above entry) -------------
o, h, l, c = grid([[(10, 10, 10, 10)], [(11, 11, 11, 11)], [(11, 11, 11, 11)],
                   [(11, 13, 11, 11)]])
ent = [[True], [False], [False], [False]]
fh = np.full((4, 1), np.nan); fh[0, 0] = 13.0    # H > entry 11
r = run(o, h, l, c, ent, stop_mode=FIB, p2=fh, p1=nan1,
        use_fib_target=1)   # fib_lo NaN -> stop falls back (won't trigger)
check("fib target fills at prior pivot high (13)",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (13/11 - 1)) < 1e-9
      and r["reason"][0] == TARGET, f"ret={r['ret']} reason={r['reason']}")

# 10. fib target H BELOW entry is ignored (no backwards target) --------------
fh_lo = np.full((4, 1), np.nan); fh_lo[0, 0] = 10.0    # H below entry 11
r = run(o, h, l, c, ent, stop_mode=FIB, p2=fh_lo, p1=nan1,
        use_fib_target=1, hold=2)
check("fib target H <= entry is dropped (exits by time, not target)",
      len(r["ret"]) == 1 and r["reason"][0] == TIME, f"reason={r['reason']}")

# 11. structure trailing ratchets under rising swing lows (p3) ---------------
o, h, l, c = grid([[(10, 10, 10, 10)], [(10, 10, 10, 10)], [(10, 12, 10, 12)],
                   [(12, 12, 11, 11)], [(11, 11, 10.7, 11)]])
ent = [[True], [False], [False], [False], [False]]
# p3[t-1] is the swing low the stop trails under at bar t; rises 9 -> 10.8
p3s = np.array([[np.nan], [9.0], [9.0], [10.8], [10.8]])
r = run(o, h, l, c, ent, stop=0.20, trail_mode=TRAIL_STRUCT, p3=p3s,
        swing_buf=0.0)
# stop ratchets to p3[3]=10.8 by bar4; low 10.7 <= 10.8 -> exit 10.8 (+8%)
check("structure trail exits under the swing low, locking a gain",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (10.8/10 - 1)) < 1e-9
      and r["reason"][0] == TRAIL, f"ret={r['ret']} reason={r['reason']}")

# 11b. fib-ladder trail ratchets under each cleared extension rung ------------
# P1=10, P2=20, P3=14 (span 10). ext rung 1.0 = 14 + 1.0*10 = 24. Price clears
# 24 (hwm 25), stop -> 24; then falls to hit it -> TRAIL exit at 24 (+140%).
o, h, l, c = grid([[(10, 10, 10, 10)], [(10, 10, 10, 10)], [(10, 25, 10, 24)],
                   [(24, 24, 23, 23.5)], [(23.5, 23.5, 20, 22)]])
ent = [[True], [False], [False], [False], [False]]
P1 = np.full((5, 1), 10.0); P2 = np.full((5, 1), 20.0); P3 = np.full((5, 1), 14.0)
r = run(o, h, l, c, ent, stop=0.50, trail_mode=2, p1=P1, p2=P2, p3=P3,
        fib_ext=[1.0, 1.618], swing_buf=0.0)
# entry 10; hwm hits 25 (>24 rung) -> stop 24; bar4 low 20 <= 24 -> exit 24
check("fib-ladder trail exits under a cleared extension rung (24)",
      len(r["ret"]) == 1 and abs(r["ret"][0] - (24/10 - 1)) < 1e-9
      and r["reason"][0] == TRAIL, f"ret={r['ret']} reason={r['reason']}")


# 12. fib ZONE entry: enter only when prior close is in the retracement band -
# H=20, L=10; zone 0.5-0.786 -> price band [20-0.786*10, 20-0.5*10] = [12.14, 15]
def zbars(pxref):
    return grid([[(pxref, pxref, pxref, pxref)], [(13, 13, 13, 13)],
                 [(13, 13, 13, 13)]])
zent = [[True], [False], [False]]
zh = np.full((3, 1), 20.0); zl = np.full((3, 1), 10.0)


def entered(pxref, mode=FIB_ENTRY_ZONE, hi=zh, lo=zl):
    o, h, l, c = zbars(pxref)
    return len(run(o, h, l, c, zent, hold=1, fib_entry=mode,
                   p2=hi, p1=lo)["ret"])


check("zone entry: in-band pullback (13) enters", entered(13.0) == 1)
check("zone entry: too-shallow pullback (16) is skipped", entered(16.0) == 0)
check("zone entry: too-deep pullback (11) is skipped", entered(11.0) == 0)
check("zone entry: no valid up-leg (NaN H/L) is skipped",
      entered(13.0, hi=np.full((3, 1), np.nan),
              lo=np.full((3, 1), np.nan)) == 0)
check("entry OFF: shallow pullback (16) still enters (mode is the cause)",
      entered(16.0, mode=0) == 1)

# 13. fib BOUNCE entry: pullback tagged the band AND the confirm bar closed up
# H=20, L=10; band [12.14, 15]. Entry at bar3 (ent[2]); look=3 -> bars 0-2,
# so bar2 is the confirm bar and bar4 is left for the exit.
bh = np.full((5, 1), 20.0); bl = np.full((5, 1), 10.0)
bent = [[False], [False], [True], [False], [False]]


def bounce_entered(confirm, ent=bent, mode=FIB_ENTRY_BOUNCE, **kw):
    # bar1 low 13 tags the band; bar2 (confirm) precedes the bar3 entry
    o, h, l, c = grid([[(16, 16, 16, 16)], [(15, 15, 13, 13)], confirm,
                       [(14, 14, 14, 14)], [(14, 14, 14, 14)]])
    return len(run(o, h, l, c, ent, hold=1, fib_entry=mode,
                   p2=bh, p1=bl, **kw)["ret"])


check("bounce entry: tagged band + up-close confirm bar enters",
      bounce_entered([(13, 14.5, 12.5, 14)]) == 1)        # green confirm
check("bounce entry: down-close confirm bar is skipped",
      bounce_entered([(14, 14.5, 12.5, 13)]) == 0)        # red confirm
check("bounce entry: pullback never reached the band is skipped",
      len(run(*grid([[(16, 16, 16, 16)], [(16, 16, 15.6, 16)],
                     [(16, 17, 15.6, 16.5)], [(16, 16, 16, 16)],
                     [(16, 16, 16, 16)]]), bent, hold=1,
              fib_entry=FIB_ENTRY_BOUNCE, p2=bh, p1=bl)["ret"]) == 0)

# 14. BOUNCE_TREND: the bounce is the entry, gated by the TREND not the signal
ones = np.ones((5, 1))
noent = [[False], [False], [False], [False], [False]]
check("bounce-trend: enters with NO strategy signal when trend gate is on",
      bounce_entered([(13, 14.5, 12.5, 14)], ent=noent,
                     mode=FIB_ENTRY_BOUNCE_TREND, gate=ones) == 1)
check("bounce-trend: no entry when the trend gate is off",
      bounce_entered([(13, 14.5, 12.5, 14)], ent=noent,
                     mode=FIB_ENTRY_BOUNCE_TREND, gate=np.zeros((5, 1))) == 0)


# 15. confluence entry: weighted sum of factors >= threshold drives the entry
o, h, l, c = grid([[(10, 10, 10, 10)], [(11, 11, 11, 11)], [(12, 12, 12, 12)]])
cent = [[False], [False], [False]]        # NO strategy signal -> conf drives it
fac = np.zeros((3, 1, 2)); fac[0, 0, 0] = 1.0; fac[0, 0, 1] = 0.5   # at entry bar


def conf_n(weights, thr, **kw):
    return len(run(o, h, l, c, cent, hold=1, conf_entry=1, factors=fac,
                   weights=np.array(weights), conf_threshold=thr, **kw)["ret"])


check("confluence: score>=threshold enters with no strategy signal",
      conf_n([1.0, 1.0], 1.0) == 1)         # 1.0+0.5 = 1.5 >= 1.0
check("confluence: score<threshold is skipped", conf_n([1.0, 1.0], 2.0) == 0)
check("confluence: zero weights -> no entry (weights drive it)",
      conf_n([0.0, 0.0], 0.5) == 0)
check("confluence: a single factor's weight can carry the entry",
      conf_n([0.0, 3.0], 1.0) == 1)         # 3*0.5 = 1.5 >= 1.0


# 16. higher-TF screen: htf factors [htf_start:] gate eligibility separately,
# and do NOT contribute to the entry score (entry uses factors [0:htf_start])
o, h, l, c = grid([[(10, 10, 10, 10)], [(11, 11, 11, 11)], [(12, 12, 12, 12)]])
cent = [[False], [False], [False]]


def screen_n(htf_val, htf_thr, screen=1, thr=1.0):
    fa = np.zeros((3, 1, 3))            # factor 0 = entry, factor 2 = htf
    fa[0, 0, 0] = 1.0
    fa[0, 0, 2] = htf_val
    return len(run(o, h, l, c, cent, hold=1, conf_entry=1, factors=fa,
                   weights=np.array([1.0, 1.0, 1.0]), conf_threshold=thr,
                   htf_start=2, htf_screen=screen, htf_threshold=htf_thr)["ret"])


check("htf screen: setup passes (htf>=thr) -> entry", screen_n(1.0, 0.5) == 1)
check("htf screen: setup fails (htf<thr) -> screened out", screen_n(0.0, 0.5) == 0)
check("htf screen OFF -> enters despite failing setup",
      screen_n(0.0, 0.5, screen=0) == 1)
check("htf factor does NOT leak into the entry score",
      # entry score = f0 only = 1.0; thr 1.5 -> no entry even with htf=1 & screen off
      screen_n(1.0, 0.0, screen=0, thr=1.5) == 0)


if __name__ == "__main__":
    print("\n" + ("ALL PORTSIM TRUST TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
