"""q876 (1598): kappa_1(m) = (offset_d(m) - offset_1(m)) (m + delta)/(d - 1) at d = 1.05, 1.10, at the midpoints of 1D event intervals.
1D states from the event states of chain_d1_long.log (e3-e10), stepped to the midpoint with q868.solve_inner (structure of the interval:
after a birth a centre atom of small weight is added; after a split the centre becomes a pair at small radius); then solved at d = 1 + eps
from the 1D state. Usage: py q876_kappa_d1.py
"""

import re
import numpy as np
from scipy.optimize import brentq
from scipy.special import jv
from q868_lfp_events_dimd import solve_inner, risk, interior_excess

D = 3.34776
arr = lambda s: [float(x) for x in s.replace("[", " ").replace("]", " ").split()]
pat = re.compile(r"CHAIN d 1 e(\d+) (birth|split): m ([0-9.]+); support \[([^\]]*)\]; weights \[([^\]]*)\]")
ev = []
for line in open("q868_runs/chain_d1_long.log", encoding="utf-8"):
    g = pat.search(line)
    if g:
        ev.append((g.group(2), float(g.group(3)), arr(g.group(4)), arr(g.group(5))))


def jz(d):
    return brentq(lambda x: jv(d / 2 - 1, x), 0.5, 3.0)


def offset(d, B, m):
    return 2 * jz(d) / np.sqrt(d - B) - m


def state_after(kind, m, s, w):
    """guess (a, w, w0) just after the event"""
    if kind == "birth":  # pairs + wall, add a centre
        return list(s[:-1]), list(w[:-1]), 0.003
    return [0.3] + list(s[1:-1]), [w[0]] + list(w[1:-1]), None  # split: centre -> small pair


rows = []
for i in range(len(ev) - 1):
    kind, m0, s, w = ev[i]
    m1 = ev[i + 1][1]
    mm = 0.5 * (m0 + m1)
    a, ww, w0 = state_after(kind, m0, s, w)
    ok = True
    for m in np.linspace(m0 + 0.02 if kind == "birth" else m0 + 0.1, mm, 8):
        sol, at, wt, res = solve_inner(1, m, a, ww, w0)
        k = len(a)
        a = list(sol[:k])
        ww = list(sol[k : 2 * k])
        w0 = sol[2 * k] if w0 is not None else None
    if res > 1e-9 or min(at) < 0 or min(wt) < -1e-12:
        print(f"interval {kind} {m0:.4f}-{m1:.4f}: 1D state at {mm:.4f} not valid (residual {res:.1e})")
        continue
    B1 = risk(1, mm, at, wt)
    off1 = np.pi / np.sqrt(1 - B1) - mm
    out = [mm, off1]
    for d in (1.05, 1.10):
        sol, atd, wtd, resd = solve_inner(d, mm, a, ww, w0)
        Bd = risk(d, mm, atd, wtd)
        offd = offset(d, Bd, mm)
        kap = (offd - off1) * (mm + D) / (d - 1)
        ex = interior_excess(d, atd, wtd, mm)
        out += [offd, kap, resd, ex]
    rows.append(out)
    print(
        f"m {mm:.4f}: offset_1 {off1:.6f}; d 1.05: offset {out[2]:.6f} kappa {out[3]:.4f} (res {out[4]:.1e}, excess {out[5]:+.1e}); "
        f"d 1.10: offset {out[6]:.6f} kappa {out[7]:.4f} (res {out[8]:.1e}, excess {out[9]:+.1e})",
        flush=True,
    )

# 1598 readout against the d = 2-5 kappa of 1591 (1)
K = {
    2: [(4.08, 0.2177), (4.982, 0.2449), (5.707, 0.2675), (6.547, 0.2859)],
    3: [(4.389, 0.2215), (5.376, 0.2500), (6.037, 0.2684), (6.946, 0.2877)],
    4: [(4.069, 0.2037), (4.681, 0.2278), (5.754, 0.2564), (6.353, 0.2713), (7.336, 0.2908)],
    5: [(4.403, 0.2152), (4.957, 0.2339), (6.115, 0.2623), (6.658, 0.2745), (7.715, 0.2941)],
}


def kmean(m):
    return np.mean([np.interp(m, [p[0] for p in K[d]], [p[1] for p in K[d]]) for d in K])


sc = [r for r in rows if 4.4 <= r[0] <= 7.3]
okA = all(abs(r[7] - kmean(r[0])) <= 0.02 for r in sc)
okB = all(abs(r[3] - r[7]) <= 0.005 for r in sc)
for r in sc:
    print(
        f"   m {r[0]:.4f}: kappa_1 (d 1.10) {r[7]:.4f}, (d 1.05) {r[3]:.4f}; mean kappa d 2-5 {kmean(r[0]):.4f}; diff {r[7] - kmean(r[0]):+.4f}"
    )
print(
    f"scored {len(sc)}; within 0.02 of d 2-5: {'PASS' if sc and okA else 'FAIL'}; d 1.05 vs 1.10 within 0.005: {'PASS' if sc and okB else 'FAIL'}"
)
