"""q980 (7 Oct 2026): READING (no verdict) of the extension X = [T | v] on the immune rows of q881/q883 (three rows of results_q584.json: quiet, before, after; J = 15 positions in K = 30 and 31).
Same construction as q881 (arb 200 bits, Jacobian by central differences with step 1e-30, dF/da with step 1e-20), T = -E S E, v = sigma E b_eff with sigma = +1 and -1; the minors are computed at 800 bits from the midpoints.
Prints per row: the signs of E b_eff, the number of initial minors of T and of X for both orientations that are certainly positive / negative / undecided, and the relative size of the finite-difference step errors (a second run of b with the step 1e-24).
Usage: py q980_immune_extension_reading.py"""

import json, sys
from flint import arb, arb_mat, ctx
import q579_immune_mp as M
import q973_extended_tp_test_hi as Q
import q974_tp_region_reading as R

rows = json.load(open("results_q584.json"))["rows"]
K = [int(r["K"]) for r in rows]
k30 = [i for i, k in enumerate(K) if k == 30]
k31 = [i for i, k in enumerate(K) if k == 31]
picks = [
    ("quiet", min(k30, key=lambda i: abs(float(rows[i]["a"]) - 24.10))),
    ("before", max(k30)),
    ("after", min(k31)),
]
HJ = arb("1e-30")
for name, i in picks:
    ctx.prec = 200
    r = rows[i]
    a = arb(float(r["a"]))
    centre = bool(r["centre"])
    I = M.IMP(a, 9)
    pp, vv, res = I.polish([float(t) for t in r["p"]], [float(t) for t in r["v"]], centre)
    J = len(pp)
    u = list(pp) + (list(vv) if centre else list(vv[:-1]))
    n = len(u)
    m = n - J
    Jm = arb_mat(n, n)
    for k in range(n):
        up = list(u)
        um = list(u)
        up[k] += HJ
        um[k] -= HJ
        Fp = I.F(up, J, centre)
        Fm = I.F(um, J, centre)
        for q in range(n):
            Jm[q, k] = ((Fp[q] - Fm[q]) / (2 * HJ)).mid()
    bs = {}
    for lab, HA in (("1e-20", arb("1e-20")), ("1e-24", arb("1e-24"))):
        Fp = M.IMP(a + HA, 9).F(u, J, centre)
        Fm = M.IMP(a - HA, 9).F(u, J, centre)
        bs[lab] = [((Fp[k] - Fm[k]) / (2 * HA)).mid() for k in range(n)]
    A = arb_mat(J, J)
    B = arb_mat(J, m)
    C = arb_mat(m, J)
    Dm = arb_mat(m, m)
    for p in range(J):
        for q in range(J):
            A[p, q] = Jm[p, q]
        for q in range(m):
            B[p, q] = Jm[p, J + q]
    for p in range(m):
        for q in range(J):
            C[p, q] = Jm[J + p, q]
        for q in range(m):
            Dm[p, q] = Jm[J + p, J + q]
    Di = Dm.inv()
    S = A - B * Di * C

    def beff_of(b):
        b1 = arb_mat([[b[k]] for k in range(J)])
        b2 = arb_mat([[b[J + k]] for k in range(m)])
        return b1 - B * Di * b2

    be = beff_of(bs["1e-20"])
    be2 = beff_of(bs["1e-24"])
    dev = max(abs(((be[k, 0] - be2[k, 0]) / be[k, 0]).mid()) for k in range(J))
    sgn = "".join("+" if ((be[k, 0].mid() * (1 if k % 2 == 0 else -1)) > 0) else "-" for k in range(J))
    ctx.prec = 800
    Tm = [[S[p, q].mid() * (-1 if (p + q) % 2 == 0 else 1) for q in range(J)] for p in range(J)]
    print(
        f"== {name}: a {float(a.mid()):.6f}, K {int(r['K'])}, centre {centre}, J {J}, masses {m}; relative difference of b_eff between the steps 1e-20 and 1e-24: {float(dev.mid()):.2e}; signs of E b_eff: {sgn}",
        flush=True,
    )
    pT = R.initial_counts(Tm, J, J)
    print(
        f"   T: initial minors positive {pT[0]}, negative {pT[1]}, undecided {pT[2]} of {J * J}", flush=True
    )
    for sig in (1, -1):
        vv_ = [sig * be[p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(J)]
        X = [list(Tm[p]) + [vv_[p]] for p in range(J)]
        pX = R.initial_counts(X, J, J + 1)
        print(
            f"   X with sigma = {sig:+d}: initial minors positive {pX[0]}, negative {pX[1]}, undecided {pX[2]} of {J * (J + 1)}",
            flush=True,
        )
print("END q980 READING (no verdict)")
