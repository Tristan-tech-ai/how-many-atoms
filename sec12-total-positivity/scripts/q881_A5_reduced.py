"""q881 (6 Oct 2026): route A5, written before any output was read (registered in amendment 1647).
After q879-q880 (outward flow = positive sum of a positive g' part and a negative g - 1 part, no sign lemma in these coordinates), the masses are
eliminated: the g - 1 and centre equations determine the masses v(p; a), and the position equations g'(p_j) = 0 become a reduced system with
Jacobian S = A - B Dm^-1 C and right-hand side b_eff = b1 - B Dm^-1 b2 (A = dG1/dp, B = dG1/dv, C = dG2/dp, Dm = dG2/dv, G1 = the g' equations,
G2 = the g - 1 and centre equations), so that dp/da = -S^-1 b_eff. Same three rows, same Jacobian recipe as q880 (arb 200 bits, step 1e-30; dF/da step 1e-20).
Criteria (all three rows): (a) every off-diagonal entry of S is >= 0 (-S is a Z-matrix) and every diagonal entry negative; (b) every entry of b_eff is >= 0;
(c) every entry of S^-1 is <= 0. Entries below 1e-12 of the row's largest magnitude count as zero. Consistency: dp/da from the reduced system equals the
flow of q880 to 1e-10 (relative). PASS = (a), (b), (c) and the consistency in all rows.
Usage: py q881_A5_reduced.py"""

import json
import numpy as np
import q579_immune_mp as M
from flint import arb, arb_mat

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
HA = arb("1e-20")
allok = True
for name, i in picks:
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
    Fp = M.IMP(a + HA, 9).F(u, J, centre)
    Fm = M.IMP(a - HA, 9).F(u, J, centre)
    b = [((Fp[k] - Fm[k]) / (2 * HA)).mid() for k in range(n)]
    du = Jm.solve(arb_mat([[-t] for t in b]))
    flow = np.array([float(du[k, 0].mid()) for k in range(J)])
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
    b1 = arb_mat([[b[k]] for k in range(J)])
    b2 = arb_mat([[b[J + k]] for k in range(m)])
    beff = b1 - B * Di * b2
    dpa = -(S.inv() * beff)
    red = np.array([float(dpa[k, 0].mid()) for k in range(J)])
    Sn = np.array([[float(S[p, q].mid()) for q in range(J)] for p in range(J)])
    Si = S.inv()
    Sin = np.array([[float(Si[p, q].mid()) for q in range(J)] for p in range(J)])
    be = np.array([float(beff[k, 0].mid()) for k in range(J)])
    off = Sn - np.diag(np.diag(Sn))
    tolS = 1e-12 * np.abs(Sn).max()
    a_ok = bool((off >= -tolS).all() and (np.diag(Sn) < 0).all())
    b_ok = bool((be >= -1e-12 * np.abs(be).max()).all())
    c_ok = bool((Sin <= 1e-12 * np.abs(Sin).max()).all())
    cons = float(np.max(np.abs(red - flow) / np.abs(flow)))
    ok = a_ok and b_ok and c_ok and cons <= 1e-10
    allok = allok and ok
    print(f"== {name}: a {float(a.mid()):.6f}, K {int(r['K'])}, centre {centre}, J {J}, masses {m}")
    print(
        f"   (a) off-diagonal entries of S: negative {int((off < -tolS).sum())}, positive or zero {int((off >= -tolS).sum()) - J}; diagonal all negative {bool((np.diag(Sn) < 0).all())} -> {'PASS' if a_ok else 'FAIL'}"
    )
    print(
        f"   (b) b_eff entries: min {be.min():+.4e}, max {be.max():+.4e}, negative {int((be < -1e-12 * np.abs(be).max()).sum())} -> {'PASS' if b_ok else 'FAIL'}"
    )
    print(
        f"   (c) S^-1 entries: positive {int((Sin > 1e-12 * np.abs(Sin).max()).sum())} of {J * J}, max {Sin.max():+.3e} -> {'PASS' if c_ok else 'FAIL'}"
    )
    print(
        f"   consistency of the reduced flow with the full flow: max relative difference {cons:.2e}; reduced flow min {red.min():.4f}"
    )
    print(
        "   sign of b_eff:",
        "".join(
            "+" if x > 1e-12 * np.abs(be).max() else "-" if x < -1e-12 * np.abs(be).max() else "0" for x in be
        ),
    )
    print("   sign pattern of -S off-diagonals (rows innermost first; + means S_ij > 0):")
    for p in range(J):
        print(
            "     "
            + "".join(
                "." if p == q else "+" if Sn[p, q] > tolS else "-" if Sn[p, q] < -tolS else "0"
                for q in range(J)
            )
        )
print("A5 VERDICT:", "PASS" if allok else "FAIL")
