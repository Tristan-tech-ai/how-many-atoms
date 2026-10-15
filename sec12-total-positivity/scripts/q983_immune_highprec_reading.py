"""q983 (7 Oct 2026): READING (no verdict) that diagnoses the FAIL of T80 on the immune rows 500 and 580 (X2: 1 and 2 initial minors not certainly positive at the 200-bit polish with residual 3.8e-47 and 2.4e-47).
The same construction as q981, but after IMP is built the precision is raised to 600 bits, the row is re-polished with the tolerance 1e-110, the Jacobian is taken by central differences with the step 1e-80 and dF/da with the step 1e-80, and the minors are computed at 1200 bits.
Prints the polish residual and the number of initial minors of X (sigma = (-1)^(J+1)) that are not certainly positive. Usage: py q983_immune_highprec_reading.py ROW [ROW ...]
"""

import json, sys
from flint import arb, arb_mat, ctx
import q579_immune_mp as M
import q973_extended_tp_test_hi as Q

rows = json.load(open("results_q584.json"))["rows"]
for i in [int(x) for x in sys.argv[1:]]:
    ctx.prec = 200
    r = rows[i]
    a = arb(float(r["a"]))
    centre = bool(r["centre"])
    I = M.IMP(a, 9)
    ctx.prec = 600
    pp, vv, res = I.polish(
        [float(t) for t in r["p"]], [float(t) for t in r["v"]], centre, tol=1e-110, itmax=40
    )
    rr = abs(float(res.mid())) if hasattr(res, "mid") else abs(float(res))
    J = len(pp)
    u = list(pp) + (list(vv) if centre else list(vv[:-1]))
    n = len(u)
    m = n - J
    HJ = arb("1e-80")
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
    HA = arb("1e-80")
    Fp = M.IMP(a + HA, 9).F(u, J, centre)
    Fm = M.IMP(a - HA, 9).F(u, J, centre)
    b = [((Fp[k] - Fm[k]) / (2 * HA)).mid() for k in range(n)]
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
    be = b1 - B * Di * b2
    ctx.prec = 1200
    sg = 1 if J % 2 == 1 else -1
    Tm = [[S[p, q].mid() * (-1 if (p + q) % 2 == 0 else 1) for q in range(J)] for p in range(J)]
    vv_ = [sg * be[p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(J)]
    g = Q.gates(Tm, vv_, J)
    print(
        f"row {i}: a {float(a.mid()):.6f}, K {int(r['K'])}, J {J}, polish residual at 600 bits {rr:.2e}: X1 {g['X1']}, X2 {g['X2']} ({g['bad']} not certainly positive of {g['count']}), X3 {g['X3']}",
        flush=True,
    )
print("END q983 READING (no verdict)")
