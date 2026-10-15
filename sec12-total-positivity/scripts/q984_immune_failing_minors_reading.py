"""q984 (7 Oct 2026): READING (no verdict) on the immune rows 500 and 580 where T80 had X2 False (1 and 2 initial minors not certainly positive).
Same construction as q981 (200-bit polish, central differences 1e-30 / 1e-20, minors at 800 bits). Lists the initial minors that are not certainly positive with their sign status (certainly negative or undecided), log10 of the absolute value of the midpoint,
the cancellation ratio sum|v_p C_p| / |minor| when the minor contains the column v (v enters linearly), and the sign of the minor when v is multiplied entrywise by 1 + delta r_p (r_p uniform in [-1, 1], three seeds) for delta = 1e-60, 1e-45, 1e-35.
Usage: py q984_immune_failing_minors_reading.py ROW [ROW ...]"""

import json, sys, random
from flint import arb, arb_mat, ctx
import q579_immune_mp as M
import q973_extended_tp_test_hi as Q

rows = json.load(open("results_q584.json"))["rows"]
HJ = arb("1e-30")
lg = lambda d: float((abs(d).log() / arb(10).log()).mid().str(6, radius=False)) if d != 0 else float("-inf")
for i in [int(x) for x in sys.argv[1:]]:
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
    HA = arb("1e-20")
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
    ctx.prec = 800
    sg = 1 if J % 2 == 1 else -1
    Tm = [[S[p, q].mid() * (-1 if (p + q) % 2 == 0 else 1) for q in range(J)] for p in range(J)]
    vv_ = [sg * be[p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(J)]
    X = [list(Tm[p]) + [vv_[p]] for p in range(J)]
    n_ = J + 1
    print(
        f"row {i}: a {float(a.mid()):.6f}, K {int(r['K'])}, J {J}, polish residual {abs(float(res.mid())) if hasattr(res, 'mid') else abs(float(res)):.2e}",
        flush=True,
    )
    wins = []
    for k in range(1, J + 1):
        wins += [
            (k, "rows 0..k-1, columns from", j, range(0, k), range(j, j + k)) for j in range(0, n_ - k + 1)
        ] + [
            (k, "rows from, columns 0..k-1", ii, range(ii, ii + k), range(0, k)) for ii in range(1, J - k + 1)
        ]
    for k, kind, idx, rs, cs in wins:
        d = Q.minor(X, rs, cs)
        if d > 0:
            continue
        status = "certainly negative" if d < 0 else "undecided"
        line = f"   order {k}, {kind} {idx}: {status}, log10|minor| {lg(d.mid()):.2f}"
        if J in cs:
            terms = []
            for p in rs:
                sub = (
                    arb_mat(k - 1, k - 1, [X[r_][c_] for r_ in rs if r_ != p for c_ in cs if c_ != J]).det()
                    if k > 1
                    else arb(1)
                )
                terms.append(vv_[p] * sub * ((-1) ** (list(rs).index(p) + k - 1)))
            tot = sum(terms, arb(0))
            canc = (
                float((sum((abs(t) for t in terms), arb(0)) / abs(tot)).mid().str(6, radius=False))
                if tot != 0
                else float("inf")
            )
            line += f", cancellation ratio {canc:.2e}"
            for delta in ("1e-60", "1e-45", "1e-35"):
                sg_ = []
                for seed in (1, 2, 3):
                    rng = random.Random(seed * 100 + k)
                    v2 = [vv_[p] * (1 + arb(delta) * arb(rng.uniform(-1, 1))) for p in range(J)]
                    X2 = [list(Tm[p]) + [v2[p]] for p in range(J)]
                    d2 = Q.minor(X2, rs, cs)
                    sg_.append("+" if d2 > 0 else ("-" if d2 < 0 else "?"))
                line += f"; delta {delta}: {''.join(sg_)}"
        print(line, flush=True)
print("END q984 READING (no verdict)")
