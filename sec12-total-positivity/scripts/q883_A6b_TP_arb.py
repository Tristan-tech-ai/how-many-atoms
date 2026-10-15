"""q883 (6 Oct 2026): A6b, the 200-bit confirmation of the total-positivity test of q882, written before any output was read (registered in amendment 1649).
q882 computed its minors in double precision from a matrix stored in double; its docstring promised a higher-precision recomputation for minors near zero and the
code did not do it, and the smallest minors (1e-28 to 1e-53) lie below double resolution. Here S and T = -D S D (D = diag((-1)^i)) are kept as arb balls (200 bits,
the Jacobian recipe of q879-q882, rows quiet, before, after) and every minor is an arb determinant.
Criteria: (1) all minors of T of orders 1, 2 and 3 are certainly positive (the arb ball of the determinant lies above 0); (2) the 20000 sampled minors for each order 4..8
(seed 20261006, the same sample as q882) are certainly positive. A minor whose ball contains 0 counts as undecided and fails the criterion. PASS = (1) and (2) in all rows.
Usage: py q883_A6b_TP_arb.py"""

import itertools, json
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
rng = np.random.default_rng(20261006)
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
    S = A - B * Dm.inv() * C
    T = [
        [(S[p, q] * (-1 if (p + q) % 2 == 0 else 1)).mid() for q in range(J)] for p in range(J)
    ]  # T = -D S D: sign (-1)^(p+q+1) times S
    ok = True
    lines = []

    def minor(rs, cs):
        k = len(rs)
        Mx = arb_mat(k, k)
        for x, ri in enumerate(rs):
            for y, ci in enumerate(cs):
                Mx[x, y] = T[ri][ci]
        return Mx.det()

    for order in (1, 2, 3):
        idx = list(itertools.combinations(range(J), order))
        bad = und = 0
        worst = None
        for rs in idx:
            for cs in idx:
                d = minor(rs, cs)
                if not d > 0:
                    if d < 0:
                        bad += 1
                    else:
                        und += 1
                lo = float(d.mid())
                if worst is None or lo < worst:
                    worst = lo
        lines.append(
            f"   order {order} (all {len(idx) ** 2} minors): smallest midpoint {worst:+.3e}, certainly negative {bad}, undecided {und}"
        )
        ok = ok and bad == 0 and und == 0
    for order in range(4, 9):
        bad = und = 0
        worst = None
        for _ in range(20000):
            rs = sorted(rng.choice(J, order, replace=False))
            cs = sorted(rng.choice(J, order, replace=False))
            d = minor([int(x) for x in rs], [int(x) for x in cs])
            if not d > 0:
                if d < 0:
                    bad += 1
                else:
                    und += 1
            lo = float(d.mid())
            if worst is None or lo < worst:
                worst = lo
        lines.append(
            f"   order {order} (20000 sampled): smallest midpoint {worst:+.3e}, certainly negative {bad}, undecided {und}"
        )
        ok = ok and bad == 0 and und == 0
    allok = allok and ok
    print(f"== {name}: J {J}")
    for l in lines:
        print(l)
    print(f"   A6b -> {'PASS' if ok else 'FAIL'}", flush=True)
print("A6b VERDICT:", "PASS" if allok else "FAIL")
