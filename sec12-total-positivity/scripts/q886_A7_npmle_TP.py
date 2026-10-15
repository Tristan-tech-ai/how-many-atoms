"""q886 (6 Oct 2026): A7, the structure found for the immune functional (amendments 1645-1650) tested on the NPMLE, written before any output was read
(registered in amendment 1652). The NPMLE functional is the one of q601 (uniform data on [-a, a]; the kernel sum of IMP.gs is replaced by W = Q/(2 Ptilde), C = sum Q;
the function below is copied as text from q601_npmle_mpcont.py). Rows of results_q601.json (200-bit, K 24-34): quiet = the K = 26 row nearest a = 21.80; before = the last
K = 26 row (just before the centre appearance at 22.10536); after = the first K = 27 row (centre atom present).
Computation as in q879-q883: re-polish in arb from the stored doubles, Jacobian by central differences (step 1e-30), dF/da by a central difference in a (step 1e-20), du/da = -J^-1 dF/da,
masses eliminated for the reduced matrix S = A - B Dm^-1 C and b_eff = b1 - B Dm^-1 b2, T = -D S D with D = diag((-1)^i).
Criteria (all must hold in all three rows): (1) control S1': the relative error of du_p/da against the central quotient of two fresh polishes at a -+ 1e-4 is at most 1e-4 for
every j; (2) every component of du_p/da is positive and the smallest exceeds 0.1; (3) every entry of the p-block of J^-1 (rows positions, columns the g' equations) is negative;
(4) S_pq has the sign -(-1)^(p+q) for every p, q and b_eff_p has the sign (-1)^p (entries below 1e-12 of the largest count as zero); (5) every minor of T of orders 1, 2, 3 and the
20000 sampled minors of each order 4..8 (seed 20261006) are certainly positive as arb determinants. PASS = (1)-(5) in all rows.
Declared prior: PASS (the three functionals share the bulk law and the centre function). FAIL records the criterion and the row.
Usage: py q886_A7_npmle_TP.py"""

import itertools, json
import numpy as np
import q579_immune_mp as M
from flint import arb, arb_mat
from flint import arb as _arb


def _gs_npmle(self, p, v, c, ys):
    P = self.P(p, v, c)
    Cn = sum(self.Q)
    W = [q / (2 * Pi) for q, Pi in zip(self.Q, P)]
    out = []
    for y in ys:
        g0 = g1 = g2 = _arb(0)
        for x, Wi in zip(self.x, W):
            dm, dp = x - y, x + y
            em, ep = M.f(dm), M.f(dp)
            g0 += Wi * (em + ep)
            g1 += Wi * (dm * em - dp * ep)
            g2 += Wi * ((dm * dm - 1) * em + (dp * dp - 1) * ep)
        out.append((g0 / Cn, g1 / Cn, g2 / Cn))
    return out, Cn


M.IMP.gs = _gs_npmle
rows = json.load(open("results_q601.json"))["rows"]
K = [int(r["K"]) for r in rows]
k26 = [i for i, k in enumerate(K) if k == 26]
k27 = [i for i, k in enumerate(K) if k == 27]
picks = [
    ("quiet", min(k26, key=lambda i: abs(float(rows[i]["a"]) - 21.80))),
    ("before", max(k26)),
    ("after", min(k27)),
]
HJ = arb("1e-30")
HA = arb("1e-20")
D1 = arb("1e-4")
rng = np.random.default_rng(20261006)
sg = lambda x, tol: "+" if x > tol else "-" if x < -tol else "0"


def build(I, u, J, centre):
    n = len(u)
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
    return Jm


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
    Jm = build(I, u, J, centre)
    Fp = M.IMP(a + HA, 9).F(u, J, centre)
    Fm = M.IMP(a - HA, 9).F(u, J, centre)
    b = [((Fp[k] - Fm[k]) / (2 * HA)).mid() for k in range(n)]
    du = Jm.solve(arb_mat([[-t] for t in b]))
    dp = np.array([float(du[k, 0].mid()) for k in range(J)])
    cq = []
    for sgn in (+1, -1):
        I2 = M.IMP(a + sgn * D1, 9)
        um = [u[k] + sgn * D1 * du[k, 0].mid() for k in range(n)]
        v_new = um[J : 2 * J] if centre else um[J : 2 * J - 1] + [arb(1) - sum(um[J : 2 * J - 1])]
        pn, vn, rn = I2.polish([float(t.mid()) for t in um[:J]], [float(t.mid()) for t in v_new], centre)
        cq.append(pn)
    cqv = np.array([float(((cq[0][k] - cq[1][k]) / (2 * D1)).mid()) for k in range(J)])
    c1 = float(np.max(np.abs(dp - cqv) / np.abs(cqv))) <= 1e-4
    c2 = bool((dp > 0).all() and dp.min() > 0.1)
    Ji = Jm.inv()
    Mpp = np.array([[float(Ji[p, q].mid()) for q in range(J)] for p in range(J)])
    c3 = bool((Mpp < 0).all())
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
    Sn = np.array([[float(S[p, q].mid()) for q in range(J)] for p in range(J)])
    be = np.array([float(beff[p, 0].mid()) for p in range(J)])
    tolS = 1e-12 * np.abs(Sn).max()
    tolb = 1e-12 * np.abs(be).max()
    c4 = all(
        sg(Sn[p, q], tolS) == ("-" if (p + q) % 2 == 0 else "+") for p in range(J) for q in range(J)
    ) and all(sg(be[p], tolb) == ("+" if p % 2 == 0 else "-") for p in range(J))
    T = [[(S[p, q] * (-1 if (p + q) % 2 == 0 else 1)).mid() for q in range(J)] for p in range(J)]

    def minor(rs, cs):
        k = len(rs)
        Mx = arb_mat(k, k)
        for x, ri in enumerate(rs):
            for y, ci in enumerate(cs):
                Mx[x, y] = T[ri][ci]
        return Mx.det()

    bad = und = 0
    worst = {}
    for order in (1, 2, 3):
        idx = list(itertools.combinations(range(J), order))
        w = None
        for rs in idx:
            for cs in idx:
                d = minor(rs, cs)
                if not d > 0:
                    if d < 0:
                        bad += 1
                    else:
                        und += 1
                x = float(d.mid())
                w = x if w is None or x < w else w
        worst[order] = w
    for order in range(4, 9):
        w = None
        for _ in range(20000):
            rs = [int(t) for t in sorted(rng.choice(J, order, replace=False))]
            cs = [int(t) for t in sorted(rng.choice(J, order, replace=False))]
            d = minor(rs, cs)
            if not d > 0:
                if d < 0:
                    bad += 1
                else:
                    und += 1
            x = float(d.mid())
            w = x if w is None or x < w else w
        worst[order] = w
    c5 = bad == 0 and und == 0
    ok = c1 and c2 and c3 and c4 and c5
    allok = allok and ok
    print(f"== {name}: row {i}, a {float(a.mid()):.6f}, K {int(r['K'])}, centre {centre}, J {J}", flush=True)
    print(
        f"   (1) S1' control {'PASS' if c1 else 'FAIL'}: max relative error {float(np.max(np.abs(dp - cqv) / np.abs(cqv))):.2e}"
    )
    print(
        f"   (2) flow: smallest {dp.min():.4f}, largest {dp.max():.4f}, all positive {bool((dp > 0).all())} -> {'PASS' if c2 else 'FAIL'}"
    )
    print(
        f"   (3) p-block of J^-1: negative entries {int((Mpp < 0).sum())} of {J * J} -> {'PASS' if c3 else 'FAIL'}"
    )
    print(
        f"   (4) checkerboard signs of S and b_eff -> {'PASS' if c4 else 'FAIL'}; b_eff signs {''.join(sg(x, tolb) for x in be)}"
    )
    print(
        f"   (5) minors certainly positive: certainly negative {bad}, undecided {und}; smallest midpoints by order {{{', '.join(f'{k}: {v:.2e}' for k, v in worst.items())}}} -> {'PASS' if c5 else 'FAIL'}"
    )
print("A7 VERDICT:", "PASS" if allok else "FAIL")
