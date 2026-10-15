"""q880 (6 Oct 2026): follow-up of q879 (amendments 1645-1646), written before any output was read.
S1' (control that does not depend on the spacing of the stored rows): for each of the three rows of q879, re-polish in arb at a - d and a + d
(d = 1e-4, Newton from the first-order predictor u -+ d du/da), and compare the central quotient (p(a+d) - p(a-d))/(2d) with du_p/da of the implicit
function theorem. PASS: relative error <= 1e-4 for every j in all three rows.
Decomposition (exploratory, no criterion): du/da = -J^-1 b is split into the part from the g' equations and the part from the g - 1 (and centre)
equations; the signs of b by equation group, the sign pattern of the p-rows of J^-1 for the second group, and the two partial sums per j are printed.
Usage: py q880_A4b_control_decomp.py"""

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
D = arb("1e-4")


def jacobian(I, u, J, centre):
    n = len(u)
    Jm = arb_mat(n, n)
    for k in range(n):
        up = list(u)
        um = list(u)
        up[k] += HJ
        um[k] -= HJ
        Fp = I.F(up, J, centre)
        Fm = I.F(um, J, centre)
        for i in range(n):
            Jm[i, k] = ((Fp[i] - Fm[i]) / (2 * HJ)).mid()
    return Jm


def to_u(pp, vv, centre):
    return list(pp) + (list(vv) if centre else list(vv[:-1]))


def sg(x, tol=1e-12):
    return "+" if x > tol else "-" if x < -tol else "0"


for name, i in picks:
    r = rows[i]
    a = arb(float(r["a"]))
    centre = bool(r["centre"])
    I = M.IMP(a, 9)
    pp, vv, res = I.polish([float(t) for t in r["p"]], [float(t) for t in r["v"]], centre)
    J = len(pp)
    u = to_u(pp, vv, centre)
    n = len(u)
    Jm = jacobian(I, u, J, centre)
    Fp = M.IMP(a + HA, 9).F(u, J, centre)
    Fm = M.IMP(a - HA, 9).F(u, J, centre)
    bv = [((Fp[k] - Fm[k]) / (2 * HA)).mid() for k in range(n)]
    du = Jm.solve(arb_mat([[-t] for t in bv]))
    dp = np.array([float(du[k, 0].mid()) for k in range(J)])
    # S1': independent polishes at a -+ d
    pq = []
    for sgn in (+1, -1):
        a2 = a + sgn * D
        I2 = M.IMP(a2, 9)
        p0 = [pp[k] + sgn * D * du[k, 0].mid() for k in range(J)]
        v0 = [vv[k] for k in range(len(vv))]
        # masses: predictor from du (unknown masses follow the same ordering as in u)
        um = [u[k] + sgn * D * du[k, 0].mid() for k in range(n)]
        v_new = um[J : 2 * J] if centre else um[J : 2 * J - 1] + [arb(1) - sum(um[J : 2 * J - 1])]
        pn, vn, rn = I2.polish([float(t.mid()) for t in p0], [float(t.mid()) for t in v_new], centre)
        pq.append([pn[k] for k in range(J)])
    cq = np.array([float(((pq[0][k] - pq[1][k]) / (2 * D)).mid()) for k in range(J)])
    rel = np.abs(dp - cq) / np.abs(cq)
    ok = bool(rel.max() <= 1e-4)
    print(f"== {name}: a {float(a.mid()):.6f}, K {int(r['K'])}, centre {centre}")
    print(
        f"   S1' central quotient of two polishes at a -+ 1e-4 vs du_p/da: max relative error {rel.max():.2e} -> {'PASS' if ok else 'FAIL'}"
    )
    # decomposition
    Ji = Jm.inv()
    ng = J  # first J equations: g'(p_j)/C
    part1 = np.array([-sum(float((Ji[k, m] * bv[m]).mid()) for m in range(ng)) for k in range(J)])
    part2 = np.array([-sum(float((Ji[k, m] * bv[m]).mid()) for m in range(ng, n)) for k in range(J)])
    print(
        "   sign of b by equation, g' group:",
        "".join(sg(float(bv[m].mid())) for m in range(ng)),
        "| g-1 and centre group:",
        "".join(sg(float(bv[m].mid())) for m in range(ng, n)),
    )
    print("   p-rows of J^-1 against the g-1 / centre equations (rows p_1..p_J innermost first):")
    for k in range(J):
        print(
            "     "
            + "".join(
                sg(float(Ji[k, m].mid()), 1e-12 * max(abs(float(Ji[0, mm].mid())) for mm in range(n)))
                for m in range(ng, n)
            )
        )
    print(
        f"   part from the g' equations: min {part1.min():+.4f} max {part1.max():+.4f}, all positive {bool((part1 > 0).all())}"
    )
    print(
        f"   part from the g-1 equations: min {part2.min():+.4f} max {part2.max():+.4f}, all positive {bool((part2 > 0).all())}"
    )
    print(f"   sum min {(part1 + part2).min():.4f} (compare du_p/da min {dp.min():.4f})")
