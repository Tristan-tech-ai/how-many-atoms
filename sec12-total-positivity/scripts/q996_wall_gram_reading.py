"""q996 (7 Oct 2026): READING (no verdict): the wall atom as one more position in the Gram-type part (hypothesis of 1751: b = dF/dA is the column of the wall position).
For a polished capacity state (600-bit polish to 1e-120, then 600-bit functions): psi_p = dQ/du_p (p = 1..m) and psi_A = dQ/dA = we [(y - A) phi(y - A) - (y + A) phi(y + A)], the mass directions chi_k = phi_k^s - phi_A^s (and chi_0 = phi(y) - phi_A^s/2 for odd K), the weight 1/Q and the quadrature step H of q920;
g = psi - P psi (projection orthogonal to span{chi}); N = Gram matrix of (g_1..g_m, g_A) of size (m + 1). Prints: (1) the largest relative spread along the first m rows of S_G,pq/N_pq (S_G of q975) as in q976, and the row scale c_p;
(2) the ratio b_eff,p / N_pA divided by c_p for p = 1..m (a constant, 1 or -1, if the wall column is the column of N); (3) the number of initial minors of E' N E' (E' = diag((-1)^j), j = 0..m) at 1200 bits that are certainly positive / negative / undecided, for the (m + 1) x (m + 1) matrix and for its first m rows (m x (m + 1)).
Usage: py q996_wall_gram_reading.py STATE.json (from temporal_within/)"""

import sys, os
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q
import q974_tp_region_reading as R


def main():
    import q920_cap_flint as F
    import q921_cap_tools as T

    path = sys.argv[1]
    ctx.prec = Q.POLPREC
    F.L2P = (arb.pi() * 2).log() / 2
    A_str, u, w, we, wc, d = F.load_state(path)
    m = len(u)
    odd = wc is not None
    cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
    nn = cap.nn
    v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
    v, res, ok = T.newton_full(cap, v, lambda *a: None, tol=arb("1e-120"), maxit=8)
    if not ok:
        print("VOID q996: polish failed")
        sys.exit(2)
    res, J, b = cap.res_jac(v, True, True)
    uu = v[:m]
    ww = v[m : 2 * m]
    we_ = v[2 * m]
    wc_ = v[2 * m + 1] if odd else None
    rows, (SA, LA, PA), (SmA, LmA, PmA), cen, Qn, lQ = cap.tables(uu, ww, we_, wc_)
    N = cap.N
    dpp = [cap.Dpp(rows[j][0], rows[j][1], rows[j][2], lQ) for j in range(m)]
    RR = list(range(m)) + list(range(2 * m, nn))
    Mc = list(range(m, nn))
    r_ = len(RR)
    As = arb_mat(m, m, [J[m + p, q] for p in range(m) for q in range(m)])
    Bs = arb_mat(m, r_, [J[m + p, c] for p in range(m) for c in Mc])
    Cs = arb_mat(r_, m, [J[r, q] for r in RR for q in range(m)])
    Dm = arb_mat(r_, r_, [J[r, c] for r in RR for c in Mc])
    SG = arb_mat(
        m, m, [As[p, q] - (dpp[p] if p == q else 0) for p in range(m) for q in range(m)]
    ) - Bs * Dm.solve(Cs)
    br = arb_mat(r_, 1, [b[r] for r in RR])
    bu = arb_mat(m, 1, [b[m + p] for p in range(m)])
    beff = bu - Bs * Dm.solve(br)
    Hh = arb(float(F.G.H))
    invQ = [1 / q for q in Qn]
    psi = [
        [ww[p] * (rows[p][2][i] * rows[p][0][i] - rows[p][5][i] * rows[p][3][i]) for i in range(N)]
        for p in range(m)
    ]
    psi.append([we_ * (PA[i] * SA[i] - PmA[i] * SmA[i]) for i in range(N)])
    phA = [PA[i] + PmA[i] for i in range(N)]
    chi = [[rows[k][2][i] + rows[k][5][i] - phA[i] for i in range(N)] for k in range(m)]
    if odd:
        chi.append([cen[2][i] - phA[i] / 2 for i in range(N)])
    nv = len(chi)

    def ip(f, g):
        return Hh * sum((f[i] * g[i] * invQ[i] for i in range(N)), arb(0))

    MV = arb_mat(nv, nv, [ip(chi[a], chi[c]) for a in range(nv) for c in range(nv)])
    Cp = arb_mat(m + 1, nv, [ip(psi[p], chi[c]) for p in range(m + 1) for c in range(nv)])
    coef = MV.solve(Cp.transpose()).transpose()
    g = [
        [psi[p][i] - sum((coef[p, c] * chi[c][i] for c in range(nv)), arb(0)) for i in range(N)]
        for p in range(m + 1)
    ]
    Nm = arb_mat(m + 1, m + 1, [ip(g[p], g[q]) for p in range(m + 1) for q in range(m + 1)])
    dev = 0.0
    cp = []
    for p in range(m):
        rat = [SG[p, q] / Nm[p, q] for q in range(m)]
        cp.append(rat[p])
        dev = (
            max(
                dev,
                float(
                    (
                        (max(float(t.mid()) for t in rat) - min(float(t.mid()) for t in rat))
                        / abs(float(rat[p].mid()))
                    )
                ),
            )
            if False
            else dev
        )
    spread = max(
        max(
            abs(float(((SG[p, q] / Nm[p, q] - SG[p, p] / Nm[p, p]) / (SG[p, p] / Nm[p, p])).mid()))
            for q in range(m)
        )
        for p in range(m)
    )
    cps = [SG[p, p] / Nm[p, p] for p in range(m)]
    wr = [beff[p, 0] / (Nm[p, m] * cps[p]) for p in range(m)]
    print(
        f"state {os.path.basename(path)}: A {A_str}, m {m}; spread of S_G,pq/N_pq along rows {spread:.3e}; b_eff,p/(c_p N_pA) from {float(min(wr, key=lambda t: float(t.mid())).mid()):+.6f} to {float(max(wr, key=lambda t: float(t.mid())).mid()):+.6f}",
        flush=True,
    )
    ctx.prec = Q.MINPREC
    E = lambda j: 1 if j % 2 == 0 else -1
    Mfull = [[Nm[p, q].mid() * E(p) * E(q) for q in range(m + 1)] for p in range(m + 1)]
    pf = R.initial_counts(Mfull, m + 1, m + 1)
    ph = R.initial_counts([row[:] for row in Mfull[:m]], m, m + 1)
    print(
        f"   (m + 1) x (m + 1) matrix E'NE': initial minors positive {pf[0]}, negative {pf[1]}, undecided {pf[2]} of {(m + 1) ** 2}; its first m rows: positive {ph[0]}, negative {ph[1]}, undecided {ph[2]} of {m * (m + 1)}",
        flush=True,
    )
    print("END q996 READING (no verdict)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q996: exception")
        sys.exit(2)
