"""q976 (7 Oct 2026): READING (no verdict): is T_G the Gram matrix of E-conjugated projected derivative functions, and is the matrix of those functions totally positive?
Setting (state polished at 600 bits, then 200 bits): Q the output density on the grid, psi_p = dQ/du_p = w_p [(y - u_p) phi(y - u_p) - (y + u_p) phi(y + u_p)] (the u_p column of q920), chi_k = phi_k^s - phi_A^s (k = 1..m, phi_k^s = phi(y - u_k) + phi(y + u_k)), chi_0 = phi(y) - phi_A^s/2 for odd K:
the directions of the masses under the mass constraint. With the weight 1/Q and the quadrature step H, g_p = psi_p - P psi_p is the projection of psi_p orthogonal to span{chi}. Prints: (1) the largest relative deviation of the row-normalised matrix S_G,pq/N_pq (N_pq = <g_p, g_q>) from a constant along each row
(S_G = c_p N up to a row scale if the identification holds), (2) for the matrix Gh_{p,i} = (-1)^p g_p(y_i) sqrt(H/Q_i) (m x N) the signs of 3000 random minors of each order k = 2..6 over random row and column subsets (increasing indices), after the overall sign (-1)^k is removed.
Usage: py q976_projected_functions_reading.py STATE.json (from temporal_within/)"""

import sys, os, random
import numpy as np
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q


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
        print("VOID q976: polish failed")
        sys.exit(2)
    res, J, b = cap.res_jac(v, True, True)
    uu = v[:m]
    ww = v[m : 2 * m]
    we_ = v[2 * m]
    wc_ = v[2 * m + 1] if odd else None
    rows, (SA, LA, PA), (SmA, LmA, PmA), cen, Qn, lQ = cap.tables(uu, ww, we_, wc_)
    N = cap.N
    RR = list(range(m)) + list(range(2 * m, nn))
    Mc = list(range(m, nn))
    r_ = len(RR)
    dpp = [cap.Dpp(rows[j][0], rows[j][1], rows[j][2], lQ) for j in range(m)]
    As = arb_mat(m, m, [J[m + p, q] for p in range(m) for q in range(m)])
    Bs = arb_mat(m, r_, [J[m + p, c] for p in range(m) for c in Mc])
    Cs = arb_mat(r_, m, [J[r, q] for r in RR for q in range(m)])
    Dm = arb_mat(r_, r_, [J[r, c] for r in RR for c in Mc])
    SG = arb_mat(
        m, m, [As[p, q] - (dpp[p] if p == q else 0) for p in range(m) for q in range(m)]
    ) - Bs * Dm.solve(Cs)
    ctx.prec = 200
    Hh = arb(float(F.G.H))
    invQ = [1 / q for q in Qn]
    psi = [
        [ww[p] * (rows[p][2][i] * rows[p][0][i] - rows[p][5][i] * rows[p][3][i]) for i in range(N)]
        for p in range(m)
    ]
    phA = [PA[i] + PmA[i] for i in range(N)]
    chi = [[rows[k][2][i] + rows[k][5][i] - phA[i] for i in range(N)] for k in range(m)]
    if odd:
        chi.append([cen[2][i] - phA[i] / 2 for i in range(N)])
    nv = len(chi)

    def ip(f, g):
        return Hh * sum((f[i] * g[i] * invQ[i] for i in range(N)), arb(0))

    MV = arb_mat(nv, nv, [ip(chi[a], chi[c]) for a in range(nv) for c in range(nv)])
    Cp = arb_mat(m, nv, [ip(psi[p], chi[c]) for p in range(m) for c in range(nv)])
    coef = MV.solve(Cp.transpose()).transpose()
    g = [
        [psi[p][i] - sum((coef[p, c] * chi[c][i] for c in range(nv)), arb(0)) for i in range(N)]
        for p in range(m)
    ]
    Ng = arb_mat(m, m, [ip(g[p], g[q]) for p in range(m) for q in range(m)])
    dev = 0.0
    for p in range(m):
        rat = [float((SG[p, q] / Ng[p, q]).mid()) for q in range(m) if abs(float(Ng[p, q].mid())) > 1e-200]
        dev = max(dev, (max(rat) - min(rat)) / abs(np.mean(rat)))
    print(
        f"state {os.path.basename(path)}: A {A_str}, m {m}, odd {odd}, grid points {N}; largest relative spread of S_G,pq/N_pq along a row: {dev:.3e}",
        flush=True,
    )
    print(
        "   row scale c_p = S_G,pp/N_pp from "
        + f"{float((SG[0, 0] / Ng[0, 0]).mid()):.4e} (p = 1) to {float((SG[m - 1, m - 1] / Ng[m - 1, m - 1]).mid()):.4e} (p = m); w_p from {float(ww[0].mid()):.4e} to {float(ww[m - 1].mid()):.4e}",
        flush=True,
    )
    Gh = [[(arb(-1) ** p) * g[p][i] * (Hh * invQ[i]).sqrt() for i in range(N)] for p in range(m)]
    rng = random.Random(20261007)
    for k in range(2, 7):
        pos = neg = zer = 0
        for _ in range(3000):
            rs = sorted(rng.sample(range(m), k))
            cs = sorted(rng.sample(range(N), k))
            dd = arb_mat(k, k, [Gh[r][c] for r in rs for c in cs]).det()
            if dd > 0:
                pos += 1
            elif dd < 0:
                neg += 1
            else:
                zer += 1
        print(
            f"   order {k}: sampled minors of Gh positive {pos}, negative {neg}, undecided {zer}", flush=True
        )
    print("END q976 READING (no verdict)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q976: exception")
        sys.exit(2)
