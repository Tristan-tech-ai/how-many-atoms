"""q998 (7 Oct 2026): verdict script of T84 (amendment 1763): the Gram matrix of the projected derivative functions of all positions, wall atom included, is totally positive after E-conjugation.
For a state (600-bit polish to 1e-120, then tables, projection and minors at 1500 bits from the exact midpoints of the polished state with the radii of the balls kept): psi_p = dQ/du_p (p = 1..m), psi_A = dQ/dA, the mass directions chi_k = phi_k^s - phi_A^s (chi_0 = phi(y) - phi_A^s/2 for odd K), g = psi - P psi in the weight 1/Q, N the Gram matrix of (g_1..g_m, g_A) of size (m + 1) (the construction of q976 and q997).
Items per state: G1 all (m + 1)^2 initial minors of E' N E' (E' = diag((-1)^j), j = 0..m) are certainly positive (the count is checked); G2 instrument check: the largest relative spread along the rows of S_G,pq/N_pq (q = 1..m) is below 1e-20 (S_G = diag(c) N, 1751-1754; q997 found 1.7e-32 at the state K 65 A = 43.5803).
Verdict (last line): PASS if G1 and G2 hold at every listed state; FAIL if a processed state fails G1; VOID if G2 fails, a state does not polish, a count differs or a control does not behave.
Controls (synthetic, 1500 bits, m + 1 = 12): PC the Gaussian kernel matrix exp(-0.1 (p - q)^2) passes G1; NC1 the same with one entry sign-flipped fails G1; NC2 the identity matrix fails G1 (zero minors).
Usage: py q998_wall_gram_test.py STATE1.json [STATE2.json ...] (from temporal_within/); --controls-only runs only the synthetic controls.
"""

import sys, os
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q
import q974_tp_region_reading as R


def controls(n=12):
    ctx.prec = 1500
    K = [[(arb(-1) / 10 * ((p - q) ** 2)).exp() for q in range(n)] for p in range(n)]
    pc = R.initial_counts(K, n, n)
    K2 = [list(r) for r in K]
    K2[2][5] = -K2[2][5]
    n1 = R.initial_counts(K2, n, n)
    I = [[arb(1) if p == q else arb(0) for q in range(n)] for p in range(n)]
    n2 = R.initial_counts(I, n, n)
    ctx.prec = 200
    return {"PC": pc[0] == n * n, "NC1": n1[0] < n * n, "NC2": n2[0] < n * n}


def state_items(path):
    import q920_cap_flint as F
    import q921_cap_tools as T

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
        return None
    ctx.prec = 1500
    F.L2P = (arb.pi() * 2).log() / 2
    cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
    res, J, b = cap.res_jac(v, True, True)
    ctx.prec = 1500
    F.L2P = (arb.pi() * 2).log() / 2
    cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
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
    spread = max(
        max(
            abs(float(((SG[p, q] / Nm[p, q] - SG[p, p] / Nm[p, p]) / (SG[p, p] / Nm[p, p])).mid()))
            for q in range(m)
        )
        for p in range(m)
    )
    cps = [SG[p, p] / Nm[p, p] for p in range(m)]
    wr = [beff[p, 0] / (Nm[p, m] * cps[p]) for p in range(m)]

    ctx.prec = 1500
    E = lambda j: 1 if j % 2 == 0 else -1
    Mfull = [[Nm[p, q] * E(p) * E(q) for q in range(m + 1)] for p in range(m + 1)]
    pf = R.initial_counts(Mfull, m + 1, m + 1)
    ctx.prec = 200
    return (
        A_str,
        m,
        odd,
        {
            "G1": pf[0] == (m + 1) ** 2,
            "pos": pf[0],
            "neg": pf[1],
            "und": pf[2],
            "G2": bool(spread < 1e-20),
            "spread": spread,
            "cnt": (m + 1) ** 2,
        },
    )


def main():
    c = controls()
    print("controls:", c, flush=True)
    if not all(c.values()):
        print("VERDICT q998 T84: VOID (a control does not behave)")
        sys.exit(2)
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--controls-only" in sys.argv or not paths:
        print("VERDICT q998 T84: INCOMPLETE (no state processed)")
        sys.exit(3)
    allok = True
    for p in paths:
        r = state_items(p)
        if r is None:
            print(f"VERDICT q998 T84: VOID (the state {p} does not polish to 1e-120)")
            sys.exit(2)
        A_str, m, odd, g = r
        if not g["G2"]:
            print(f"VERDICT q998 T84: VOID (instrument check G2 fails at {p}: spread {g['spread']:.3e})")
            sys.exit(2)
        allok = allok and g["G1"]
        print(
            f"state {os.path.basename(p)}: A {A_str}, m {m}, odd {odd}: G1 {g['G1']} (initial minors positive {g['pos']}, negative {g['neg']}, undecided {g['und']} of {g['cnt']}); G2 True (spread {g['spread']:.2e})",
            flush=True,
        )
    print(f"VERDICT q998 T84: {'PASS' if allok else 'FAIL'}; states {len(paths)}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VERDICT q998 T84: VOID (exception)")
        sys.exit(2)
