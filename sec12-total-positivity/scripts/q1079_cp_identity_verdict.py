"""q1079 (7 Oct 2026): verdict script of T111 (registered in the amendment that follows 1827; built from q998 of T84): the coefficients c_p = S_G,pp / N_pp of the identity S_G = diag(c) N equal -1 / (2 w_p), where w_p is the mass of the atom at +-u_p. Derivation (paper_births/H2_PROOF_NOTE.md, Step 5): at a stationary configuration the rows D'(u_p) of the KKT Jacobian are -(1/(2 w_p)) times the Gram rows of psi_p and the value rows are -(1/2) times the Gram rows of chi_j, so the Schur complement over the mass block is -(1/2) diag(1/w) N.
Items per state (600-bit polish to 1e-120, then 1500-bit tables as in q998): C1 the largest |2 w_p c_p + 1| over p = 1..m is below 1e-20; C2 (instrument, as G2 of q998) the largest relative spread along the rows of S_G,pq / N_pq is below 1e-20.
Verdict (last line): PASS if C1 and C2 hold at every listed state; FAIL if a processed state fails C1; VOID if C2 fails, a state does not polish, a control does not behave or an exception occurs.
Controls (synthetic and on the state data): PC c_p = -1/(2 w_p) built from the masses of a state passes C1; NC1 c_p = -1/w_p fails; NC2 the masses permuted (rows 1 and 2 exchanged) fail C1 on the real c_p of the first state.
Usage: py q1079_cp_identity_verdict.py STATE1.json [STATE2.json ...] | py q1079_cp_identity_verdict.py --registration-run STATE1.json (controls on the first state only)
"""

import sys, os
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q
import q974_tp_region_reading as R


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
    cw = [float((2 * ww[p] * cps[p] + 1).mid()) for p in range(m)]
    ccf = [float(c.mid()) for c in cps]
    wwf = [float(x.mid()) for x in ww]
    return (
        A_str,
        m,
        odd,
        {
            "cw": cw,
            "cps": ccf,
            "ww": wwf,
            "G1": pf[0] == (m + 1) ** 2,
            "pos": pf[0],
            "neg": pf[1],
            "und": pf[2],
            "G2": bool(spread < 1e-20),
            "spread": spread,
            "cnt": (m + 1) ** 2,
        },
    )


TOL = 1e-20


def c1(cps, ww):
    from fractions import Fraction

    return max(abs(2 * Fraction(w) * Fraction(c) + 1) for c, w in zip(cps, ww)) < Fraction(1, 10**20)


def main():
    reg = "--registration-run" in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not paths:
        print("VERDICT q1079 T111: VOID (no state given)")
        sys.exit(2)
    r0 = state_items(paths[0])
    if r0 is None:
        print("VERDICT q1079 T111: VOID (the first state does not polish to 1e-120)")
        sys.exit(2)
    _, m0, _, g0 = r0
    from fractions import Fraction

    pc = c1([-1 / (2 * Fraction(w)) for w in g0["ww"]], g0["ww"])
    n1 = not c1([-1 / Fraction(w) for w in g0["ww"]], g0["ww"])
    w2 = list(g0["ww"])
    w2[0], w2[1] = w2[1], w2[0]
    n2 = not c1(g0["cps"], w2)
    print(
        f"controls: PC {pc}, NC1 {n1}, NC2 {n2}; instrument G2 at the first state {g0['G2']} (spread {g0['spread']:.2e})",
        flush=True,
    )
    if not (pc and n1 and n2):
        print("VERDICT q1079 T111: VOID (a control does not behave)")
        sys.exit(2)
    if reg:
        print("VERDICT q1079 T111: INCOMPLETE (controls run only)")
        sys.exit(3)
    bad = 0
    for p in paths:
        r = state_items(p)
        if r is None:
            print(f"VERDICT q1079 T111: VOID (the state {p} does not polish to 1e-120)")
            sys.exit(2)
        A_str, m, odd, g = r
        if not g["G2"]:
            print(f"VERDICT q1079 T111: VOID (instrument check C2 fails at {p}: spread {g['spread']:.3e})")
            sys.exit(2)
        dev = max(abs(x) for x in g["cw"])
        ok = dev < TOL
        bad += not ok
        print(
            f"state {os.path.basename(p)}: A {A_str}, m {m}, odd {odd}: C1 {ok} (largest |2 w_p c_p + 1| = {dev:.3e}); C2 True (spread {g['spread']:.2e}); c_p from {min(g['cps']):.5f} to {max(g['cps']):.5f}",
            flush=True,
        )
    print(f"VERDICT q1079 T111: {'PASS' if bad == 0 else 'FAIL'}; states {len(paths)}; failing {bad}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VERDICT q1079 T111: VOID (exception)")
        sys.exit(2)
