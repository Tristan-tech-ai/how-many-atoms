"""q1089 (7 Oct 2026): verdict script of T118 (registered in the amendment that follows 1837; built from q998 and q1082): the residual r = S 1 + b_eff of the rigid translation (S the reduced Hessian with the diagonal term, b_eff the forcing, both at the polished stationary state, 1 the vector of ones) has the sign (-1)^p at every atom p = 1..m, and b_eff has the sign (-1)^(p+1) (reading q1082 at the K 65 state, amendment 1835).
Items per state: L1 every r_p is certainly non-zero with sign (-1)^p; L2 every b_eff_p is certainly non-zero with sign (-1)^(p+1); I1 (instrument) the relative spread G2 of q998 is below 1e-20 (the state is stationary to the precision needed).
Verdict (last line): PASS if L1, L2 and I1 hold at all 6 states; FAIL if a processed state violates L1 or L2; VOID if I1 fails, a state does not polish, a control misbehaves or an exception occurs.
Controls (on the sign-pattern function): PC the pattern (-1)^p for p = 1..31 passes; NC1 the same with one entry flipped fails; NC2 the all-positive pattern fails.
States: tower/K65 A = 43.5803 (m 31) and tower/K71 A = 46.4359, 46.5589, 46.6889, 46.8189, 46.9439 (m 34).
Usage: py q1089_translation_residual_sign_verdict.py [--registration-run] (controls only with the flag; no state is read)
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

    Sfull = arb_mat(m, m, [SG[p, q] + (dpp[p] if p == q else 0) for p in range(m) for q in range(m)])
    one = arb_mat(m, 1, [1] * m)
    rr = Sfull * one + beff
    sg_ok = [(1 if (rr[p, 0] > 0) else (-1 if (rr[p, 0] < 0) else 0)) for p in range(m)]
    sb_ok = [(1 if (beff[p, 0] > 0) else (-1 if (beff[p, 0] < 0) else 0)) for p in range(m)]
    ctx.prec = 200
    return (
        A_str,
        m,
        odd,
        {
            "sr": sg_ok,
            "sb": sb_ok,
            "rmin": min(abs(float(rr[p, 0].mid())) for p in range(m)),
            "spread": spread,
        },
    )


STATES = ["tower/K65/results_q220_gauss_K65_A43.5803.json"] + [
    f"tower/K71/results_q220_gauss_K71_A{a}.json"
    for a in ("46.4359", "46.5589", "46.6889", "46.8189", "46.9439")
]


def pattern_ok(signs, first):
    """signs[p] for p = 0..m-1 (atom p + 1); first is the sign expected at atom 1."""
    return all(s == first * (1 if p % 2 == 0 else -1) for p, s in enumerate(signs))


def main():
    reg = "--registration-run" in sys.argv
    base = [(-1) ** p for p in range(1, 32)]
    flipped = list(base)
    flipped[7] = -flipped[7]
    pc = pattern_ok(base, -1)
    nc1 = not pattern_ok(flipped, -1)
    nc2 = not pattern_ok([1] * 31, -1)
    print(f"controls: PC {pc}, NC1 {nc1}, NC2 {nc2}", flush=True)
    if not (pc and nc1 and nc2):
        print("VERDICT q1089 T118: VOID (a control does not behave)")
        sys.exit(2)
    if reg:
        print("VERDICT q1089 T118: INCOMPLETE (controls run only)")
        sys.exit(3)
    bad = 0
    for p in STATES:
        r = state_items(p)
        if r is None:
            print(f"VERDICT q1089 T118: VOID (the state {p} does not polish to 1e-120)")
            sys.exit(2)
        A_str, m, odd, g = r
        if not g["spread"] < 1e-20:
            print(f"VERDICT q1089 T118: VOID (instrument check I1 fails at {p}: spread {g['spread']:.3e})")
            sys.exit(2)
        l1 = pattern_ok(g["sr"], -1)
        l2 = pattern_ok(g["sb"], 1)
        bad += not (l1 and l2)
        print(
            f"{os.path.basename(p)}: m {m}: L1 {l1} (signs of r: first {g['sr'][:4]}, smallest |r_p| {g['rmin']:.3e}); L2 {l2}; I1 True (spread {g['spread']:.2e})",
            flush=True,
        )
    print(f"VERDICT q1089 T118: {'PASS' if bad == 0 else 'FAIL'}; states {len(STATES)}; failing {bad}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VERDICT q1089 T118: VOID (exception)")
        sys.exit(2)
