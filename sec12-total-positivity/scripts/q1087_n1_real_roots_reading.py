"""q1087 (7 Oct 2026): READING (no verdict): the real positive roots of the polynomial N1(t) of the margin (q988) by sign changes on a grid in t with bisection (a replacement for the interpolation of q1087, whose roots beyond the tenth are noise).
N1(t) is the determinant of the m x m matrix [T_G + t diag(-D'') | v] without its first column. The diagonal term enters only the columns 2..m, so N1 is a polynomial of degree m - 1 in t. The script evaluates N1 at t = 0, 1, ..., m - 1 at the precision of q988 (600-bit polish, 1200-bit minors),
interpolates the coefficients in 400-digit arithmetic (mpmath), prints N1(0), N1(1), their ratio, the real and complex roots sorted by modulus, and the sums sum ln|1 - 1/t_k|.
Usage: py q1087_n1_polynomial_roots_reading.py STATE.json"""

import sys, os
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
        print(f"VOID q1087: {path} does not polish")
        sys.exit(2)
    res, J, b = cap.res_jac(v, True, True)
    rows, _, _, cen, Qn, lQ = cap.tables(v[:m], v[m : 2 * m], v[2 * m], v[2 * m + 1] if odd else None)
    dpp = [cap.Dpp(rows[j][0], rows[j][1], rows[j][2], lQ) for j in range(m)]
    R = list(range(m)) + list(range(2 * m, nn))
    Mc = list(range(m, nn))
    r_ = len(R)
    As = arb_mat(m, m, [J[m + p, q] for p in range(m) for q in range(m)])
    Bs = arb_mat(m, r_, [J[m + p, c] for p in range(m) for c in Mc])
    Cs = arb_mat(r_, m, [J[r, q] for r in R for q in range(m)])
    Dm = arb_mat(r_, r_, [J[r, c] for r in R for c in Mc])
    SG = arb_mat(
        m, m, [As[p, q] - (dpp[p] if p == q else 0) for p in range(m) for q in range(m)]
    ) - Bs * Dm.solve(Cs)
    br = arb_mat(r_, 1, [b[r] for r in R])
    bu = arb_mat(m, 1, [b[m + p] for p in range(m)])
    beff = bu - Bs * Dm.solve(br)
    ctx.prec = Q.MINPREC
    sg = 1 if m % 2 == 1 else -1
    TG = [[SG[p, q].mid() * (-1 if (p + q) % 2 == 0 else 1) for q in range(m)] for p in range(m)]
    vv = [sg * beff[p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(m)]
    dd = [-dpp[p].mid() for p in range(m)]

    def N1(t):
        Xt = [[TG[p][q] + (arb(t) * dd[p] if p == q else 0) for q in range(m)] + [vv[p]] for p in range(m)]
        return Q.minor(Xt, range(m), range(1, m + 1))

    import math
    from flint import arb as _arb

    sgn = lambda t: (lambda x: 1 if x > 0 else (-1 if x < 0 else 0))(N1(f"{t:.12f}"))
    grid = [1.0 + 0.002 * k for k in range(0, 100)] + [1.2 * (1.03**k) for k in range(0, 330)]
    grid = sorted(set(round(g, 12) for g in grid))
    prev = None
    roots = []
    und = 0
    for g in grid:
        s0 = sgn(g)
        if s0 == 0:
            und += 1
            continue
        if prev is not None and s0 != prev[1]:
            lo, hi = prev[0], g
            for _ in range(40):
                mid = (lo + hi) / 2
                sm = sgn(mid)
                if sm == 0:
                    break
                if sm == prev[1]:
                    lo = mid
                else:
                    hi = mid
            roots.append((lo + hi) / 2)
        prev = (g, s0)
    print(
        f"READING q1087: {os.path.basename(path)}: A {A_str}, m {m}; sign of N1 at t = 1: {sgn(1.0)}; sign changes of N1 on t in [0.0, {grid[-1]:.1f}] (grid of {len(grid)} points, undecided {und}): {len(roots)} real positive roots",
        flush=True,
    )
    s0 = sgn(1e-9)
    print(f"sign of N1 at t = 0: {s0}")
    for k, r in enumerate(roots):
        print(
            f"  t_{k + 1:<2d} = {r:.9f}   (t_k - 1) J = {(r - 1) * m:.4f}   t_k / t_(k-1) = {(r / roots[k - 1]) if k else float('nan'):.4f}"
        )
    print("END q1087 READING (no verdict)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q1087: exception")
        sys.exit(2)
