"""q988 (7 Oct 2026): READING (no verdict): the margin t* of the diagonal term for the flow of the innermost position along a family.
For a polished state (600-bit polish to 1e-120, 1200-bit minors): N1(t) = det of the m x m matrix X_t without its first column, X_t = [T_G + t diag(-D'') | v]; by the Cramer formula of 1751 the sign of N1 is the sign of the flow of u_1 (positive at t = 1 in all states of T77b, T78, T81).
Prints per state: A, m, N1 at t = 1 (positive?), the smallest t on the grid t = 1.00, 1.02, ..., 2.00 with N1 <= 0 (not certainly positive) and the bisection of the crossing to 1e-4, and also the largest t at which T_t alone is certainly positive in its last two initial minors (order m and m - 1).
Usage: py q988_margin_along_family_reading.py STATE.json [STATE2.json ...] (from temporal_within/)"""

import sys, os
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q


def main():
    import q920_cap_flint as F
    import q921_cap_tools as T

    for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
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
            print(f"VOID q988: {path} does not polish")
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
            Xt = [
                [TG[p][q] + (arb(t) * dd[p] if p == q else 0) for q in range(m)] + [vv[p]] for p in range(m)
            ]
            return Q.minor(Xt, range(m), range(1, m + 1))

        n1 = N1("1")
        grid = [f"{1 + 0.02 * k:.2f}" for k in range(51)]
        tfail = None
        for t in grid:
            if not (N1(t) > 0):
                tfail = t
                break
        cross = None
        if tfail is not None and tfail != "1.00":
            lo = float(grid[grid.index(tfail) - 1])
            hi = float(tfail)
            for _ in range(10):
                mid = (lo + hi) / 2
                if N1(f"{mid:.6f}") > 0:
                    lo = mid
                else:
                    hi = mid
            cross = (lo + hi) / 2
        print(
            f"{os.path.basename(path)}: A {A_str}, m {m}; N1(1) positive {bool(n1 > 0)}; first grid t with N1 not certainly positive {tfail}; crossing {cross if cross is None else f'{cross:.4f}'}",
            flush=True,
        )
        ctx.prec = 200
    print("END q988 READING (no verdict)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q988: exception")
        sys.exit(2)
