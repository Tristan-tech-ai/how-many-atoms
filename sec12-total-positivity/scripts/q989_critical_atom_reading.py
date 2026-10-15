"""q989 (7 Oct 2026): READING (no verdict): which atoms' diagonal terms -D''(u_p) drive the sign of the Cramer minor N1 of the innermost flow (q988).
For a polished state: N1(t_vector) = det of [T_G + diag(t_p d_p) | v] without its first column, with d_p = -D''(u_p); for each p the factor t_p (all others 1) is raised along t = 1, 1.5, 2, 3, 5, 10, 30, 100, 1000 and the smallest t on this grid with N1 not certainly positive is bisected (to 1e-3 relative); infinity is printed when the grid is exhausted.
Also prints d_p. Usage: py q989_critical_atom_reading.py STATE.json (from temporal_within/)"""

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
        print("VOID q989: polish failed")
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

    def N1(p, t):
        Xt = [
            [
                TG[r][q] + (arb(t) * dd[r] if (r == q and r == p) else (dd[r] if r == q else 0))
                for q in range(m)
            ]
            + [vv[r]]
            for r in range(m)
        ]
        return Q.minor(Xt, range(m), range(1, m + 1))

    print(
        f"state {os.path.basename(path)}: A {A_str}, m {m}; N1 at the actual diagonal positive {bool(N1(0, '1') > 0)}",
        flush=True,
    )
    grid = ["1.5", "2", "3", "5", "10", "30", "100", "1000"]
    for p in range(m):
        tf = None
        for t in grid:
            if not (N1(p, t) > 0):
                tf = t
                break
        crit = "infinity (grid up to 1000)" if tf is None else None
        if tf is not None:
            lo = float("1" if tf == grid[0] else grid[grid.index(tf) - 1])
            hi = float(tf)
            for _ in range(14):
                mid = (lo + hi) / 2
                if N1(p, f"{mid:.8f}") > 0:
                    lo = mid
                else:
                    hi = mid
            crit = f"{(lo + hi) / 2:.4f}"
        print(
            f"   atom {p + 1}: d_p {float(dd[p].mid()):.4e}; critical factor for this atom alone {crit}",
            flush=True,
        )
    print("END q989 READING (no verdict)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q989: exception")
        sys.exit(2)
