"""q979 (7 Oct 2026): READING (no verdict): the same margin scan as q978 for the extended matrix X_t = [T_t | v], v = sigma E b_eff (the margin of T79, M3, where X_t was the failing matrix at t = 1.25).
For a state (600-bit polish, 1200-bit minors): the number of initial minors of X_t that are not certainly positive for t = 1, 1.02, 1.05, 1.1, 1.15, 1.2, 1.25 and the (order, type, index) of the first few.
Usage: py q979_extended_margin_reading.py STATE.json (from temporal_within/)"""

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
        print("VOID q979: polish failed")
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
    TG = [[SG[p, q].mid() * (-1 if (p + q) % 2 == 0 else 1) for q in range(m)] for p in range(m)]
    dd = [-dpp[p].mid() for p in range(m)]
    sgm = 1 if m % 2 == 1 else -1
    vv = [sgm * beff[p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(m)]
    print(f"state {os.path.basename(path)}: A {A_str}, m {m}", flush=True)
    for t in ("1", "1.02", "1.05", "1.1", "1.15", "1.2", "1.25"):
        Tt = [[TG[p][q] + (arb(t) * dd[p] if p == q else 0) for q in range(m)] for p in range(m)]
        Xt = [list(Tt[p]) + [vv[p]] for p in range(m)]
        bad = []
        for k in range(1, m + 1):
            for j in range(0, m + 1 - k + 1):
                if not (Q.minor(Xt, range(0, k), range(j, j + k)) > 0):
                    bad.append((k, "rows 0..k-1, columns from", j))
            for i in range(1, m - k + 1):
                if not (Q.minor(Xt, range(i, i + k), range(0, k)) > 0):
                    bad.append((k, "rows from, columns 0..k-1", i))
        print(
            f"   t = {t}: initial minors not certainly positive {len(bad)} of {m * (m + 1)}; first few (order, type, index): {bad[:4]}",
            flush=True,
        )
    print("END q979 READING (no verdict)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q979: exception")
        sys.exit(2)
