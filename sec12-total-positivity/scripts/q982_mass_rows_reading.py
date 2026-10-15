"""q982 (7 Oct 2026): READING (no verdict) on the masses: with the positions moving outward (L1, amendment 1751), is the flow of the centre mass dwc/dA = [M1 du_u + M2]_wc, M1 = -Dm^-1 C, M2 = -Dm^-1 b_r, implied by the signs of the wc row of M1 and of M2?
For a polished state (600 bits, residual 1e-120) it prints, for the mass unknowns (w_1..w_m, we, wc), the number of positive and negative entries of the row of M1 and the sign of M2 and the value of the flow (M1 du_u + M2), with du_u from the reduced system.
Usage: py q982_mass_rows_reading.py STATE.json (from temporal_within/)"""

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
        print("VOID q982: polish failed")
        sys.exit(2)
    res, J, b = cap.res_jac(v, True, True)
    R = list(range(m)) + list(range(2 * m, nn))
    Mc = list(range(m, nn))
    r_ = len(R)
    As = arb_mat(m, m, [J[m + p, q] for p in range(m) for q in range(m)])
    Bs = arb_mat(m, r_, [J[m + p, c] for p in range(m) for c in Mc])
    Cs = arb_mat(r_, m, [J[r, q] for r in R for q in range(m)])
    Dm = arb_mat(r_, r_, [J[r, c] for r in R for c in Mc])
    br = arb_mat(r_, 1, [b[r] for r in R])
    bu = arb_mat(m, 1, [b[m + p] for p in range(m)])
    S = As - Bs * Dm.solve(Cs)
    beff = bu - Bs * Dm.solve(br)
    du = -(S.solve(beff))
    M1 = -(Dm.solve(Cs))
    M2 = -(Dm.solve(br))
    dm = M1 * du + M2
    names = [f"w_{k + 1}" for k in range(m)] + ["we"] + (["wc"] if odd else [])
    print(
        f"state {os.path.basename(path)}: A {A_str}, m {m}, odd {odd}; du_u all positive: {all(du[p, 0] > 0 for p in range(m))}",
        flush=True,
    )
    for idx in [0, 1, m // 2, m - 1, m] + ([m + 1] if odd else []):
        row = [M1[idx, q] for q in range(m)]
        pos = sum(1 for t in row if t > 0)
        neg = sum(1 for t in row if t < 0)
        print(
            f"   {names[idx]}: row of M1 positive {pos}, negative {neg} of {m}; M2 {'+' if M2[idx, 0] > 0 else '-'} ({float(M2[idx, 0].mid()):.3e}); flow {float(dm[idx, 0].mid()):+.4e}; sum of positive terms {float(sum((t * du[q, 0] for q, t in enumerate(row) if t > 0), arb(0)).mid()):.3e}, negative {float(sum((t * du[q, 0] for q, t in enumerate(row) if t < 0), arb(0)).mid()):.3e}",
            flush=True,
        )
    print("END q982 READING (no verdict)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q982: exception")
        sys.exit(2)
