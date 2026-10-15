"""q1056 (7 Oct 2026): READING (no verdict): sign structure of the consecutive minors of the full Jacobian J of the KKT system at the K 65 state A = 43.5803 (64 x 64; unknowns u_1..u_m, w_1..w_m, we, wc; rows D(u_j) - D(A), D'(u_j), mass, centre).
Orderings: interleaved columns (u_1, w_1, u_2, w_2, ..., we, wc) and interleaved rows (D'_1, D_1, D'_2, D_2, ..., mass, centre). For orders k = 1, 2, 3 and every window of k consecutive rows and k consecutive columns (inside the first 2 m rows and columns) the sign of the minor is computed at 200 bits (certain signs only); the signs are classified by (row index mod 2, column index mod 2, and whether the window is above, on or below the diagonal band).
A sign-regular structure would give one sign per class. Usage: py q1056_full_jacobian_minor_signs_reading.py
"""

import sys
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q


def main():
    import q920_cap_flint as F
    import q921_cap_tools as T

    path = "tower/K65/results_q220_gauss_K65_A43.5803.json"
    ctx.prec = Q.POLPREC
    F.L2P = (arb.pi() * 2).log() / 2
    A_str, u, w, we, wc, d = F.load_state(path)
    m = len(u)
    odd = wc is not None
    cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
    v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
    v, res, ok = T.newton_full(cap, v, lambda *a: None, tol=arb("1e-120"), maxit=8)
    if not ok:
        print("VOID q1056: polish failed")
        sys.exit(2)
    res, J, b = cap.res_jac(v, True, True)
    n = J.nrows()
    rows = [r for j in range(m) for r in (m + j, j)]
    cols = [c for j in range(m) for c in (j, m + j)]
    ctx.prec = 400
    M = [[J[rows[i], cols[j]].mid() for j in range(2 * m)] for i in range(2 * m)]
    print(
        f"READING q1056: consecutive minors of the interleaved Jacobian, K 65 A 43.5803, {2 * m} x {2 * m} block"
    )
    for k in (1, 2, 3):
        cls = {}
        for i in range(2 * m - k + 1):
            for j in range(2 * m - k + 1):
                sub = arb_mat(k, k, [M[i + a][j + c] for a in range(k) for c in range(k)])
                dt = sub.det()
                sg = 0 if not (dt > 0 or dt < 0) else (1 if dt > 0 else -1)
                band = "above" if j - i > k else ("below" if i - j > k else "band")
                key = (i % 2, j % 2, band)
                cls.setdefault(key, [0, 0, 0])
                cls[key][0 if sg > 0 else (1 if sg < 0 else 2)] += 1
        print(
            f"order {k}: classes (row parity, column parity, position) -> counts [positive, negative, undecided]"
        )
        for key in sorted(cls):
            print(f"   {key}: {cls[key]}")
    print("END q1056 READING (no verdict)")


if __name__ == "__main__":
    main()
