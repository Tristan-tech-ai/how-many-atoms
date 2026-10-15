"""q1057 (7 Oct 2026): READING (no verdict): sign structure of the consecutive minors of the EXTENDED Jacobian of the capacity KKT system at the K 65 state A = 43.5803: the value rows are taken without the subtraction of the wall row, so that the rows are D'(u_j) and D(u_j) (the same function D at the atoms) followed by the wall row D(A), and the columns are u_k, w_k and the wall mass we (the centre column, mass row and centre row are left out).
Rows (interleaved): D'_1, D_1, D'_2, D_2, ..., D'_m, D_m, D_A; columns: u_1, w_1, ..., u_m, w_m, we. Entry formulas as in q920.res_jac: D'(u_j) row = -HD Gm[iPS + j, c] + Dpp (c = u_j), D(u_j) row (undifferenced) = -HD Gm[j, c] + D'(u_j) (c = u_j) (D'(u_j) = 0 at the solution), wall row = -HD Gm[iPA, c].
Consecutive minors of orders 1 to 3 are classified as in q1056 (row parity, column parity, position relative to the diagonal band); the wall row and column are included in the interleaved block as the last row and column.
Usage: py q1057_extended_jacobian_minor_signs_reading.py"""

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
    HD = F.HD
    cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
    v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
    v, res, ok = T.newton_full(cap, v, lambda *a: None, tol=arb("1e-120"), maxit=8)
    if not ok:
        print("VOID q1057: polish failed")
        sys.exit(2)
    N = cap.N
    uu = v[:m]
    ww = v[m : 2 * m]
    wee = v[2 * m]
    wcc = v[2 * m + 1] if odd else None
    rows_, (SA, LA, PA), (SmA, LmA, PmA), cen, Qn, lQ = cap.tables(uu, ww, wee, wcc)
    invQ = [1 / q for q in Qn]
    cols = []
    for k in range(m):
        Sp, Lp, Pp, Sm, Lm, Pm = rows_[k]
        cols.append([ww[k] * (Pp[i] * Sp[i] - Pm[i] * Sm[i]) * invQ[i] for i in range(N)])
    for k in range(m):
        Pp = rows_[k][2]
        Pm = rows_[k][5]
        cols.append([(Pp[i] + Pm[i]) * invQ[i] for i in range(N)])
    cols.append([(PA[i] + PmA[i]) * invQ[i] for i in range(N)])
    ncol = len(cols)
    R = (
        [rows_[j][2] for j in range(m)]
        + [PA]
        + ([cen[2]] if odd else [])
        + [[rows_[j][2][i] * rows_[j][0][i] for i in range(N)] for j in range(m)]
    )
    nrow = len(R)
    Rm = arb_mat(nrow, N, [x for r in R for x in r])
    Ct = arb_mat(N, ncol, [cols[c][i] for i in range(N) for c in range(ncol)])
    Gm = Rm * Ct
    iPA = m
    iPS = m + (2 if odd else 1)
    dpp = [cap.Dpp(rows_[j][0], rows_[j][1], rows_[j][2], lQ) for j in range(m)]
    # extended matrix, natural order: rows [D'_1..D'_m, D_1..D_m, D_A], columns [u_1..u_m, w_1..w_m, we]
    nr = 2 * m + 1
    nc = 2 * m + 1
    E = [[arb(0)] * nc for _ in range(nr)]
    for j in range(m):
        for c in range(nc):
            E[j][c] = -HD * Gm[iPS + j, c] + (dpp[j] if c == j else 0)
            E[m + j][c] = -HD * Gm[j, c]
    for c in range(nc):
        E[2 * m][c] = -HD * Gm[iPA, c]
    rows = [r for j in range(m) for r in (j, m + j)] + [2 * m]
    cols_o = [c for j in range(m) for c in (j, m + j)] + [2 * m]
    ctx.prec = 400
    M = [[E[rows[i]][cols_o[j]].mid() for j in range(nc)] for i in range(nr)]
    print(
        f"READING q1057: consecutive minors of the extended interleaved matrix, K 65 A 43.5803, {nr} x {nc}"
    )
    for k in (1, 2, 3):
        cls = {}
        for i in range(nr - k + 1):
            for j in range(nc - k + 1):
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
    # diagnostic: positions of the minority signs of order 1 in the pure classes
    exp = {}
    for i in range(nr):
        for j in range(nc):
            sg = 1 if M[i][j] > 0 else -1
            key = (i % 2, j % 2, "above" if j - i > 1 else ("below" if i - j > 1 else "band"))
            exp.setdefault(key, []).append((i, j, sg))
    for key in ((0, 0, "above"), (0, 0, "below"), (0, 1, "below"), (1, 0, "above")):
        lst = exp[key]
        pos = sum(1 for x in lst if x[2] > 0)
        neg = len(lst) - pos
        minority = [x for x in lst if x[2] == (-1 if pos > neg else 1)]
        print(
            f"order 1, class {key}: minority entries (row, column): {[(a, b) for a, b, c in minority][:40]}"
        )
    print("END q1057 READING (no verdict)")


if __name__ == "__main__":
    main()
