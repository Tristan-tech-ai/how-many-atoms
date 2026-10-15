"""q1058 (7 Oct 2026): library and reading of the sign classes of consecutive minors of the extended Jacobian (see q1057).
Matrix (m = number of positive atoms, even K or odd K without the centre column and rows): rows in the interleaved order D'_1, D_1, ..., D'_m, D_m, D_A (types d, v, ..., v) and columns u_1, w_1, ..., u_m, w_m, we (types u, w, ..., w); entries as in q920.res_jac without the wall-row subtraction in the value rows (q1057).
Interior windows: k consecutive rows i .. i + k - 1 and columns j .. j + k - 1 with i + k <= 2 m and j + k <= 2 m (the wall row and wall column are not in the window). Position: above if j - i > k + 2, below if i - j > k + 2, otherwise band (the band, which holds the nearest neighbours, is not classified; the first version used the band |i - j| <= k and had the nearest-neighbour entries as minorities).
Class key: (k, type of the first row, type of the first column, position).
Usage as a library: signs_by_class(m, M, kmax, imin=0) (windows whose first row index or first column index is below imin are skipped; q1060 uses imin = 6); extended_matrix(path) -> (m, M) -> dict class -> [positive, negative, undecided]; as a script: py q1058_extended_jacobian_sign_classes.py STATE.json [KMAX]
"""

import sys
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q


def extended_matrix(path):
    import q920_cap_flint as F
    import q921_cap_tools as T

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
        raise RuntimeError("polish failed")
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
    Rm = arb_mat(len(R), N, [x for r in R for x in r])
    Ct = arb_mat(N, ncol, [cols[c][i] for i in range(N) for c in range(ncol)])
    Gm = Rm * Ct
    iPA = m
    iPS = m + (2 if odd else 1)
    dpp = [cap.Dpp(rows_[j][0], rows_[j][1], rows_[j][2], lQ) for j in range(m)]
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
    return m, [[E[rows[i]][cols_o[j]].mid() for j in range(nc)] for i in range(nr)]


def signs_by_class(m, M, kmax, imin=0):
    cls = {}
    rtype = lambda i: "d" if i % 2 == 0 else "v"
    ctype = lambda j: "u" if j % 2 == 0 else "w"
    for k in range(1, kmax + 1):
        for i in range(2 * m - k + 1):
            for j in range(2 * m - k + 1):
                pos = "above" if j - i > k + 2 else ("below" if i - j > k + 2 else "band")
                if pos == "band" or i < imin or j < imin:
                    continue
                dt = arb_mat(k, k, [M[i + a][j + c] for a in range(k) for c in range(k)]).det()
                sg = 0 if not (dt > 0 or dt < 0) else (1 if dt > 0 else -1)
                key = (k, rtype(i), ctype(j), pos)
                cls.setdefault(key, [0, 0, 0])
                cls[key][0 if sg > 0 else (1 if sg < 0 else 2)] += 1
    return cls


if __name__ == "__main__":
    path = sys.argv[1]
    kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    m, M = extended_matrix(path)
    cls = signs_by_class(m, M, kmax)
    print(
        f"READING q1058: interior windows outside the band, {path}, m {m}, orders 1 to {kmax}: class (order, first row type, first column type, position) -> [positive, negative, undecided]"
    )
    for key in sorted(cls):
        print(f"   {key}: {cls[key]}")
    print("END q1058 READING (no verdict)")
