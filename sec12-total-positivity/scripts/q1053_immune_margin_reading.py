"""q1053 (7 Oct 2026): READING (no verdict): the margin t* of the diagonal term for the Cramer minor of the innermost flow, for the immune and NPMLE functionals (the analogue of q988 for the capacity).
Rows of results_q584.json (immune) and results_q601_ext.json (NPMLE) polished as in q986/q1024 (700 bits, residual 1e-120). S = A_blk - B_blk Dm^-1 C_blk of the Jacobian of the KKT system of IMPH, the diagonal explicit term of the derivative rows is g''(p_j) / C = -d_j (g2_over_C of q1024/q1025), so S_G = S + diag(d) and T_G = -E S_G E in the sign convention of q986; v = sigma_J E b_eff. N1(t) = det of [T_G + t diag(d) | v] without its first column (1200-bit minors, q973.minor); t* = the smallest t on 1.00, 1.02, ... with N1 <= 0 (bisection to 1e-4).
Prints per row: J, K, N1(1) positive, t*, J (t* - 1), K (t* - 1).
Usage: py q1053_immune_margin_reading.py immune|npmle ROW [ROW ...]"""

import sys, json
from flint import arb, arb_mat, ctx
import q986_immune_hp_test as H
import q1024_immune_ripple_reading as R24
import q1025_npmle_ripple_reading as R25
import q973_extended_tp_test_hi as Q

kind = sys.argv[1]
rows_idx = [int(t) for t in sys.argv[2:]]
if kind == "immune":
    file, cls, g2f = "results_q584.json", H.IMPH, R24.g2_over_C
else:
    file, cls, g2f = "results_q601_ext.json", R25.IMPHN, R25.g2_over_C
rows = json.load(open(file))["rows"]
print(f"READING q1053: margin t* for the {kind} functional")
for i in rows_idx:
    r = rows[i]
    H.IMPH = cls
    I, u, J, centre, res = R24.polish(r)
    n = len(u)
    m = n - J
    Hh = arb("1e-90")
    a = I.a if hasattr(I, "a") else arb(float(r["a"]))
    Jm = H.jac(I, u, J, centre, Hh)
    d2, p = g2f(I, u, J, centre)
    dd = [-x for x in d2]
    Fp = cls(arb(float(r["a"])) + Hh).F(u, J, centre)
    Fm = cls(arb(float(r["a"])) - Hh).F(u, J, centre)
    b = [((Fp[k] - Fm[k]) / (2 * Hh)).mid() for k in range(n)]
    A_ = arb_mat(J, J)
    B_ = arb_mat(J, m)
    C_ = arb_mat(m, J)
    D_ = arb_mat(m, m)
    for pp in range(J):
        for q in range(J):
            A_[pp, q] = Jm[pp, q]
        for q in range(m):
            B_[pp, q] = Jm[pp, J + q]
    for pp in range(m):
        for q in range(J):
            C_[pp, q] = Jm[J + pp, q]
        for q in range(m):
            D_[pp, q] = Jm[J + pp, J + q]
    Di = D_.inv()
    S = A_ - B_ * Di * C_
    b1 = arb_mat([[b[k]] for k in range(J)])
    b2 = arb_mat([[b[J + k]] for k in range(m)])
    be = b1 - B_ * Di * b2
    ctx.prec = Q.MINPREC
    sg = 1 if J % 2 == 1 else -1
    TG = [
        [
            (S[pp, q].mid() + (dd[pp].mid() if pp == q else 0)) * (-1 if (pp + q) % 2 == 0 else 1)
            for q in range(J)
        ]
        for pp in range(J)
    ]
    vv = [sg * be[pp, 0].mid() * (1 if pp % 2 == 0 else -1) for pp in range(J)]
    ddm = [x.mid() for x in dd]

    def N1(t):
        Xt = [
            [TG[pp][q] + (arb(t) * ddm[pp] if pp == q else 0) for q in range(J)] + [vv[pp]] for pp in range(J)
        ]
        return Q.minor(Xt, range(J), range(1, J + 1))

    n1 = N1("1")
    grid = [f"{1 + 0.02 * k:.2f}" for k in range(101)]
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
    Kv = int(r["K"])
    print(
        f"row {i}: a {float(r['a']):.4f}, K {Kv}, J {J}, centre {centre}; N1(1) positive {bool(n1 > 0)}; crossing {cross if cross is None else f'{cross:.4f}'}; J (t* - 1) {'' if cross is None else f'{J * (cross - 1):.3f}'}; K (t* - 1) {'' if cross is None else f'{Kv * (cross - 1):.3f}'}",
        flush=True,
    )
    ctx.prec = H.PREC
print("END q1053 READING (no verdict)")
