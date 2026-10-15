"""q968 (7 Oct 2026): READING (no verdict) for BACKLOG T16: the structure of the flow du_u/dA = P b_eff of a capacity state, with P = E T^-1 E entrywise non-negative when T = -E S E is totally positive (S the reduced Hessian of q924b, E = diag((-1)^p)),
and b_eff = b_u - B_s Dm^-1 b_r the forcing on the positions after the masses are eliminated.
Facts printed per state (200-bit arb): sign pattern of E b_eff (all one sign?), smallest entry of P, the maximum difference between P b_eff and the flow du of the full system (consistency), and for the row i of P the terms t_ij = P_ij b_eff_j:
the ratios |t_i,j+1|/|t_ij| (the terms of the alternating sum grow with j if all are >= 1; then the sign of the flow is the sign of the last term), their minimum over j for rows 1, m/2 and m, the minimum over all rows and columns,
and the ratio of the last term to the absolute sum of the others for each row.
By total positivity of P the ratio P_i,j+1/P_ij is non-decreasing in i, so if the ratios of row 1 are all >= |b_eff_j|/|b_eff_j+1| the ratios of every row are.
Usage: py q968_row1_terms_reading.py STATE.json   (from temporal_within/)"""

import sys, json, os
from flint import arb, arb_mat, ctx
import q920_cap_flint as F
import q921_cap_tools as T

arbf = lambda t, n=5: t.str(n, radius=False)


def main():
    path = sys.argv[1]
    A_str, u, w, we, wc, d = F.load_state(path)
    m = len(u)
    odd = wc is not None
    cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
    nn = cap.nn
    v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
    ctx.prec = 200
    v, res, ok = T.newton_full(cap, v, lambda *a: None, tol=arb("1e-50"), maxit=10)
    if not ok:
        print(f"VOID q968 {path}: the state does not polish")
        sys.exit(2)
    res, J, b = cap.res_jac(v, True, True)
    du = T.solve_lin(J, [-t for t in b])
    R = list(range(m)) + list(range(2 * m, nn))
    Mc = list(range(m, nn))
    r_ = len(R)
    As = arb_mat(m, m, [J[m + p, q] for p in range(m) for q in range(m)])
    Bs = arb_mat(m, r_, [J[m + p, c] for p in range(m) for c in Mc])
    Cs = arb_mat(r_, m, [J[r, q] for r in R for q in range(m)])
    Dm = arb_mat(r_, r_, [J[r, c] for r in R for c in Mc])
    S = As - Bs * Dm.solve(Cs)
    br = arb_mat(r_, 1, [b[r] for r in R])
    bu = arb_mat(m, 1, [b[m + p] for p in range(m)])
    beff = bu - Bs * Dm.solve(br)
    E = lambda p: -1 if p % 2 else 1
    T_ = arb_mat(m, m, [S[p, q] * (-E(p) * E(q)) for p in range(m) for q in range(m)])
    Ti = T_.inv()
    P = [[Ti[p, q] * (E(p) * E(q)) for q in range(m)] for p in range(m)]
    be = [beff[p, 0] for p in range(m)]
    pm = min(P[p][q].mid() for p in range(m) for q in range(m))
    flow = [sum((P[p][q] * be[q] for q in range(m)), arb(0)) for p in range(m)]
    cons = max(abs((flow[p] - du[p]).mid()) / max(abs(du[p].mid()), arb(1e-300)) for p in range(m))
    sg = [(be[p] * E(p)).mid() > 0 for p in range(m)]
    print(
        f"state {os.path.basename(path)}: A {A_str}, m {m}, odd {odd}; E b_eff entries positive in {sum(sg)} of {m} (all one sign: {all(sg) or not any(sg)}); smallest entry of P {arbf(pm, 4)}; max relative |P b_eff - du| {arbf(cons, 3)}"
    )
    print(
        "   b_eff magnitudes growth |b_j+1|/|b_j|: "
        + " ".join(f"{float(abs(be[j + 1] / be[j]).mid()):.2f}" for j in range(m - 1))
    )
    allmin = None
    for lab, i in (("row 1", 0), (f"row {m // 2 + 1}", m // 2), (f"row {m}", m - 1)):
        rat = [float(abs((P[i][j + 1] * be[j + 1]) / (P[i][j] * be[j])).mid()) for j in range(m - 1)]
        print(
            f"   {lab}: ratios |t_j+1|/|t_j| min {min(rat):.3f} at j {rat.index(min(rat)) + 1}, max {max(rat):.3f}, number below 1: {sum(1 for x in rat if x < 1)}; last term / sum of the abs of the others {float(abs(P[i][m - 1] * be[m - 1]) / sum((abs(P[i][j] * be[j]) for j in range(m - 1)), arb(0)).mid()):.3f}"
        )
    mins = []
    for i in range(m):
        rat = [float(abs((P[i][j + 1] * be[j + 1]) / (P[i][j] * be[j])).mid()) for j in range(m - 1)]
        mins.append(min(rat))
    print(
        f"   minimum over all rows of the minimal ratio: {min(mins):.3f} (row {mins.index(min(mins)) + 1}); rows with minimal ratio below 1: {sum(1 for x in mins if x < 1)} of {m}"
    )
    print("END q968 READING (no verdict)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q968: exception")
        sys.exit(2)
