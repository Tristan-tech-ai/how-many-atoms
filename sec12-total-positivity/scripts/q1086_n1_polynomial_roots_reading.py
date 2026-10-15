"""q1086 (7 Oct 2026): READING (no verdict): the polynomial N1(t) of the margin (q988) and its roots.
N1(t) is the determinant of the m x m matrix [T_G + t diag(-D'') | v] without its first column. The diagonal term enters only the columns 2..m, so N1 is a polynomial of degree m - 1 in t. The script evaluates N1 at t = 0, 1, ..., m - 1 at the precision of q988 (600-bit polish, 1200-bit minors),
interpolates the coefficients in 400-digit arithmetic (mpmath), prints N1(0), N1(1), their ratio, the real and complex roots sorted by modulus, and the sums sum ln|1 - 1/t_k|.
Usage: py q1086_n1_polynomial_roots_reading.py STATE.json"""

import sys, os
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q
from mpmath import mp, mpf, mpc, polyroots, log, fabs, nstr

mp.dps = 400


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
        print(f"VOID q1086: {path} does not polish")
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

    deg = m - 1
    ts = list(range(deg + 1))
    vals = []
    for t in ts:
        x = N1(str(t))
        vals.append(mpf(x.mid().str(380, radius=False)))
    # Newton divided differences -> monomial coefficients
    n = len(ts)
    coef = vals[:]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (ts[i] - ts[i - j])
    poly = [mpf(0)] * n
    basis = [mpf(1)]
    for i in range(n):
        for k, c in enumerate(basis):
            poly[k] += coef[i] * c
        nb = [mpf(0)] * (len(basis) + 1)
        for k, c in enumerate(basis):
            nb[k + 1] += c
            nb[k] -= ts[i] * c
        basis = nb
    P = lambda t: sum(poly[k] * mpf(t) ** k for k in range(n))
    print(
        f"READING q1086: {os.path.basename(path)}: A {A_str}, m {m}, degree {deg}; N1(0) = {nstr(P(0), 8)}, N1(1) = {nstr(P(1), 8)}, N1(0)/N1(1) = {nstr(P(0) / P(1), 8)}",
        flush=True,
    )
    chk = max(fabs(P(t) - vals[i]) / fabs(vals[i]) for i, t in enumerate(ts))
    print(f"interpolation check (relative, at the nodes): {nstr(chk, 3)}")
    lead = poly[deg]
    roots = polyroots(poly[::-1], maxsteps=2000, extraprec=2000)
    roots = sorted(roots, key=lambda z: abs(z))
    print("roots t_k sorted by modulus (real part, imaginary part):")
    for k, z in enumerate(roots):
        print(f"  {k + 1:3d}  {nstr(z.real, 10):>16}  {nstr(z.imag, 4):>12}")
    nreal = sum(1 for z in roots if abs(z.imag) < mpf(10) ** -30 * abs(z))
    print(
        f"real roots (|Im| < 1e-30 |t|): {nreal} of {len(roots)}; negative real roots: {sum(1 for z in roots if abs(z.imag) < mpf(10) ** -30 * abs(z) and z.real < 0)}"
    )
    s = sum(log(fabs(1 - 1 / z)) for z in roots)
    print(f"sum ln|1 - 1/t_k| = {nstr(s, 8)}; ln(N1(1)/N1(0)) = {nstr(log(P(1) / P(0)), 8)}")
    print("END q1086 READING (no verdict)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q1086: exception")
        sys.exit(2)
