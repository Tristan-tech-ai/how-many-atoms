"""q1088 (7 Oct 2026): verdict script of T117 (registered in the amendment that follows 1836): the polynomial N1(t) of the margin of the innermost flow (q988; N1(t) = det of the m x m matrix [T_G + t diag(-D'') | v] without its first column, degree m - 1 in t) has exactly m - 1 real positive roots, all simple, at each listed state.
Method: N1 is evaluated with certified signs (arb balls at 1200 bits, state polished to 1e-120 at 600 bits as in q988) on the grid t = 0.005 k (k = 1..200) and t = 1.0 * 1.03^k up to 1e12; a sign change between neighbouring grid points counts as one root. N1(t) is a polynomial of degree m - 1, so m - 1 sign changes prove that all roots are real, simple and positive.
Items per state: R1 the number of certified sign changes equals m - 1; I1 (instrument) no sign is undecided.
Verdict (last line): PASS if R1 and I1 hold at all 6 states; INCOMPLETE if I1 holds and some state has fewer than m - 1 sign changes on the grid (the property is then not certified); VOID if I1 fails, a state does not polish, the count exceeds m - 1 (impossible for a polynomial of degree m - 1) or a control misbehaves.
Controls (synthetic polynomials with the same counting function): PC the polynomial prod_{k=1..30} (t - 1.1 * 1.8^k)... of degree 30 with 30 real positive roots must give 30 changes; NC1 the same with one factor replaced by (t^2 + 1) (a complex pair, degree 30) must give 28 changes; NC2 the polynomial of degree 30 with all roots negative must give 0.
States: tower/K65 A = 43.5803 (m 31) and tower/K71 A = 46.4359, 46.5589, 46.6889, 46.8189, 46.9439 (m 34).
Usage: py q1088_n1_real_rootedness_verdict.py [--registration-run] (controls only with the flag; no state is read)
"""

import sys, os
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q

STATES = ["tower/K65/results_q220_gauss_K65_A43.5803.json"] + [
    f"tower/K71/results_q220_gauss_K71_A{a}.json"
    for a in ("46.4359", "46.5589", "46.6889", "46.8189", "46.9439")
]


def grid():
    g = [0.005 * k for k in range(1, 201)]
    t = 1.0
    while t < 1e12:
        t *= 1.03
        g.append(t)
    return sorted(set(round(x, 9) for x in g))


def count_changes(f, g):
    """f(t) -> arb ball; returns (sign changes, undecided points)."""
    prev = None
    ch = 0
    und = 0
    for t in g:
        x = f(t)
        if x > 0:
            s = 1
        elif x < 0:
            s = -1
        else:
            und += 1
            continue
        if prev is not None and s != prev:
            ch += 1
        prev = s
    return ch, und


def poly_from_roots(roots, extra_quad=False):
    def f(t):
        ctx.prec = 400
        x = arb(1)
        for r in roots:
            x *= arb(repr(t)) - arb(repr(r))
        if extra_quad:
            x *= arb(repr(t)) ** 2 + 1
        return x

    return f


def controls():
    ctx.prec = 400
    g = grid()
    pc = count_changes(poly_from_roots([1.1 * 1.8**k for k in range(1, 31)]), g)
    nc1 = count_changes(poly_from_roots([1.1 * 1.8**k for k in range(1, 29)], extra_quad=True), g)
    nc2 = count_changes(poly_from_roots([-1.1 * 1.8**k for k in range(1, 31)]), g)
    ok = pc == (30, 0) and nc1 == (28, 0) and nc2 == (0, 0)
    return ok, {"PC": pc, "NC1": nc1, "NC2": nc2}


def build_N1(path):
    import q920_cap_flint as F
    import q921_cap_tools as T

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
        return None
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
        ctx.prec = Q.MINPREC
        Xt = [
            [TG[p][q] + (arb(repr(t)) * dd[p] if p == q else 0) for q in range(m)] + [vv[p]] for p in range(m)
        ]
        return Q.minor(Xt, range(m), range(1, m + 1))

    return m, N1


def main():
    reg = "--registration-run" in sys.argv
    ok, c = controls()
    print("controls:", c, "->", ok, flush=True)
    if not ok:
        print("VERDICT q1088 T117: VOID (a control does not behave)")
        sys.exit(2)
    if reg:
        print("VERDICT q1088 T117: INCOMPLETE (controls run only)")
        sys.exit(3)
    g = grid()
    incomplete = 0
    for p in STATES:
        r = build_N1(p)
        if r is None:
            print(f"VERDICT q1088 T117: VOID (the state {p} does not polish to 1e-120)")
            sys.exit(2)
        m, N1 = r
        ch, und = count_changes(N1, g)
        if und:
            print(f"VERDICT q1088 T117: VOID (instrument item I1 fails at {p}: {und} undecided signs)")
            sys.exit(2)
        if ch > m - 1:
            print(f"VERDICT q1088 T117: VOID (count {ch} exceeds the degree {m - 1} at {p})")
            sys.exit(2)
        okk = ch == m - 1
        incomplete += not okk
        print(
            f"{os.path.basename(p)}: m {m}, degree {m - 1}, certified sign changes {ch}, undecided {und}, R1 {okk}",
            flush=True,
        )
    print(
        f"VERDICT q1088 T117: {'PASS' if incomplete == 0 else 'INCOMPLETE'}; states {len(STATES)}; not certified {incomplete}"
    )
    sys.exit(0 if incomplete == 0 else 3)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VERDICT q1088 T117: VOID (exception)")
        sys.exit(2)
