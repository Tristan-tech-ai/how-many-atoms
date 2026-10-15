"""q969 (7 Oct 2026): verdict script of T77 (amendment 1748, BACKLOG T16 candidate (i)): is the reduced Hessian T = -E S E of a capacity state, extended by the forcing column v = E b_eff, a strictly totally positive matrix X = [T | v] (m rows, m + 1 columns)?
E = diag((-1)^p); S = A_s - B_s Dm^-1 C_s the reduced Hessian of q924b; b_eff = b_u - B_s Dm^-1 b_r the forcing on the positions after the masses are eliminated; all in 200-bit arb arithmetic from a polished state.
Why it matters: by Cramer's rule (T^-1 v)_i = det(T with column i replaced by v)/det T; if X is TP, the determinant with v moved to the last column is a minor of X and is positive, so T^-1 v alternates as (-1)^(m-i) and the flow du_u = E T^-1 v has one sign at all positions.
Items per state: X1 every entry of v is certainly positive; X2 all m*(m+1) initial minors of X (Gasca-Pena: consecutive rows and columns of order k that contain the first row or the first column; they decide strict total positivity of a matrix) are certainly positive (the arb ball lies above 0);
X3 the m maximal minors det(X without column i), i = 1..m, are certainly positive after moving v to the last column (that is the sign pattern of T^-1 v).
Verdict (last line): PASS if X1, X2, X3 hold at every listed state; FAIL if a state is processed and an item fails; VOID if a state does not polish, a count differs from m(m+1), or a control does not behave.
Controls (synthetic, 200 bits): PC the Gaussian kernel K_pq = exp(-0.1 (p-q)^2) with the extra column exp(-0.1 (p-m)^2) must pass X1-X3; NC1 the same with the entry v[3] sign-flipped must fail X2; NC2 the same with v alternating in sign (v_p (-1)^p) must fail X1 and X2; NC3 the same with v replaced by the first column of K shifted so that it is not a continuation (v_p = exp(-0.1 (p-m/2)^2)... a column inside the range) must fail X2 or X3.
Usage: py q969_extended_tp_test.py STATE1.json [STATE2.json ...] (from temporal_within/); with --controls-only only the synthetic controls run.
"""

import sys, os
from flint import arb, arb_mat, ctx

arbf = lambda t, n=5: t.str(n, radius=False)


def minor(X, rows, cols):
    return arb_mat(len(rows), len(cols), [X[r][c] for r in rows for c in cols]).det()


def initial_minors(X, m, n):
    cnt = 0
    bad = 0
    worst = None
    for k in range(1, m + 1):
        for j in range(0, n - k + 1):
            d = minor(X, range(0, k), range(j, j + k))
            cnt += 1
            if not (d > 0):
                bad += 1
        for i in range(1, m - k + 1):
            d = minor(X, range(i, i + k), range(0, k))
            cnt += 1
            if not (d > 0):
                bad += 1
    return cnt, bad


def gates(Tm, v, m):
    X = [list(Tm[p]) + [v[p]] for p in range(m)]
    n = m + 1
    x1 = all(t > 0 for t in v)
    cnt, bad = initial_minors(X, m, n)
    x2 = cnt == m * n and bad == 0
    x3 = True
    for i in range(m):
        cols = [c for c in range(m) if c != i] + [m]
        if not (minor(X, range(m), cols) > 0):
            x3 = False
    return {"X1": bool(x1), "X2": bool(x2), "X3": bool(x3), "count": cnt, "bad": bad}


def controls(m=12):
    ctx.prec = 400
    K = [[(arb(-1) / 10 * ((p - q) ** 2)).exp() for q in range(m)] for p in range(m)]
    v = [(arb(-1) / 10 * ((p - m) ** 2)).exp() for p in range(m)]
    pc = gates(K, v, m)
    v1 = list(v)
    v1[3] = -v1[3]
    n1 = gates(K, v1, m)
    v2 = [v[p] * (1 if p % 2 == 0 else -1) for p in range(m)]
    n2 = gates(K, v2, m)
    v3 = [(arb(-1) / 10 * ((p - m / 2) ** 2)).exp() for p in range(m)]
    n3 = gates(K, v3, m)
    c = {
        "PC": pc["X1"] and pc["X2"] and pc["X3"],
        "NC1": not n1["X2"],
        "NC2": (not n2["X1"]) and (not n2["X2"]),
        "NC3": (not n3["X2"]) or (not n3["X3"]),
    }
    ctx.prec = 200
    return c, (pc, n1, n2, n3)


def state_gates(path):
    import q920_cap_flint as F
    import q921_cap_tools as T

    A_str, u, w, we, wc, d = F.load_state(path)
    m = len(u)
    odd = wc is not None
    cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
    nn = cap.nn
    v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
    ctx.prec = 200
    v, res, ok = T.newton_full(cap, v, lambda *a: None, tol=arb("1e-50"), maxit=10)
    if not ok:
        return None
    res, J, b = cap.res_jac(v, True, True)
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
    Tm = [[S[p, q] * (-1 if (p + q) % 2 == 0 else 1) for q in range(m)] for p in range(m)]
    vv = [beff[p, 0] * (1 if p % 2 == 0 else -1) for p in range(m)]
    return A_str, m, odd, gates(Tm, vv, m)


def main():
    c, parts = controls()
    print(
        "controls:",
        c,
        "(PC",
        parts[0],
        "| NC1",
        parts[1],
        "| NC2",
        parts[2],
        "| NC3",
        parts[3],
        ")",
        flush=True,
    )
    if not all(c.values()):
        print("VERDICT q969 T77: VOID (a control does not behave)")
        sys.exit(2)
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--controls-only" in sys.argv or not paths:
        print("VERDICT q969 T77: INCOMPLETE (no state processed)")
        sys.exit(3)
    allok = True
    for p in paths:
        r = state_gates(p)
        if r is None:
            print(f"VERDICT q969 T77: VOID (the state {p} does not polish)")
            sys.exit(2)
        A_str, m, odd, g = r
        if g["count"] != m * (m + 1):
            print(f"VERDICT q969 T77: VOID (initial minor count {g['count']} differs from {m * (m + 1)})")
            sys.exit(2)
        ok = g["X1"] and g["X2"] and g["X3"]
        allok = allok and ok
        print(
            f"state {os.path.basename(p)}: A {A_str}, m {m}, odd {odd}: X1 {g['X1']}, X2 {g['X2']} ({g['count']} initial minors, {g['bad']} not certainly positive), X3 {g['X3']}",
            flush=True,
        )
    print(f"VERDICT q969 T77: {'PASS' if allok else 'FAIL'}; states {len(paths)}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VERDICT q969 T77: VOID (exception)")
        sys.exit(2)
