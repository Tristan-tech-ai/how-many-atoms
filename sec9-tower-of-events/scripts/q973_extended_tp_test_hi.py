"""q973 (7 Oct 2026): verdict script of T77b (amendment 1749), the repaired replicate of T77 (q969, FAIL by the letter at 200 bits, amendment 1749 gives the facts): is X = [T | v] (reduced Hessian T = -E S E extended by the forcing column v = sigma E b_eff) strictly totally positive?
Repairs against q969: (1) the state is polished again at 600 bits to the residual 1e-120 (the stored states have 1e-39; the corner minors of X have cancellation ratios of 1e51, so their signs need a relative accuracy better than 1e-52 in v; reading q972); (2) all minors are computed at 1200 bits from the exact midpoints;
(3) the orientation is sigma = +1 for odd m and -1 for even m (the rule of the Cramer sign (-1)^(m+1); at 200 bits X1 held for m = 29 and 31 and failed for m = 30, reading of q970); (4) a control on every state: the opposite orientation must fail X1.
Items per state: X1 every entry of v = sigma E b_eff is certainly positive; X2 all m(m + 1) initial minors of X (Gasca-Pena) are certainly positive (the count must equal m(m + 1)); X3 the m maximal minors of X without column i, with v moved to the last column, are certainly positive; X4 the opposite orientation fails X1.
Verdict (last line): PASS if X1 to X4 hold at every listed state; FAIL if a processed state fails an item; VOID if a state does not polish to 1e-100, a count differs from m(m + 1) or a control does not behave.
Controls (synthetic, 1200 bits): PC the Gaussian kernel exp(-0.1 (p - q)^2) with the extra column exp(-0.1 (p - m)^2) passes X1 to X3; NC1 the same with the sign of v[3] flipped fails X2; NC2 the same with v alternating in sign fails X1 and X2; NC3 the same with the extra column replaced by a column inside the range fails X2 or X3.
Usage: py q973_extended_tp_test_hi.py STATE1.json [STATE2.json ...] (from temporal_within/); with --controls-only only the synthetic controls run.
"""

import sys, os
from flint import arb, arb_mat, ctx

arbf = lambda t, n=5: t.str(n, radius=False)
MINPREC, POLPREC = 1200, 600


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
    ctx.prec = MINPREC
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

    ctx.prec = POLPREC
    F.L2P = (arb.pi() * 2).log() / 2
    A_str, u, w, we, wc, d = F.load_state(path)
    m = len(u)
    odd = wc is not None
    cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
    nn = cap.nn
    v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
    v, res, ok = T.newton_full(cap, v, lambda *a: None, tol=arb("1e-120"), maxit=8)
    resid = F.maxabs(res)
    if not ok or not (resid < arb("1e-100")):
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
    ctx.prec = MINPREC
    sg = 1 if m % 2 == 1 else -1
    Tm = [[S[p, q].mid() * (-1 if (p + q) % 2 == 0 else 1) for q in range(m)] for p in range(m)]
    vv = [sg * beff[p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(m)]
    g = gates(Tm, vv, m)
    opp = [-x for x in vv]
    g["X4"] = not all(t > 0 for t in opp)
    g["resid"] = arbf(resid, 3)
    ctx.prec = 200
    return A_str, m, odd, g


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
        print("VERDICT q973 T77b: VOID (a control does not behave)")
        sys.exit(2)
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--controls-only" in sys.argv or not paths:
        print("VERDICT q973 T77b: INCOMPLETE (no state processed)")
        sys.exit(3)
    allok = True
    for p in paths:
        r = state_gates(p)
        if r is None:
            print(f"VERDICT q973 T77b: VOID (the state {p} does not polish to 1e-100 at 600 bits)")
            sys.exit(2)
        A_str, m, odd, g = r
        if g["count"] != m * (m + 1):
            print(f"VERDICT q973 T77b: VOID (initial minor count {g['count']} differs from {m * (m + 1)})")
            sys.exit(2)
        ok = g["X1"] and g["X2"] and g["X3"] and g["X4"]
        allok = allok and ok
        print(
            f"state {os.path.basename(p)}: A {A_str}, m {m}, odd {odd}, polished residual {g['resid']}: X1 {g['X1']}, X2 {g['X2']} ({g['count']} initial minors, {g['bad']} not certainly positive), X3 {g['X3']}, X4 {g['X4']}",
            flush=True,
        )
    print(f"VERDICT q973 T77b: {'PASS' if allok else 'FAIL'}; states {len(paths)}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VERDICT q973 T77b: VOID (exception)")
        sys.exit(2)
