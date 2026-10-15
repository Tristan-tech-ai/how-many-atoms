"""q970 (7 Oct 2026): READING (no verdict) that diagnoses the FAIL of T77 (q969, amendment 1749): for a state, the initial minors of X = [T | sigma v] that were not certainly positive at 200 bits are recomputed at 800 bits from the midpoints of S and b_eff,
with sigma = +1 and sigma = -1 (the orientation of E b_eff depends on the parity of m: X1 held for m = 29 and 31 and failed for m = 30). It prints for each sigma the count of initial minors that are certainly positive, certainly negative and undecided at 800 bits, by order k,
and the magnitudes (log10) of the smallest and largest positive minors.
Usage: py q970_extended_tp_diagnosis.py STATE.json [STATE2.json ...] (from temporal_within/)"""

import sys, os, math
from flint import arb, arb_mat, ctx
import q969_extended_tp_test as Q


def build(path):
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
    Tm = [[S[p, q].mid() * (-1 if (p + q) % 2 == 0 else 1) for q in range(m)] for p in range(m)]
    vv = [beff[p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(m)]
    return A_str, m, odd, Tm, vv


def classify(X, m, n):
    out = {}

    def add(k, d):
        c = out.setdefault(k, [0, 0, 0, None, None])
        if d > 0:
            c[0] += 1
        elif d < 0:
            c[1] += 1
        else:
            c[2] += 1

    for k in range(1, m + 1):
        for j in range(0, n - k + 1):
            add(k, Q.minor(X, range(0, k), range(j, j + k)))
        for i in range(1, m - k + 1):
            add(k, Q.minor(X, range(i, i + k), range(0, k)))
    return out


def main():
    for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
        r = build(path)
        if r is None:
            print(f"VOID q970: {path} does not polish")
            sys.exit(2)
        A_str, m, odd, Tm, vv = r
        print(f"state {os.path.basename(path)}: A {A_str}, m {m}, odd {odd}", flush=True)
        for prec in (200, 800):
            ctx.prec = prec
            Tb = [[arb(x) for x in row] for row in Tm]
            for sig in (1, -1):
                v2 = [arb(sig * x) for x in vv]
                X = [list(Tb[p]) + [v2[p]] for p in range(m)]
                c = classify(X, m, m + 1)
                pos = sum(x[0] for x in c.values())
                neg = sum(x[1] for x in c.values())
                und = sum(x[2] for x in c.values())
                bad = {k: (x[1], x[2]) for k, x in c.items() if x[1] or x[2]}
                print(
                    f"   prec {prec}, sigma {sig:+d}: initial minors certainly positive {pos}, certainly negative {neg}, undecided {und}; orders with non-positive minors (negative, undecided): "
                    + (", ".join(f"{k}:{v}" for k, v in sorted(bad.items())) if bad else "none"),
                    flush=True,
                )
        ctx.prec = 200
    print("END q970 READING (no verdict)")


if __name__ == "__main__":
    main()
