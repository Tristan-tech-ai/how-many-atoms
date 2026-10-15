"""q974 (7 Oct 2026): READING (no verdict) of the size of the region of configurations in which X = [T | v] stays strictly totally positive (amendment 1751, item a).
A stored state is polished is NOT done: the configuration is the stored state with the interior positions moved by delta h_j r_j (h_j the local spacing, r_j uniform in [-1, 1]) and the interior masses multiplied by 1 + delta r'_j (wall atom, wall mass and centre mass unchanged);
S, b_eff, T = -E S E and v = sigma E b_eff (sigma = (-1)^(m+1)) are recomputed from the same Jacobian formulas at 600 bits (exact for the configuration) and the initial minors of X and of T alone are computed at 1200 bits.
Prints per configuration: X1 (v > 0), the number of initial minors of X certainly positive / certainly negative / undecided, and the same for T alone.
With the flag --polish the stored state is first polished at 600 bits to 1e-120 (the perturbation is then taken around the polished configuration).
Usage: py q974_tp_region_reading.py STATE.json [--polish] DELTA SEED [DELTA SEED ...]  (from temporal_within/)
"""

import sys, os, random
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q


def initial_counts(X, m, n):
    pos = neg = und = 0
    for k in range(1, m + 1):
        wins = [(range(0, k), range(j, j + k)) for j in range(0, n - k + 1)] + [
            (range(i, i + k), range(0, k)) for i in range(1, m - k + 1)
        ]
        for rs, cs in wins:
            d = Q.minor(X, rs, cs)
            if d > 0:
                pos += 1
            elif d < 0:
                neg += 1
            else:
                und += 1
    return pos, neg, und


POLISH = "--polish" in sys.argv


def run(path, delta, seed):
    import q920_cap_flint as F

    ctx.prec = Q.POLPREC
    F.L2P = (arb.pi() * 2).log() / 2
    A_str, u, w, we, wc, d = F.load_state(path)
    m = len(u)
    odd = wc is not None
    cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
    rng = random.Random(seed * 7919 + int(delta * 1e6))
    uu = [t.mid() for t in u]
    ww = [t.mid() for t in w]
    if POLISH:
        import q921_cap_tools as T

        v0 = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
        v0, res0, ok0 = T.newton_full(cap, v0, lambda *a: None, tol=arb("1e-120"), maxit=8)
        if not ok0:
            raise RuntimeError("polish failed")
        uu = v0[:m]
        ww = v0[m : 2 * m]
        we = v0[2 * m]
        wc = v0[2 * m + 1] if odd else None
    h = [
        (
            (uu[min(j + 1, m - 1)] - uu[max(j - 1, 0)]) / 2
            if 0 < j < m - 1
            else (uu[1] - uu[0] if j == 0 else uu[m - 1] - uu[m - 2])
        )
        for j in range(m)
    ]
    u2 = [uu[j] + arb(str(delta)) * h[j] * arb(rng.uniform(-1, 1)) for j in range(m)]
    w2 = [ww[j] * (1 + arb(str(delta)) * arb(rng.uniform(-1, 1))) for j in range(m)]
    ordered = all(u2[j] < u2[j + 1] for j in range(m - 1)) and u2[0] > 0
    v = u2 + w2 + [arb(we).mid()] + ([arb(wc).mid()] if odd else [])
    res, J, b = cap.res_jac(v, True, True)
    nn = cap.nn
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
    ctx.prec = Q.MINPREC
    sg = 1 if m % 2 == 1 else -1
    Tm = [[S[p, q].mid() * (-1 if (p + q) % 2 == 0 else 1) for q in range(m)] for p in range(m)]
    vv = [sg * beff[p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(m)]
    X = [list(Tm[p]) + [vv[p]] for p in range(m)]
    x1 = all(t > 0 for t in vv)
    cx = initial_counts(X, m, m + 1)
    ct = initial_counts([row[:m] for row in X], m, m)
    ctx.prec = 200
    return m, ordered, x1, cx, ct


def main():
    path = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    for i in range(0, len(args), 2):
        delta = float(args[i])
        seed = int(args[i + 1])
        m, ordered, x1, cx, ct = run(path, delta, seed)
        print(
            f"{os.path.basename(path)} delta {delta} seed {seed}: m {m}, ordered {ordered}, X1 {x1}; X initial minors positive {cx[0]}, negative {cx[1]}, undecided {cx[2]} of {m * (m + 1)}; T alone positive {ct[0]}, negative {ct[1]}, undecided {ct[2]} of {m * m}",
            flush=True,
        )
    print("END q974 READING (no verdict)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q974: exception")
        sys.exit(2)
