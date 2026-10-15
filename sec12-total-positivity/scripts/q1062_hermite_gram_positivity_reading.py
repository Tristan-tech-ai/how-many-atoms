"""q1062 (7 Oct 2026): READING (no verdict): are the generalized collocation minors of the Gram matrix of the Hermite system {phi(y - u_j), (y - u_j) phi(y - u_j)} (weight 1 / Q of the K 65 state A = 43.5803, atoms u_1 < ... < u_m and the wall atom A) positive?
Rows and columns are index sets of the form: a set of nodes, at each node either the value function alone or the value function and the derivative function (a Hermite pattern); the minor is det G[R, C] with R and C in the order (node, derivative order). Random patterns: sizes 2 to 9, 3000 draws (the first version trimmed unequal patterns with a loop that never ended when the last function was a value function; it was stopped after 22 minutes without output and rewritten), nodes chosen at random from the m + 1 nodes, minors at 400 bits (certain signs only). Prints the counts of positive, negative and undecided minors by size and the smallest |minor| / geometric mean of the diagonal.
Gram entries: G_ab = HD sum_i f_a(y_i) f_b(y_i) / Q(y_i) from the tables of q920 (f = P and S P).
Usage: py q1062_hermite_gram_positivity_reading.py"""

import sys, random
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
        print("VOID q1062: polish failed")
        sys.exit(2)
    uu = v[:m]
    ww = v[m : 2 * m]
    wee = v[2 * m]
    wcc = v[2 * m + 1] if odd else None
    rows_, (SA, LA, PA), (SmA, LmA, PmA), cen, Qn, lQ = cap.tables(uu, ww, wee, wcc)
    N = cap.N
    invQ = [1 / q for q in Qn]
    funcs = (
        []
    )  # (node index, derivative order, values on the grid); nodes 0..m-1 interior atoms, node m the wall atom A (the positive wall atom only)
    for k in range(m):
        Sp, Lp, Pp, Sm, Lm, Pm = rows_[k]
        funcs.append((k, 0, Pp))
        funcs.append((k, 1, [Sp[i] * Pp[i] for i in range(N)]))
    funcs.append((m, 0, PA))
    funcs.append((m, 1, [SA[i] * PA[i] for i in range(N)]))
    nf = len(funcs)
    Fm = arb_mat(N, nf, [funcs[c][2][i] for i in range(N) for c in range(nf)])
    Wd = arb_mat(N, nf, [Fm[i, c] * invQ[i] for i in range(N) for c in range(nf)])
    G = (Fm.transpose() * Wd) * HD
    ctx.prec = 400
    index = {(f[0], f[1]): i for i, f in enumerate(funcs)}
    rnd = random.Random(20261007)
    stats = {}

    def pattern(size):
        # a Hermite pattern of exactly `size` functions: random nodes in increasing order; at each node the value function, and with probability 1/2 the derivative function as well
        nodes = sorted(rnd.sample(range(m + 1), size))
        idx = []
        for nd in nodes:
            if len(idx) == size:
                break
            idx.append(index[(nd, 0)])
            if len(idx) < size and rnd.random() < 0.5:
                idx.append(index[(nd, 1)])
        return idx

    for _ in range(3000):
        k = rnd.randint(2, 9)
        R = pattern(k)
        C = pattern(k)
        sub = arb_mat(len(R), len(C), [G[r, c] for r in R for c in C])
        dt = sub.det()
        s_ = stats.setdefault(k, [0, 0, 0])
        s_[0 if dt > 0 else (1 if dt < 0 else 2)] += 1
    print(
        f"READING q1062: Hermite Gram positivity at the K 65 state, m {m}, functions {nf}; counts by size [positive, negative, undecided]"
    )
    for k in sorted(stats):
        print(f"   size {k}: {stats[k]}")
    print("END q1062 READING (no verdict)")


if __name__ == "__main__":
    main()
