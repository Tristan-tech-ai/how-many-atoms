"""q811 (BACKLOG 42, 03 Oct 2026): the Szego loss condition evaluated on the leading-order (extremal-sector) formal density of the
ellipse, for every ring size, with no ring solves. Density in t: 1 + sum_k c_2k cos 2kt, c_2k = 2 a_k (eps/2)^k (q764; a_k at one R).
Moments mu(m) = c_|m|/2 (even m), mu(odd) = 0, as in q806. Verblunsky coefficients alpha_0 .. alpha_(2N-1) by the Levinson-Durbin
recursion (checked against q806's Toeplitz solves at small n, CHECK=1). For each even K the copying ring is lost where alpha_(K-1)
reaches tau = +1 (atom at the major vertex) or -1 (none); the copying regime needs |alpha_j| < 1 for j <= K - 2.
Usage: py q811_szego_series.py A_JSON EPS_LO EPS_HI NEPS [KMAX]   (env DPS, CHECK)"""

import sys, os, json
from mpmath import mp, mpf, matrix, lu_solve, log, exp

mp.dps = int(os.environ.get("DPS", "120"))
d = json.load(open(sys.argv[1]))
A = {int(x["k"]): mpf(x["a"]) for x in d["a"]}
N = max(A)
lo, hi, ne = mpf(sys.argv[2]), mpf(sys.argv[3]), int(sys.argv[4])
KMAX = int(sys.argv[5]) if len(sys.argv) > 5 else 2 * N


def verblunsky(eps, nmax):
    mu = [mpf(1)] + [
        (A[m // 2] * (eps / 2) ** (m // 2) if m % 2 == 0 else mpf(0)) for m in range(1, nmax + 1)
    ]
    a = [mpf(1)]
    E = mu[0]
    al = []
    for n in range(nmax):  # predictor a (a_0 = 1), reflection k_n; Phi_(n+1)(0) = k_n
        acc = sum(a[i] * mu[n + 1 - i] for i in range(n + 1))
        k = -acc / E
        an = a + [mpf(0)]
        a = [an[i] + k * an[n + 1 - i] for i in range(n + 2)]
        E = E * (1 - k * k)
        al.append(-k)  # alpha_n = -Phi_(n+1)(0)
    return al


if os.environ.get("CHECK") == "1":  # q806's Toeplitz route at one eps
    eps = (lo * hi) ** mpf(0.5)
    mu = lambda m: (
        A[abs(m) // 2] * (eps / 2) ** (abs(m) // 2)
        if abs(m) % 2 == 0 and m != 0
        else (mpf(1) if m == 0 else mpf(0))
    )
    for n in (1, 3, 5, 9):
        M = matrix(n + 1, n + 1)
        r = matrix(n + 1, 1)
        for j in range(n + 1):
            for q in range(n + 1):
                M[j, q] = mu(j - q)
            r[j] = -mu(j - n - 1)
        print(
            "check n",
            n,
            "Toeplitz",
            mp.nstr(-lu_solve(M, r)[0], 15),
            "Levinson",
            mp.nstr(verblunsky(eps, n + 1)[n], 15),
        )
    sys.exit()
grid = [exp(log(lo) + (log(hi) - log(lo)) * i / (ne - 1)) for i in range(ne)]
AL = [verblunsky(e, KMAX) for e in grid]
print(f"q811: a_k from {sys.argv[1]} (R {d['R']}, N {N}); eps {lo}..{hi}, {ne} points; K up to {KMAX}")
rows = []
for K in range(4, KMAX + 1, 2):
    j = K - 1
    ev = []
    for i in range(ne - 1, 0, -1):  # from large eps downward
        for tau in (1, -1):
            f1, f0 = AL[i][j] - tau, AL[i - 1][j] - tau
            if f1 * f0 < 0:
                inside = all(abs(AL[i][q]) < 1 for q in range(K - 1))
                e = exp(log(grid[i]) + (log(grid[i - 1]) - log(grid[i])) * f1 / (f1 - f0))
                ev.append((float(e), tau, inside))
    first = [x for x in ev if x[2]]
    rows.append((K, first[0] if first else None, len(ev)))
    if first:
        print(
            f"K {K:4d}: first in-regime crossing eps {first[0][0]:.7f} tau {first[0][1]:+d}; K eps {K*first[0][0]:.4f}; crossings {len(ev)}"
        )
    else:
        print(f"K {K:4d}: no in-regime crossing ({len(ev)} crossings total)")
