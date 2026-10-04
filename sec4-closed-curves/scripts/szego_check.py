# my own check of the Szego identity (not B's code): Toeplitz solves for alpha_n, paraorthogonal nodes, weights by moment fit
import json, sys
from mpmath import mp, mpf, matrix, lu_solve, polyroots, arg, cos, pi, re

mp.dps = 50
d = json.load(open(sys.argv[1]))
K = int(sys.argv[2])
tau = int(sys.argv[3])
c = {0: mpf(2)}
for i, x in enumerate(d["c"]):
    c[2 * (i + 1)] = mpf(x)
mu = lambda m: c.get(abs(m), mpf(0)) / 2 if abs(m) % 2 == 0 else mpf(0)


def phi(n):  # monic orthogonal polynomial of degree n: coefficients a_0..a_{n-1}, then 1
    if n == 0:
        return [mpf(1)]
    A = matrix(n, n)
    rhs = matrix(n, 1)
    for j in range(n):
        for k in range(n):
            A[j, k] = mu(j - k)
        rhs[j] = -mu(j - n)
    a = lu_solve(A, rhs)
    return [a[k] for k in range(n)] + [mpf(1)]


alpha = [-phi(n + 1)[0] for n in range(K)]  # alpha_n = -Phi_{n+1}(0), n = 0..K-1
P = phi(K - 1)
Ps = P[::-1]  # Phi_{K-1} and its reversed polynomial (real coefficients)
para = [mpf(0)] + P  # z Phi_{K-1}
para = [para[i] - tau * (Ps[i] if i < len(Ps) else 0) for i in range(K + 1)]
roots = polyroots(para[::-1], maxsteps=400, extraprec=200)
t = sorted(arg(z) for z in roots)
from mpmath import exp, mpc

M = matrix(K, K)
r = matrix(K, 1)
for j in range(K):
    for k in range(K):
        M[j, k] = exp(-1j * j * t[k])
    r[j] = mu(j)
w = [re(x) for x in lu_solve(M, r)]
bK = 2 * sum(w[k] * cos(K * t[k]) for k in range(K))
prod = mpf(1)
for j in range(K - 1):
    prod *= 1 - alpha[j] ** 2
print(f"eps {d['eps']}: alpha_odd {[mp.nstr(alpha[j], 8) for j in range(1, K, 2)]}")
print(
    f"nodes deg {[mp.nstr(x*180/pi, 8) for x in t if 0 <= x <= pi/2 + 1e-30]}, min weight {mp.nstr(min(w), 6)}"
)
print(
    f"c_K {mp.nstr(c[K], 15)}, b_K {mp.nstr(bK, 15)}, c_K - b_K {mp.nstr(c[K] - bK, 12)}, 2(alpha - tau) prod {mp.nstr(2*(alpha[K-1] - tau)*prod, 12)}"
)
