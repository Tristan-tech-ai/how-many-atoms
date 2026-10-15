"""q1781 (10 Oct 2026, READING, no verdict word): the third diagonal g_k, the coefficient of y^(2k+2) in the logarithm level L_k, from the SECOND-lowest coefficients of F_(2K+2).
Setup (q1772, q1780): with x = u t / 8, y = x/(1-x), z = x^2/u, ln F = -ln(1-x) + sum_k u^-k L_k(y), L_k = d_k y^(2k) + e_k y^(2k+1) + g_k y^(2k+2) + ...  Then u^-k y^(2k+m) = z^k x^m (1-x)^-(2k+m), so ln F = s_0(x) + sum_(k>=1) s_k(x) z^k with s_0 = -ln(1-x) and s_k(x) = sum_m l_(k,m) x^m (1-x)^-(2k+m), (l_(k,0), l_(k,1), l_(k,2)) = (d_k, e_k, g_k). F = sum_K E_K(x) z^K with E_K(x) = sum_eps c_K(2K+eps) x^eps, c_K(j) = 8^j [u^(j-K)] F_j. For eps = 0, 1, 2 these are the lowest, the second lowest... precisely: c_K(2K) and c_K(2K+1) are the bottom coefficients of F_(2K) and F_(2K+1) (q1780) and c_K(2K+2) = 8^(2K+2) [pi^-(2K+4)] F_(2K+2) is the SECOND-lowest coefficient of F_(2K+2).
Recurrence of the exponential: K E_K = sum_(i=1..K) i s_i E_(K-i), E_0 = 1/(1-x). Hence s_K = (1-x) (E_K - R_K) with R_K = (1/K) sum_(i<K) i s_i E_(K-i), and l_(K,0) = s_K[0], l_(K,1) = s_K[1] - 2K l_(K,0), l_(K,2) = s_K[2] - C(2K+1, 2) l_(K,0) - (2K+1) l_(K,1). Everything truncated at x^2. Data: F_2 .. F_14 give K = 0..6 for eps = 0, 1, 2, so g_1 .. g_6.
NOT BLIND: g_1 = -44/3 (y^4 of L_1) and g_2 = 568/3 (y^6 of L_2) are known from q1771; g_3 to g_6 were not computed. Predictions (written before the run):
 P1 the code reproduces g_1 = -44/3 and g_2 = 568/3 (control and identity check; here they come from the second-lowest coefficients of F_4 and F_6 and not from the polynomials of q1771);
 P2 g_k is positive for k = 2..6 (g_1 is negative and g_2 positive, so this may fail);
 P3 the ratios s_k = g_k / e_k for k = 3..6 have differences s_(k+1) - s_k (k = 3, 4, 5) of a single sign;
 P4 the d_k and e_k from this solver equal the values of q1780 for k = 1..6 (consistency of the forward map).
Controls (otherwise VOID): C1 a synthetic series built from chosen d_k, e_k, g_k by the forward map is recovered exactly; C2 a changed coefficient is not; C3 the data give E_0 = 1/(1-x) to x^2 (1, 1, 1).
Usage: py q1781_third_diagonal.py (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
NX = 3
R = sp.Rational


def mul(a, b):
    c = [R(0)] * NX
    for i in range(NX):
        for j in range(NX - i):
            c[i + j] += a[i] * b[j]
    return c


def add(a, b):
    return [p + q for p, q in zip(a, b)]


def scal(a, s):
    return [p * s for p in a]


ONE_M_X = [R(1), R(-1), R(0)]
E0 = [R(1), R(1), R(1)]


def s_from_l(k, l):
    """s_k(x) = sum_m l_m x^m (1-x)^-(2k+m), truncated at x^2."""
    n = 2 * k
    return [l[0], l[1] + n * l[0], l[2] + (n + 1) * l[1] + R((n + 1) * n, 2) * l[0]]


def forward(l, N):
    """E_K for K = 0..N from l[k] = (d_k, e_k, g_k), k = 1..N, by K E_K = sum i s_i E_(K-i)."""
    s = {k: s_from_l(k, l[k]) for k in range(1, N + 1)}
    E = {0: list(E0)}
    for K in range(1, N + 1):
        acc = [R(0)] * NX
        for i in range(1, K + 1):
            acc = add(acc, scal(mul(s[i], E[K - i]), i))
        E[K] = scal(acc, R(1, K))
    return E


def inverse(E, N):
    """l[k] = (d_k, e_k, g_k), k = 1..N, from the targets E_K."""
    l = {}
    s = {}
    for K in range(1, N + 1):
        acc = [R(0)] * NX
        for i in range(1, K):
            acc = add(acc, scal(mul(s[i], E[K - i]), i))
        RK = scal(acc, R(1, K))
        sK = mul(ONE_M_X, add(E[K], scal(RK, -1)))
        s[K] = sK
        d = sK[0]
        e = sK[1] - 2 * K * d
        g = sK[2] - R((2 * K + 1) * 2 * K, 2) * d - (2 * K + 1) * e
        l[K] = (d, e, g)
    return l


lab = lambda q: "holds" if q else "does not hold"
# ---- controls on synthetic series
N = 6
lS = {k: (R(-k, 3 + k), R(-(k * k + 2), 5 * k + 1), R(k * k - 3, 2 * k + 7)) for k in range(1, N + 1)}
ES = forward(lS, N)
lR = inverse(ES, N)
c1 = all(lR[k] == lS[k] for k in lS)
lB = dict(lS)
lB[4] = (lS[4][0], lS[4][1], lS[4][2] + R(1, 9))
EB = forward(lB, N)
lRB = inverse(EB, N)
c2 = (lRB[4] == lB[4]) and (lRB[4] != lS[4]) and any(inverse(ES, N)[k] != lRB[k] for k in range(4, N + 1))
ES2 = {K: list(ES[K]) for K in ES}
ES2[5][2] += R(1, 11)
c2b = inverse(ES2, N)[5] != lS[5]
print(
    "CONTROL C1 synthetic d, e, g recovered exactly: %s; C2 a changed coefficient is not recovered: %s"
    % (lab(c1), lab(c2 and c2b))
)
if not (c1 and c2 and c2b):
    print("VOID: controls failed")
    sys.exit(2)
# ---- data
src = open(os.path.join(H, "births_law.py"), encoding="utf-8").read().splitlines()
ns = {"PI": sp.pi}
for ln in src:
    if re.match(r"^(k1 =|c6 =|c10 =|f1[2-9] =|f20 =)", ln):
        exec(ln, ns)
F = {1: ns["k1"], 0: sp.Integer(1)}
F.update(
    {
        j: f
        for j, f in zip(
            range(2, 11),
            [ns["k4"], ns["c6"], ns["c8"], ns["c10"], ns["f12"], ns["f14"], ns["f16"], ns["f18"], ns["f20"]],
        )
    }
)
txt = open(os.path.join(H, "q1557_stdout_ORD30.txt"), encoding="utf-8").read()
for j, n in zip(range(11, 16), (22, 24, 26, 28, 30)):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    F[j] = sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi})


def coeffs(expr):
    px = sp.Symbol("px")
    return {m[0]: sp.Rational(c) for m, c in sp.Poly(sp.expand(expr.subs(sp.pi, 1 / px)), px).terms()}


CF = {j: coeffs(F[j]) for j in range(1, 15)}


def c(K, eps):
    j = 2 * K + eps
    if j == 0:
        return R(1)
    return CF[j][2 * (j - K)] * sp.Integer(8) ** j  # the term x^j u^-K is u^(j-K) = pi^-(2 (j - K))


E = {K: [R(c(K, 0)), R(c(K, 1)), R(c(K, 2))] for K in range(0, N + 1)}
c3 = E[0] == [1, 1, 1]
print("CONTROL C3 data give E_0 = 1/(1-x) to x^2: %s" % lab(c3))
if not c3:
    print("VOID: data control failed")
    sys.exit(2)
l = inverse(E, N)
print("c_K(2K), c_K(2K+1), c_K(2K+2) for K = 1..6:")
[print("  K=%d: %s, %s, %s" % (K, E[K][0], E[K][1], E[K][2])) for K in range(1, N + 1)]
print("d_k, k = 1..6: " + "; ".join("%d: %s" % (k, l[k][0]) for k in range(1, N + 1)))
print("e_k, k = 1..6: " + "; ".join("%d: %s" % (k, l[k][1]) for k in range(1, N + 1)))
print("g_k, k = 1..6: " + "; ".join("%d: %s (%.6g)" % (k, l[k][2], float(l[k][2])) for k in range(1, N + 1)))
sk = {k: l[k][2] / l[k][1] for k in range(3, N + 1)}
print("s_k = g_k / e_k, k = 3..6: " + " ".join("%.5f" % float(sk[k]) for k in range(3, N + 1)))
print(
    "differences s_(k+1) - s_k, k = 3..5: " + " ".join("%.5f" % float(sk[k + 1] - sk[k]) for k in range(3, N))
)
print("ratios g_(k+1)/g_k, k = 1..5: " + " ".join("%.5f" % float(l[k + 1][2] / l[k][2]) for k in range(1, N)))
big = {k: sorted(p for p in sp.factorint(abs(l[k][2].p)) if p > 31) for k in range(1, N + 1)}
print("primes above 31 in the numerators of g_k: %s" % ({k: v for k, v in big.items() if v} or "none"))
q80 = open(os.path.join(H, "q1780_out.txt"), encoding="utf-8").read()


def parse(tag):
    line = [ln for ln in q80.splitlines() if ln.startswith(tag)][0]
    return {
        int(m.group(1)): R(m.group(2)) for m in re.finditer(r"(\d): (-?\d+(?:/\d+)?)", line.split(":", 1)[1])
    }


d80 = parse("d_k, k = 1..7")
e80 = parse("e_k, k = 1..7")
p1 = l[1][2] == R(-44, 3) and l[2][2] == R(568, 3)
p2 = all(l[k][2] > 0 for k in range(2, N + 1))
dd = [sk[k + 1] - sk[k] for k in range(3, N)]
p3 = all(q > 0 for q in dd) or all(q < 0 for q in dd)
p4 = all(l[k][0] == d80[k] and l[k][1] == e80[k] for k in range(1, N + 1))
print(
    "P1 g_1 = -44/3 and g_2 = 568/3 from the second-lowest coefficients: %s; P2 g_k positive for k = 2..6: %s; P3 single sign of s_(k+1) - s_k for k = 3..5: %s; P4 d_k, e_k equal q1780 for k = 1..6: %s"
    % (lab(p1), lab(p2), lab(p3), lab(p4))
)
print("READING q1781 done (labels by the script; no verdict word)")
