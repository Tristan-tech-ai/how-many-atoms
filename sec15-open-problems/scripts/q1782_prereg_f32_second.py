"""q1782 (10 Oct 2026, REGISTRATION, no verdict word): prospective prediction, written before the exact f32 = F_16 exists (q1557_stdout_ORD32.txt has no f32 line), of the SECOND-lowest coefficient of f32, the coefficient of pi^-18, and of the sum S2 of its two lowest powers (pi^-16 and pi^-18).
Chain of the argument (q1772, q1780, q1781): F = sum_K E_K(x) z^K with E_K = (c_K(2K), c_K(2K+1), c_K(2K+2)) in x; the target is c_7(16) = 8^16 [pi^-18] F_16 = E_7[2]. From K E_K = sum i s_i E_(K-i) one gets E_7[2] = R_7[2] + s_7[0] + s_7[1] + s_7[2], with R_7 and the data E_0..E_6 known, s_7[0], s_7[1] fixed by d_7 and e_7 (known from F_14 and F_15) and s_7[2] = g_7 + 15 e_7 + 105 d_7. So E_7[2] is the known part plus g_7, and g_7 is the only unknown.
RULE (fixed here, before the run). s_k = g_k / e_k for k = 3..6 (exact, q1781). Three extrapolations of s_7: (A) s_6 + (s_6 - s_5); (B) A + (s_6 - 2 s_5 + s_4); (C) A + the mean of the two second differences of s_3..s_6. The band of s_7 is [0.94 min(A, B, C), 1.06 max(A, B, C)]; the widths 6 percent are fixed from the hold-out of the same rules at k = 6 (printed as facts, rule A -2.9 percent, rules B and C +5.2 percent). g_7 = s_7 e_7 and E_7[2] follows linearly. The registered statements are:
 R1 the coefficient of pi^-18 in f32 times 8^16 is negative and lies in the band [lo, hi] printed by this script (written to q1782_prereg_f32_second.txt);
 R2 the sum S2 = T16 + term2 of the two lowest powers lies in the band printed below, where T16 = bhat(8) 8^-16 pi^-16 with |bhat(8)| in the band of q1775 and term2 = c_7(16) 8^-16 pi^-18 with c_7(16) in the band of R1.
Controls (otherwise VOID): C1 the forward map reproduces the data E_K[0..2] for K = 1..6 from the l_(k,m) it found; C2 d_7 and e_7 from F_14 and F_15 equal the values of q1780; C3 the known part plus the true g_k reproduces c_K(2K+2) for K = 6 when the same code is applied one step lower (hold-out of the formula, not of the extrapolation).
NOT BLIND: g_k for k <= 6 and d_7, e_7 are known; the prediction is of f32, which does not exist yet. Note that R1 and R2 are not independent of q1775 (R2 uses its band).
Usage: py q1782_prereg_f32_second.py (from small_gap_q206/; seconds; writes q1782_prereg_f32_second.txt)."""

import os, re, sys, hashlib, datetime
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
H = os.path.dirname(os.path.abspath(__file__))
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
    n = 2 * k
    return [l[0], l[1] + n * l[0], l[2] + (n + 1) * l[1] + R((n + 1) * n, 2) * l[0]]


def forward(l, N):
    s = {k: s_from_l(k, l[k]) for k in range(1, N + 1)}
    E = {0: list(E0)}
    for K in range(1, N + 1):
        acc = [R(0)] * NX
        for i in range(1, K + 1):
            acc = add(acc, scal(mul(s[i], E[K - i]), i))
        E[K] = scal(acc, R(1, K))
    return E


def inverse(E, N):
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
    return l, s


# ---- data
src = open(os.path.join(H, "births_law.py"), encoding="utf-8").read().splitlines()
ns = {"PI": sp.pi}
for ln in src:
    if re.match(r"^(k1 =|c6 =|c10 =|f1[2-9] =|f20 =)", ln):
        exec(ln, ns)
F = {1: ns["k1"]}
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


CF = {j: coeffs(F[j]) for j in range(1, 16)}


def c(K, eps):
    j = 2 * K + eps
    return R(1) if j == 0 else CF[j][2 * (j - K)] * sp.Integer(8) ** j


N = 6
E = {K: [R(c(K, 0)), R(c(K, 1)), R(c(K, 2))] for K in range(0, N + 1)}
l, s = inverse(E, N)
# ---- K = 7: known part
E7 = [R(c(7, 0)), R(c(7, 1))]
acc = [R(0)] * NX
for i in range(1, 7):
    acc = add(acc, scal(mul(s[i], E[7 - i]), i))
R7 = scal(acc, R(1, 7))
s70 = E7[0] - R7[0]
s71 = (E7[1] - R7[1]) - s70
d7 = s70
e7 = s71 - 14 * d7


def E7_2(g7):
    return (
        R7[2] + s70 + s71 + (g7 + 15 * e7 + 105 * d7)
    )  # s_7[2] = g7 + 15 e7 + 105 d7, and (s_7 E_0)[2] = s70 + s71 + s_7[2]


q80 = open(os.path.join(H, "q1780_out.txt"), encoding="utf-8").read()


def parse(tag):
    line = [ln for ln in q80.splitlines() if ln.startswith(tag)][0]
    return {
        int(m.group(1)): R(m.group(2)) for m in re.finditer(r"(\d): (-?\d+(?:/\d+)?)", line.split(":", 1)[1])
    }


c1 = forward(l, N)
ok1 = all(c1[K][:3] == E[K][:3] for K in range(1, N + 1))
ok2 = d7 == parse("d_k, k = 1..7")[7] and e7 == parse("e_k, k = 1..7")[7]
# C3: one step lower, the same formula for K = 6 with the true g_6 reproduces c_6(14)
acc6 = [R(0)] * NX
for i in range(1, 6):
    acc6 = add(acc6, scal(mul(s[i], E[6 - i]), i))
R6 = scal(acc6, R(1, 6))
s60 = E[6][0] - R6[0]
s61 = (E[6][1] - R6[1]) - s60
d6 = s60
e6 = s61 - 12 * d6
ok3 = (R6[2] + s60 + s61 + (l[6][2] + 13 * e6 + 78 * d6)) == E[6][2] and d6 == l[6][0] and e6 == l[6][1]
lab = lambda q: "holds" if q else "does not hold"
print(
    "CONTROLS C1 forward map reproduces the data %s; C2 d_7, e_7 equal q1780 %s; C3 K = 6 formula reproduces c_6(14) %s"
    % (lab(ok1), lab(ok2), lab(ok3))
)
if not (ok1 and ok2 and ok3):
    print("VOID: controls failed")
    sys.exit(2)
sk = {k: l[k][2] / l[k][1] for k in range(3, 7)}
A = sk[6] + (sk[6] - sk[5])
sd1 = sk[5] - 2 * sk[4] + sk[3]
sd2 = sk[6] - 2 * sk[5] + sk[4]
B = A + sd2
C = A + (sd1 + sd2) / 2
hA = sk[5] + (sk[5] - sk[4])
hB = hA + (sk[5] - 2 * sk[4] + sk[3])
print(
    "hold-out at k = 6 (facts): rule A predicts s_6 = %.5f, error %.2f percent; rules B and C predict %.5f, error %.2f percent; actual s_6 = %.5f"
    % (float(hA), float((hA / sk[6] - 1) * 100), float(hB), float((hB / sk[6] - 1) * 100), float(sk[6]))
)
print("s_7 extrapolations: A %.5f, B %.5f, C %.5f" % (float(A), float(B), float(C)))
slo = R(94, 100) * min(A, B, C)
shi = R(106, 100) * max(A, B, C)
c7 = lambda sv: E7_2(sv * e7)
cA, cB, cC = c7(A), c7(B), c7(C)
c_lo, c_hi = sorted([c7(slo), c7(shi)])
print(
    "c_7(16) = 8^16 [pi^-18] F_16 for A, B, C: %.6e %.6e %.6e; band [%.6e, %.6e]"
    % (float(cA), float(cB), float(cC), float(c_lo), float(c_hi))
)
u = mp.pi**-2
sc = mp.mpf(8) ** -16
bh = (-2.557337e10 + -2.561941e10) / 2
T16 = bh * sc * u**8
t2 = lambda cv: mp.mpf(sp.N(cv, 30)) * sc * u**9
T_lo, T_hi = -2.638799e10 * sc * u**8, -2.480617e10 * sc * u**8
S2_cen = T16 + t2((cA + cB + cC) / 3)
S2_lo = min(T_lo + t2(c_lo), T_hi + t2(c_hi), T_lo + t2(c_hi), T_hi + t2(c_lo))
S2_hi = max(T_lo + t2(c_lo), T_hi + t2(c_hi), T_lo + t2(c_hi), T_hi + t2(c_lo))
print(
    "T16 = %.6e; term2 central = %.6e; S2 central = %.6e; S2 band [%.6e, %.6e]"
    % (float(T16), float(t2((cA + cB + cC) / 3)), float(S2_cen), float(S2_lo), float(S2_hi))
)
print(
    "for comparison only: q1778 predicted F_16 = -3.675078e-12; S2 / F_16 (q1778) = %.4f"
    % float(S2_cen / mp.mpf("-3.675078e-12"))
)
me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
out = (
    "Registration q1782 at %s on %s by code (f32 does not exist yet; the run py q1557_f12_symbolic.py 32 started at about 14:00:30). Prediction R1: the coefficient of pi^-18 in f32 times 8^16 is negative and lies in [%.6e, %.6e] (extrapolations A, B, C of s_7 = g_7 / e_7 give %.6e, %.6e, %.6e). Prediction R2: the sum S2 of the two lowest powers (pi^-16 and pi^-18) of f32 lies in [%.6e, %.6e] (central %.6e). Rule in the docstring of small_gap_q206/q1782_prereg_f32_second.py (sha256 %s). Evaluate with q1784 when q1557_stdout_ORD32.txt contains the line for f32.\n"
    % (
        datetime.datetime.now().strftime("%H:%M:%S"),
        datetime.date.today().isoformat(),
        float(c_lo),
        float(c_hi),
        float(cA),
        float(cB),
        float(cC),
        float(S2_lo),
        float(S2_hi),
        float(S2_cen),
        me,
    )
)
open(os.path.join(H, "q1782_prereg_f32_second.txt"), "w", encoding="utf-8").write(out)
print(out.strip())
