"""q1780 (10 Oct 2026, READING, no verdict word): the next-to-diagonal coefficients e_k of the log levels, from the bottom coefficients of the ODD bulk coefficients F_(2K+1).
Setup (readings 206 to 212): F = sum_j F_j t^j, x = u t / 8, y = x/(1-x), and ln F = -ln(1-x) + sum_k u^-k L_k(y) with L_k(y) = d_k y^(2k) + e_k y^(2k+1) + ... The coefficient of x^j u^-k in F is c_k(j) = 8^j [u^(j-k)] F_j. With z = x^2/u the terms x^(2K) u^-K and x^(2K+1) u^-K are the bottom coefficients of F_(2K) and F_(2K+1). A product of factors that reaches x^(2K+1) u^-K exceeds the minimal x-power 2 sum k_i by exactly one, so exactly one factor is raised by one power: the term x of -ln(1-x), or y^(2k) = x^(2k) (1 + 2k x + ...), or the e_k y^(2k+1) = e_k x^(2k+1). Hence
  sum_K bhat_even(K) z^K = exp(D(z)),  D(z) = sum_k d_k z^k  (this is q1772),
  sum_K bhat_odd(K) z^K = exp(D(z)) (1 + E(z)),  E(z) = sum_k (2 k d_k + e_k) z^k,
with bhat_even(K) = 8^(2K) b(2K), bhat_odd(K) = 8^(2K+1) b(2K+1), b(j) the coefficient of the lowest power u^ceil(j/2) of F_j (bhat_odd(0) = 1 from F_1 = 1/(8 pi^2)). So e_k = [z^k] (B_odd/B_even - 1) - 2 k d_k. Data: F_2 .. F_15 give K = 0..7 for both sums, so e_1 .. e_7.
NOT BLIND: e_1 = -20/3 and e_2 = -112/15 are the y^3 and y^5 coefficients of L_1 and L_2 of q1771; e_3 to e_7 were not computed. Predictions (written before the run):
 P1 the code reproduces e_1 = -20/3 and e_2 = -112/15 exactly (control, and a registered check of the identity above, since e_1 and e_2 come here from F_3, F_5 and F_2, F_4 and not from the polynomials of q1771);
 P2 e_k is negative for k = 3..7;
 P3 the ratios r_k = e_k / d_k are smooth: the differences r_(k+1) - r_k have a single sign for k = 2..6.
Controls (otherwise VOID): C1 a synthetic series built from chosen d'_k and e'_k with the same identity is recovered exactly; C2 the same synthetic odd series with one bottom coefficient changed does not give the chosen e'_k back.
Usage: py q1780_next_to_diagonal.py (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
z = sp.Symbol("z")
x = sp.Symbol("x")
NK = 8


def coeffs(expr):
    return {m[0]: sp.Rational(c) for m, c in sp.Poly(sp.expand(expr.subs(sp.pi, 1 / x)), x).terms()}


def solve_e(be, bo):
    """be, bo: dicts K -> bhat_even(K), bhat_odd(K) for K = 0..NK-1. Returns d_k and e_k for k = 1..NK-1."""
    Be = sum(be[K] * z**K for K in range(NK))
    Bo = sum(bo[K] * z**K for K in range(NK))
    D = sp.series(sp.log(Be), z, 0, NK).removeO()
    R = sp.series(Bo / Be - 1, z, 0, NK).removeO()
    d = {k: sp.Rational(D.coeff(z, k)) for k in range(1, NK)}
    e = {k: sp.Rational(R.coeff(z, k)) - 2 * k * d[k] for k in range(1, NK)}
    return d, e


def synth(d, e):
    """bhat_even and bhat_odd from chosen d_k, e_k by the identity of the docstring."""
    D = sum(d[k] * z**k for k in range(1, NK))
    E = sum((2 * k * d[k] + e[k]) * z**k for k in range(1, NK))
    Ee = sp.series(sp.exp(D), z, 0, NK).removeO()
    Eo = sp.series(sp.exp(D) * (1 + E), z, 0, NK).removeO()
    return {K: sp.Rational(Ee.coeff(z, K)) for K in range(NK)}, {
        K: sp.Rational(Eo.coeff(z, K)) for K in range(NK)
    }


lab = lambda q: "holds" if q else "does not hold"
# ---- controls on synthetic series
dS = {k: sp.Rational(-k, 3 + k) for k in range(1, NK)}
eS = {k: sp.Rational(-(k * k + 2), 5 * k + 1) for k in range(1, NK)}
beS, boS = synth(dS, eS)
dR, eR = solve_e(beS, boS)
c1 = all(dR[k] == dS[k] and eR[k] == eS[k] for k in range(1, NK))
boBad = dict(boS)
boBad[4] = boBad[4] + sp.Rational(1, 7)
dB, eB = solve_e(beS, boBad)
c2 = any(eB[k] != eS[k] for k in range(1, NK))
print(
    "CONTROL C1 synthetic d'_k, e'_k recovered exactly: %s; C2 changed bottom coefficient gives other e: %s"
    % (lab(c1), lab(c2))
)
if not (c1 and c2):
    print("VOID: controls failed")
    sys.exit(2)
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


def bottom(j):
    cf = coeffs(F[j])
    p = min(cf)
    assert p == 2 * ((j + 1) // 2), (j, p)  # lowest power of 1/pi is 2 ceil(j/2)
    return cf[p] * sp.Integer(8) ** j


be = {0: sp.Integer(1)}
bo = {0: bottom(1)}
for K in range(1, NK):
    be[K] = bottom(2 * K)
    bo[K] = bottom(2 * K + 1)
d, e = solve_e(be, bo)
print("bhat_odd(K) = 8^(2K+1) b(2K+1), K = 0..7: " + ", ".join(str(bo[K]) for K in range(NK)))
print("d_k, k = 1..7: " + "; ".join("%d: %s" % (k, d[k]) for k in range(1, NK)))
print("e_k, k = 1..7: " + "; ".join("%d: %s (%.6g)" % (k, e[k], float(e[k])) for k in range(1, NK)))
r = {k: e[k] / d[k] for k in range(1, NK)}
print("r_k = e_k / d_k, k = 1..7: " + " ".join("%.5f" % float(r[k]) for k in range(1, NK)))
print(
    "differences r_(k+1) - r_k, k = 1..6: "
    + " ".join("%.5f" % float(r[k + 1] - r[k]) for k in range(1, NK - 1))
)
print("ratios e_(k+1)/e_k, k = 1..6: " + " ".join("%.5f" % float(e[k + 1] / e[k]) for k in range(1, NK - 1)))
big = {k: sorted(p for p in sp.factorint(abs(e[k].p)) if p > 31) for k in range(1, NK)}
print("primes above 31 in the numerators of e_k: %s" % ({k: v for k, v in big.items() if v} or "none"))
p1 = e[1] == sp.Rational(-20, 3) and e[2] == sp.Rational(-112, 15)
p2 = all(e[k] < 0 for k in range(3, NK))
dif = [r[k + 1] - r[k] for k in range(2, NK - 1)]
p3 = all(q > 0 for q in dif) or all(q < 0 for q in dif)
print(
    "P1 e_1 = -20/3 and e_2 = -112/15 from the odd bottoms: %s; P2 e_k negative for k = 3..7: %s; P3 single sign of r_(k+1) - r_k for k = 2..6: %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1780 done (labels by the script; no verdict word)")
