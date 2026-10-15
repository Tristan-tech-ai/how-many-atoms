"""q1674 (10 Oct 2026, READING, no verdict word, predictions written at 09:57): the lower births K_b 61..82 of q1671 with the stored series to nmax = 20 (d22_out_20.json) and the wall pair refitted on K_b 83..130 at nmax = 20 (bulk law through c28, two linearisation rounds from the q1669 pair), against the nmax = 18 case.
Predictions:
 P1 the largest size over K_b 61..69 at nmax = 20 differs from the nmax = 18 value 1.055e-11 by less than 25 percent;
 P2 all differences at K_b 61..69 stay negative at nmax = 20;
 P3 the change between nmax = 18 and 20 at K_b 61 is smaller in size than 3e-12.
Control: nmax = 18 with the q1669 pair reproduces the q1671 differences (largest size 1.055e-11 over K_b 61..69, 8.804e-12 over 70..82).
Usage: py q1674_lower_k_nmax20.py (from small_gap_q206/; about 2 minutes)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(H, "..", "scratch")
C = os.path.join(H, "..", "centre_scripts")
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
E = {n: float(S20["e%d" % n]) for n in range(2, 21)}
txt = open(os.path.join(H, "q1557_stdout_ORD28.txt"), encoding="utf-8").read()
f = {}
for n in (26, 28):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    f[n] = float(mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30)))
B.F = ((B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28]])[:15]
B.NG = 14
B.g = [1.0]
for n in range(1, 15):
    B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))


def meas(K):
    fn = (
        os.path.join(A, "tower_K%d_birth_out.json" % (K + 1))
        if K % 2 == 0
        else os.path.join(A, "tower_K%d_As2.json" % (K + 1))
    )
    d = json.load(open(fn))
    return d["A_cross"] if K % 2 == 0 else d["A_s"]


Kin = [K for K in range(83, 131) if B.measured(K) is not None]
min_ = np.array([B.measured(K) for K in Kin])
h = 1e-7


def refit(nmax, nu, d0):
    ex = [E[n] for n in range(6, nmax + 1)]
    for rnd in range(2):
        p0 = np.array([B.a_pred(K, nu, d0, ex) for K in Kin])
        M = np.stack(
            [
                [(B.a_pred(K, nu + h, d0, ex) - B.a_pred(K, nu - h, d0, ex)) / (2 * h) for K in Kin],
                [(B.a_pred(K, nu, d0 + h, ex) - B.a_pred(K, nu, d0 - h, ex)) / (2 * h) for K in Kin],
            ],
            1,
        )
        sol, *_ = np.linalg.lstsq(M, -(p0 - min_), rcond=None)
        nu, d0 = nu + sol[0], d0 + sol[1]
    return nu, d0


def diffs(nmax, pair):
    ex = [E[n] for n in range(6, nmax + 1)]
    return {K: B.a_pred(K, *pair, ex) - meas(K) for K in range(61, 83)}


p18 = (-0.0937736508347, 0.0011599160063)
p20 = refit(20, -0.0937736508339, 0.0011599160056)
d18, d20 = diffs(18, p18), diffs(20, p20)
for name, d in (("nmax 18", d18), ("nmax 20", d20)):
    print(
        "%s: differences (1e-12) K_b 61..69: %s; largest size 61..69 %.3e, 70..82 %.3e"
        % (
            name,
            ", ".join("%+.1f" % (d[K] * 1e12) for K in range(61, 70)),
            max(abs(d[K]) for K in range(61, 70)),
            max(abs(d[K]) for K in range(70, 83)),
        )
    )
print("change nmax 18 to 20 at K_b 61: %+.2e; pair nmax 20: (%.13f, %.13f)" % (d20[61] - d18[61], *p20))
lab = lambda q: "holds" if q else "does not hold"
m18, m20 = max(abs(d18[K]) for K in range(61, 70)), max(abs(d20[K]) for K in range(61, 70))
p1 = abs(m20 / 1.055e-11 - 1) < 0.25
p2 = all(d20[K] < 0 for K in range(61, 70))
p3 = abs(d20[61] - d18[61]) < 3e-12
ctl = abs(m18 - 1.055e-11) < 5e-15 and abs(max(abs(d18[K]) for K in range(70, 83)) - 8.804e-12) < 5e-15
print(
    "P1 largest size within 25 percent of 1.055e-11: %s; P2 all negative at K_b 61..69: %s; P3 change at K_b 61 below 3e-12: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1674 done (labels by the script; no verdict word)")
