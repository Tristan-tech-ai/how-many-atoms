"""q1683 (10 Oct 2026, READING, no verdict word, predictions written at 10:14): do the bulk coefficients f_(2n) and the centre coefficients e_n share the same factorial growth rate? Ratios R_n = f_(2n+2)/f_(2n) for n = 10..13 (from f20, f22, f24, f26, f28; the five coefficients give four ratios, printed for n = 10, 11, 12, 13) against n, in units of one order of 1/rho^2; the centre ratio e_(n+1)/e_n against n from d22_out_20.json, converted to the same unit by dividing by 16 pi^2 (v^2 = 16 pi^2 rho^2).
Predictions:
 P1 the bulk ratios R_n for n = 10..13 are linear in n with a slope between 0.025 and 0.035 per order;
 P2 the centre slope alpha/(16 pi^2) = 4.445/157.91 = 0.02815 per order lies within 15 percent of the bulk slope;
 P3 the linear extrapolation of the bulk ratios predicts f30/f28 between 0.45 and 0.50 (the registered F9 range is 0.43 to 0.52).
Control: a synthetic series f_n = Gamma(n + 1.5) A^-n with A = 20 returns the slope 1/A = 0.05 within 2 percent through the same fitting code.
Usage: py q1683_bulk_growth_rate.py (from small_gap_q206/; seconds)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

mp.mp.dps = 30
H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
txt = open(os.path.join(H, "q1557_stdout_ORD28.txt"), encoding="utf-8").read()
fx = {}
for n in (20, 22, 24, 26, 28):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    fx[n] = float(mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30)))
R = {n: fx[2 * n + 2] / fx[2 * n] for n in (10, 11, 12, 13)}


def slope(ns, vals):
    c = np.polyfit(ns, vals, 1)
    return c[0], c[1]


sb, ib = slope(list(R), list(R.values()))
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
E = {n: float(S20["e%d" % n]) for n in range(2, 21)}
cr = {n: E[n + 1] / E[n] for n in range(10, 20)}
sc, ic = slope(list(cr), list(cr.values()))
print(
    "bulk ratios f_(2n+2)/f_(2n) for n = 10..13: "
    + ", ".join("%d: %.4f" % (n, R[n]) for n in R)
    + "; slope %.5f per order, intercept %.4f" % (sb, ib)
)
print(
    "centre ratios e_(n+1)/e_n for n = 10..19: slope %.4f per order, intercept %.3f; divided by 16 pi^2: %.5f; alpha/(16 pi^2) = %.5f"
    % (sc, ic, sc / (16 * np.pi**2), 4.445 / (16 * np.pi**2))
)
pred = sb * 14 + ib
print(
    "linear extrapolation of the bulk ratio to n = 14: f30/f28 = %.4f; centre slope within %.1f percent of the bulk slope"
    % (pred, 100 * abs(4.445 / (16 * np.pi**2) / sb - 1))
)
A = 20.0
fs = {n: float(mp.gamma(n + 1.5) * mp.mpf(A) ** -n) for n in range(9, 15)}
Rs = {n: fs[n + 1] / fs[n] for n in (10, 11, 12, 13)}
ss, _ = slope(list(Rs), list(Rs.values()))
lab = lambda q: "holds" if q else "does not hold"
p1 = 0.025 <= sb <= 0.035
p2 = abs(4.445 / (16 * np.pi**2) / sb - 1) < 0.15
p3 = 0.45 <= pred <= 0.50
ctl = abs(ss / 0.05 - 1) < 0.02
print(
    "P1 bulk slope in [0.025, 0.035]: %s; P2 centre slope within 15 percent of the bulk slope: %s; P3 f30/f28 in [0.45, 0.50]: %s; control (synthetic slope %.5f): %s"
    % (lab(p1), lab(p2), lab(p3), ss, lab(ctl))
)
print("READING q1683 done (labels by the script; no verdict word)")
