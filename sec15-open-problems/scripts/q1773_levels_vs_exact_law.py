"""q1773 (10 Oct 2026, READING, no verdict word): how well do the levels 0..2 of ln F = -ln(1-x) + sum_k u^-k L_k(y), y = x/(1-x), x = u t / 8, u = 1/pi^2, reproduce the exact partial sum S(t) = sum_(j<=15) F_j t^j of the bulk law at t = 1/rho^2? L_1(y) = -2/3 y^2 - 20/3 y^3 - 44/3 y^4 - 8 y^5 and L_2(y) as in q1771. F_1..F_15 are the exact forms (births_law.py and q1557_stdout_ORD30.txt), mpmath at 50 digits.
NOT BLIND: the level forms are known. Predictions (written at 14:01):
 P1 the relative error of ln S against the levels 0..2 at rho = 1 is below 1e-7;
 P2 the error grows when rho decreases (x grows): err(0.8) > err(1.0) > err(1.2);
 P3 the error of the levels 0..1 only at rho = 1 lies between 1e-6 and 1e-4.
Controls (otherwise VOID): C1 at x = 1e-4 (t = 8e-4 pi^2 ... chosen so that x = 1e-4) the error of the levels 0..2 is below 1e-14; C2 a changed coefficient of L_1 (the y^2 coefficient -2/3 replaced by -0.67) gives an error above 1e-5 at rho = 1.
The error means |ln S - (levels)| / |ln S|; the absolute error and ln S are also printed (added after the first output, which is kept as q1773_out_first_run.txt; the labels do not change). Note: S is cut at j = 15, so for rho = 0.8 the neglected F_16 t^16 is part of the difference; it is printed as a size estimate from the registered f32 coefficient when that exists (it does not yet).
Usage: py q1773_levels_vs_exact_law.py (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp
import mpmath as mp

mp.mp.dps = 50
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "births_law.py"), encoding="utf-8").read().splitlines()
ns = {"PI": sp.pi}
for ln in src:
    if re.match(r"^(k1 =|c6 =|c10 =|f1[2-9] =|f20 =)", ln):
        exec(ln, ns)
Fs = {
    0: sp.Integer(1),
    1: ns["k1"],
    2: ns["k4"],
    3: ns["c6"],
    4: ns["c8"],
    5: ns["c10"],
    6: ns["f12"],
    7: ns["f14"],
    8: ns["f16"],
    9: ns["f18"],
    10: ns["f20"],
}
txt = open(os.path.join(H, "q1557_stdout_ORD30.txt"), encoding="utf-8").read()
for j, n in zip(range(11, 16), (22, 24, 26, 28, 30)):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    Fs[j] = sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi})
F = {j: mp.mpf(sp.N(Fs[j], 60)) for j in Fs}
u = 1 / mp.pi**2


def L1(y, c2=mp.mpf(-2) / 3):
    return c2 * y**2 - mp.mpf(20) / 3 * y**3 - mp.mpf(44) / 3 * y**4 - 8 * y**5


def L2(y):
    return (
        -mp.mpf(14) / 9 * y**4
        - mp.mpf(112) / 15 * y**5
        + mp.mpf(568) / 3 * y**6
        + mp.mpf(13616) / 9 * y**7
        + mp.mpf(33416) / 9 * y**8
        + mp.mpf(10880) / 3 * y**9
        + mp.mpf(3680) / 3 * y**10
    )


def S(t):
    return sum(F[j] * t**j for j in range(0, 16))


def err(t, levels=2, c2=mp.mpf(-2) / 3):
    x = u * t / 8
    y = x / (1 - x)
    approx = -mp.log(1 - x) + (L1(y, c2) / u if levels >= 1 else 0) + (L2(y) / u**2 if levels >= 2 else 0)
    ls = mp.log(S(t))
    return abs(ls - approx) / abs(ls)


t_small = 8 * mp.mpf("1e-4") / u
c1 = err(t_small) < mp.mpf("1e-14")
c2 = err(1 / mp.mpf(1) ** 2, 2, mp.mpf("-0.67")) > mp.mpf("1e-5")
print(
    "C1 error at x = 1e-4 below 1e-14: %s (%s); C2 changed L_1 coefficient gives an error above 1e-5 at rho = 1: %s (%s)"
    % (c1, mp.nstr(err(t_small), 3), c2, mp.nstr(err(mp.mpf(1), 2, mp.mpf("-0.67")), 3))
)
if not (c1 and c2):
    print("VOID: control failed")
    sys.exit(2)
res = {}
for rho in ("0.8", "1.0", "1.2"):
    t = 1 / mp.mpf(rho) ** 2
    x = u * t / 8
    res[rho] = (err(t, 2), err(t, 1), err(t, 0))
    print(
        "rho = %s: ln S = %s; absolute error of the levels 0..2: %s"
        % (rho, mp.nstr(mp.log(S(t)), 6), mp.nstr(res[rho][0] * abs(mp.log(S(t))), 4))
    )
    print(
        "rho = %s: x = %s; relative error of ln S against levels 0..2: %s; levels 0..1: %s; level 0 only: %s"
        % (rho, mp.nstr(x, 5), mp.nstr(res[rho][0], 4), mp.nstr(res[rho][1], 4), mp.nstr(res[rho][2], 4))
    )
lab = lambda q: "holds" if q else "does not hold"
p1 = res["1.0"][0] < mp.mpf("1e-7")
p2 = res["0.8"][0] > res["1.0"][0] > res["1.2"][0]
p3 = mp.mpf("1e-6") <= res["1.0"][1] <= mp.mpf("1e-4")
print(
    "P1 error of the levels 0..2 at rho = 1 below 1e-7: %s; P2 the error grows when rho decreases: %s; P3 error of the levels 0..1 at rho = 1 between 1e-6 and 1e-4: %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1773 done (labels by the script; no verdict word)")
