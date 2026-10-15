r"""q1879_pr2e_kkt_edge_certificate_profile.py (14 Oct 2026, task PR2e, claim B3): ball-arithmetic certificate that the edge KKT function
K(x) = int phi(t - x) log G(t) dt is >= 0 for every x <= 0, for a test profile read from small_gap_q206/q1878_profiles.json,
    G(t) = Phi(-g - t) + sum_k nu_k phi(t - s_k),  s_n = -g exactly,
rounded to 9 decimals (the decimal strings are the certified object; the innermost atom is set to the string "-" + g). The method is that of q1876
(see its docstring for the duality step e^(C + h_1) <= M_A), generalised in two places:
 - the first atom need not sit at 0: from G >= nu_1 phi(t - s_1) and G <= Gmax = 1 + sum nu / sqrt(2 pi),
   |log G(t)| <= a2 + t^2 with a2 = max(log Gmax, s_1^2 - log(nu_1 / sqrt(2 pi)), 0); with t^2 <= 2u^2 + 2x^2 the truncation tails become
   2[(a2 + 2X^2) M0 + 2 M2], 2[(a2 + 2X^2) M1 + 2 M3], 2[(a2 + 2X^2)(M2 + M0) + 2(M4 + M2)];
 - in lemma T, N(x) = phi(D)((a2 + T^2)/D + 2/D^3), D = T - x, from |log G| <= a2 + t^2 and (T + v)^2 <= T^2 + v^2 for T <= 0, v >= 0.
X0 is an argument (default 16); lemma T needs -X0 <= 2T + g with T = -g - 1/nu_n.
Registered before the first run on the profile (14 Oct 2026), criteria as q1876:
 C1 the integrator returns finite balls everywhere (else VOID);
 C2 every cell of [-X0, 0] is certified (no certified negative value, no cell left at the minimum width 2^-24);
 C3 the premises of T hold (s_n <= -g on exact rationals, nu_1 > 0, -X0 <= 2T + g) and P(-X0) > N(-X0);
 C4 the dual value 2(sum nu - g) is a ball of radius below 1e-12 (it is printed to 10 digits and quoted to 5).
Verdict (last line): VERDICT q1879: PASS if C1 to C4 hold; FAIL if a certified negative value of K is found on [-X0, 0]; INCOMPLETE if C2 or C3
fails without a certified negative value; VOID if C1 or C4 fails or the script crashes.
Prediction of task PR2e (written before the run): PASS, with the dual value 3.98195 for the profile n = 5.
Selftest (corrupted profiles of n = 5, run before the real run): nu_1 lowered by 0.05 must give FAIL; s_n = -g + 1e-9 must give INCOMPLETE (premise of T
false while the cells stay positive, since the shift moves K by far less than the grid margin 1e-7 of q1878).
Registration for task PR2f (14 Oct 2026, before its run): profile q1880_profiles.json, n = 5, --x0 15; same criteria C1 to C4; the prediction of PR2f is a
certified dual value below 3.990 (the float optimiser already ended at 3.990008, so that prediction holds only if the 9-decimal profile rounds down).
Usage: py small_gap_q206/q1879_pr2e_kkt_edge_certificate_profile.py [--profile FILE] [--n 5] [--x0 16] [--registration-run] [--selftest]   (from temporal_within/)
"""

import json
import os
import sys
import time
from fractions import Fraction

import numpy as np
from flint import acb, arb, ctx
from scipy.special import ndtr

ctx.prec = 160
HERE = os.path.dirname(os.path.abspath(__file__))
L = 16
W0 = arb(1) / 4
WMIN = arb(2) ** -24
SQ2 = arb(2).sqrt()
SQ2PI = (2 * arb.pi()).sqrt()


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def load_profile(n):
    # 14 Oct 2026: --profile FILE (default q1878_profiles.json) added for task PR2f; the criteria are unchanged
    d = json.load(open(os.path.join(HERE, arg("--profile", "q1878_profiles.json"))))[str(n)]
    gs = "%.9f" % d["g"]
    atoms = [("%.9f" % s, "%.9f" % nu) for s, nu in zip(d["s"][:-1], d["nu"][:-1])] + [
        ("-" + gs, "%.9f" % d["nu"][-1])
    ]
    return gs, atoms


def float_min_K(gs, atoms, x0):
    z, w = np.polynomial.hermite.hermgauss(120)
    g = float(gs)
    s = np.array([float(a) for a, _ in atoms])
    nu = np.array([float(b) for _, b in atoms])
    xs = np.arange(-x0, 0.0001, 0.05)
    pts = xs[:, None] + np.sqrt(2) * z[None, :]
    G = ndtr(-g - pts) + sum(
        nk * np.exp(-((pts - sk) ** 2) / 2) / np.sqrt(2 * np.pi) for sk, nk in zip(s, nu)
    )
    return float((np.log(G) @ (w / np.sqrt(np.pi))).min())


def make(gs, atoms):
    g = arb(gs)
    at = [(arb(s), arb(n)) for s, n in atoms]

    def Gc(z):
        v = (acb(g) + z) / acb(SQ2)
        r = v.erfc() / 2
        for s, n in at:
            u = z - acb(s)
            r += acb(n) * (-(u * u) / 2).exp() / acb(SQ2PI)
        return r

    return g, at, Gc


def kint(Gc, x, j, c):
    xa = acb(x)

    def f(t, analytic):
        lg = Gc(t).log(analytic=analytic)
        u = t - xa
        w = acb(1) if j == 0 else (u if j == 1 else u * u - 1)
        return w * (-(u * u) / 2).exp() / acb(SQ2PI) * lg

    v = acb.integral(f, c - L, c + L, abs_tol=arb(2) ** -150, eval_limit=200000)
    return v.real


def tails(a2, X, R):
    phR = (-(R * R) / 2).exp() / SQ2PI
    M0 = (R / SQ2).erfc() / 2
    M1 = phR
    M2 = R * phR + M0
    M3 = (R * R + 2) * phR
    M4 = R**3 * phR + 3 * M2
    b = a2 + 2 * X * X
    return [2 * (b * M0 + 2 * M2), 2 * (b * M1 + 2 * M3), 2 * (b * (M2 + M0) + 2 * (M4 + M2))]


def Phi(z):
    return (-z / SQ2).erfc() / 2


def run(gs, atoms, x0):
    g, at, Gc = make(gs, atoms)
    lines = []
    nusum = sum((n for _, n in at), arb(0))
    Gmax = 1 + nusum / SQ2PI
    s1, nu1 = max(at, key=lambda z: float(z[0]))
    sn, nun = min(at, key=lambda z: float(z[0]))
    a2 = max(Gmax.log(), s1 * s1 - (nu1 / SQ2PI).log(), arb(0))
    prem_a = bool(nu1 > 0) and bool(nun > 0)
    tb = tails(a2, arb(x0) + 1, arb(L) - W0)
    cache = {}

    def K01(x):
        if x not in cache:
            xa = arb(x)
            cache[x] = (
                kint(Gc, xa, 0, x) + arb(0, tb[0].upper()),
                kint(Gc, xa, 1, x) + arb(0, tb[1].upper()),
            )
        return cache[x]

    void = False
    neg = None
    stuck = []
    ncell = 0
    minlb = None
    stack = [(-(i + 1) / 4.0, -i / 4.0) for i in reversed(range(int(x0 * 4)))]
    t0 = time.time()
    while stack:
        l, r = stack.pop()
        kl, dl = K01(l)
        kr, dr = K01(r)
        if not all(k.is_finite() for k in (kl, kr, dl, dr)):
            void = True
            break
        if kl < 0 or kr < 0:
            neg = l if kl < 0 else r
            break
        cell = arb((l + r) / 2, (r - l) / 2)
        k2 = kint(Gc, cell, 2, (l + r) / 2) + arb(0, tb[2].upper())
        if not k2.is_finite():
            void = True
            break
        M = abs(k2).upper()
        h = arb(r - l)
        lb = max(
            min(kl.lower(), (kl + dl * h - M * h * h / 2).lower()),
            min(kr.lower(), (kr - dr * h - M * h * h / 2).lower()),
        )
        ncell += 1
        if lb > 0:
            minlb = lb if minlb is None or lb < minlb else minlb
        elif h / 2 >= WMIN:
            m = (l + r) / 2
            stack.append((l, m))
            stack.append((m, r))
        else:
            stuck.append((l, r))
    lines.append(
        "cells certified or examined: %d; endpoint evaluations: %d; seconds %.0f"
        % (ncell, len(cache), time.time() - t0)
    )
    if minlb is not None:
        lines.append("smallest certified lower bound on a cell: %s" % minlb.str(5, radius=False))
    for x in (0.0, -0.25, -4.0, -float(x0)):
        if x in cache:
            lines.append("K(%s) in %s; K'(%s) in %s" % (x, cache[x][0].str(8), x, cache[x][1].str(8)))
    T = -g - 1 / nun
    Tp = -g - 2 / nun
    x = -arb(x0)
    D = T - x
    P = nun / (2 * Gmax) * (-((x + g) ** 2) / 4).exp() / (2 * arb.pi().sqrt()) * Phi(SQ2 * (Tp - (x - g) / 2))
    N = (-(D * D) / 2).exp() / SQ2PI * ((a2 + T * T) / D + 2 / D**3)
    prem_s = min(Fraction(sa) for sa, _ in atoms) <= -Fraction(gs)
    prem_x = bool(x <= 2 * T + g)
    c3 = prem_a and prem_s and prem_x and bool(P > N)
    lines.append(
        "tail lemma: s_n <= -g %s; nu > 0 %s; -X0 <= 2T + g = %s %s; P(-X0) = %s, N(-X0) = %s; a2 = %s"
        % (prem_s, prem_a, (2 * T + g).str(6), prem_x, P.str(5), N.str(5), a2.str(6))
    )
    dual = 2 * (nusum - g)
    c4 = dual.is_finite() and dual.rad() < arb("1e-12")
    lines.append("dual value 2(sum nu - g) = %s" % dual.str(10))
    if void:
        return lines, "VOID"
    if neg is not None:
        lines.append("certified negative value of K at x = %s: %s" % (neg, cache[neg][0].str(6)))
        return lines, "FAIL"
    if not c4:
        return lines, "VOID"
    if stuck or not c3:
        lines.append("cells left at minimum width: %d" % len(stuck))
        return lines, "INCOMPLETE"
    return lines, "PASS"


def selftest(n, x0):
    gs, atoms = load_profile(n)
    bad1 = [(atoms[0][0], "%.9f" % (float(atoms[0][1]) - 0.05))] + atoms[1:]
    bad2 = atoms[:-1] + [("%.9f" % (-float(gs) + 1e-9), atoms[-1][1])]
    ok = True
    for name, at, want in [("nu_1 lowered by 0.05", bad1, "FAIL"), ("s_n = -g + 1e-9", bad2, "INCOMPLETE")]:
        lines, v = run(gs, at, x0)
        print("selftest %-22s -> %s (want %s); %s" % (name, v, want, lines[-1]))
        ok = ok and v == want
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    n = arg("--n", 5)
    x0 = arg("--x0", 16)
    if "--registration-run" in sys.argv:
        print(__doc__.split("Usage:")[0])
        print("registration run: criteria printed, nothing evaluated")
        sys.exit(0)
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest(n, x0) else 1)
    try:
        gs, atoms = load_profile(n)
        print("profile n = %d (9 decimals): g = %s; atoms %s" % (n, gs, atoms))
        print(
            "float check of the rounded profile: min K on the grid [-%d, 0] step 0.05 = %.3e"
            % (x0, float_min_K(gs, atoms, x0))
        )
        lines, v = run(gs, atoms, x0)
    except Exception as e:
        print("crash:", repr(e))
        print("VERDICT q1879: VOID")
        sys.exit(2)
    for l in lines:
        print(l)
    print("VERDICT q1879: %s" % v)
