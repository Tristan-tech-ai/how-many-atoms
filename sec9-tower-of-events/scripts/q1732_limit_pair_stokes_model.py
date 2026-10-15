"""q1732 (10 Oct 2026, READING, no verdict word): the extension of the bulk coefficients beyond f30 with a model grounded in the Stokes constant. The observed ratios F_j / F_(j-1) for j = 11..15 are 0.3386, 0.3707, 0.4036, 0.4336, 0.4620; a straight line through the origin gives r_j = (j + c) / A. For factorial growth the constant A is the rate of the exponentially small ambiguity: the recorded decay of ln Da is 0.2087 per unit v^2 (q1673) with v = 4 pi rho, so exp(-0.2087 v^2) = exp(-A rho^2) with A = 0.2087 * 16 pi^2 = 32.96.
NOT BLIND: q1729 and q1730 were read; the observation r_j = (j + 0.24) / 32.96 came from a hand calculation on the five ratios.
Predictions:
 P1 the least-squares fit r_j = (j + c) / A on j = 11..15 gives A within 5 percent of 32.96 and c between 0 and 0.5;
 P2 with model E (r_j = (j + c) / A for j >= 16, the fitted A and c) the limit pair lies within 3e-12 of the limit of model A in nu0 and in d0;
 P3 the control of q1729 (steps 12 -> 13, 13 -> 14, 14 -> 15 within 1 percent) holds again.
Usage: py q1732_limit_pair_stokes_model.py (a few minutes)."""

import os, sys, json, re
import numpy as np
import mpmath as mp

sys.dont_write_bytecode = True
mp.mp.dps = 50
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
Z = json.load(open(os.path.join(here, "q1708_z_NG14_N90.json")))
nu, d0 = mp.mpf(repr(Z["nu_used"])), mp.mpf(repr(Z["d0_used"]))
pi = mp.pi
F = [
    mp.mpf(1),
    1 / (8 * pi**2),
    1 / (64 * pi**4) - 1 / (96 * pi**2),
    1 / (512 * pi**6) - 13 / (768 * pi**4),
    1 / (4096 * pi**8) - 17 / (1536 * pi**6) - 1 / (3072 * pi**4),
    1 / (32768 * pi**10) - 29 / (6144 * pi**8) - 109 / (368640 * pi**6),
]
txt = open(os.path.join(here, "q1557_stdout_ORD30.txt"), encoding="utf-8").read()
for n in range(12, 31, 2):
    m = re.search(r"^\s*f%d = (.*?)\s{2,}= (\S+)\s*$" % n, txt, re.M)
    F.append(eval(m.group(1).replace("pi", "PI"), {"PI": pi}))
assert len(F) == 16


def extend(inc):
    G = list(F)
    r = [None] + [G[j] / G[j - 1] for j in range(1, 16)]
    for j in range(16, 27):
        G.append(G[-1] * (r[15] + (j - 15) * inc if False else (G[15] / G[14]) + (j - 15) * inc))
    return G


def law(G, NG):
    g = [mp.mpf(1)]
    for n in range(1, NG + 1):
        g.append(-sum(G[j] * g[n - j] for j in range(1, n + 1)))
    dl = (
        lambda r: 16
        * pi**2
        * (g[0] * r**3 / 3 + sum(g[n] * r ** (3 - 2 * n) / (3 - 2 * n) for n in range(1, NG + 1)))
    )
    Nl = (
        lambda r: 16
        * pi**2
        * (
            g[0] * r**4 / 4
            + g[1] * r**2 / 2
            + g[2] * mp.log(r)
            + sum(g[n] * r ** (4 - 2 * n) / (4 - 2 * n) for n in range(3, NG + 1))
        )
    )
    dN = lambda r: 16 * pi**2 * r * sum(g[n] * r ** (2 - 2 * n) for n in range(0, NG + 1))
    return dl, Nl, dN


KS = list(range(29, 62))  # atoms 29..61 (the window 30..60 and its neighbours for s_k)
rho_cache = {}


def depths(G, NG):
    dl, Nl, dN = law(G, NG)
    out = {}
    for k in KS:
        t = k + nu
        r = (t / (4 * pi**2)) ** mp.mpf("0.25") + mp.mpf("0.3")
        if (NG - 1, k) in rho_cache:
            r = rho_cache[(NG - 1, k)]
        for _ in range(60):
            dr = (Nl(r) - t) / dN(r)
            r = r - dr
            if abs(dr) < mp.mpf(10) ** -45:
                break
        rho_cache[(NG, k)] = r
        out[k] = dl(r) + d0
    return out


def fit(Dn, s):
    idx = list(range(30, 61))
    A = np.vstack([np.ones(len(idx)), [float(s[k]) for k in idx]]).T
    y = np.array([float(Dn[k]) for k in idx])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


obs = {}
P = {
    ng: json.load(open(os.path.join(here, fn)))["final_pair"]
    for ng, fn in (
        (12, "q1698_res_NG12_N90.json"),
        (13, "q1714_res_NG13_N90.json"),
        (14, "q1698_res_NG14_N90.json"),
        (15, "q1705_res_NG15_N90.json"),
    )
}
for ng in (13, 14, 15):
    obs[ng] = P[ng][0] - P[ng - 1][0]
BIRTHS = (-0.0937736508428, 0.0011599160135)
res = {}
rj = np.array([float(F[j] / F[j - 1]) for j in range(11, 16)])
jj = np.arange(11, 16)
Afit_inv, cA = np.polyfit(jj, rj, 1)
Afit = 1 / Afit_inv
cfit = cA / Afit_inv
print(
    "fit r_j = (j + c) / A on j = 11..15: A = %.3f, c = %.3f (0.2087 * 16 pi^2 = %.3f); residuals %s"
    % (Afit, cfit, 0.2087 * 16 * np.pi**2, " ".join("%+.4f" % x for x in (rj - Afit_inv * jj - cA)))
)
for name, inc in (("A", None), ("B", 0.0318), ("E", "E")):
    G = extend(0.0) if False else None
    base = list(F)
    rr = [None] + [F[j] / F[j - 1] for j in range(1, 16)]
    incA = rr[15] - rr[14]
    incx = incA if inc is None else (Afit_inv if inc == "E" else inc)
    G = list(F)
    for j in range(16, 27):
        G.append(G[-1] * ((Afit_inv * j + cA) if inc == "E" else (rr[15] + (j - 15) * incx)))
    rho_cache.clear()
    dep = {NG: depths(G, NG) for NG in range(12, 26)}
    s = {k: (dep[14][k + 1] - dep[14][k - 1]) / 2 for k in range(30, 61)}
    steps = {}
    for NG in range(12, 25):
        Dn = {k: dep[NG + 1][k] - dep[NG][k] for k in range(30, 61)}
        a, b = fit(Dn, s)
        steps[NG + 1] = (-b, -a)
    res[name] = (steps, incx, [float(G[j] / G[j - 1]) for j in range(11, 27)])
    print(
        "model %s: increment %.5f; ratios F_j/F_(j-1) for j = 11..26: %s"
        % (name, incx, " ".join("%.4f" % x for x in res[name][2]))
    )
    print(
        "  steps in nu0 (NG-1 -> NG): " + ", ".join("%d %+.3e" % (ng, steps[ng][0]) for ng in sorted(steps))
    )
lab = lambda q: "holds" if q else "does not hold"
st = res["A"][0]
p3 = all(abs(st[ng][0] / obs[ng] - 1) < 0.01 for ng in (13, 14, 15))
lim = {}
for name in ("A", "B", "E"):
    steps = res[name][0]
    mags = [abs(steps[ng][0]) for ng in range(16, 25)]
    last = 16 + int(np.argmin(mags))
    tn = sum(steps[ng][0] for ng in range(16, last + 1))
    td = sum(steps[ng][1] for ng in range(16, last + 1))
    lim[name] = (P[15][0] + tn, P[15][1] + td)
    print(
        "model %s: limit pair (%.13f, %.13f); tail nu0 %+.3e, d0 %+.3e; minus the c30 births pair %+.3e, %+.3e"
        % (name, lim[name][0], lim[name][1], tn, td, lim[name][0] - BIRTHS[0], lim[name][1] - BIRTHS[1])
    )
p1 = abs(Afit / 32.96 - 1) < 0.05 and 0 <= cfit <= 0.5
p2 = abs(lim["E"][0] - lim["A"][0]) < 3e-12 and abs(lim["E"][1] - lim["A"][1]) < 3e-12
print(
    "P1 A within 5 percent of 32.96 and c in [0, 0.5]: %s; P2 model E within 3e-12 of model A: %s; P3 control within 1 percent: %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1732 done (labels by the script; no verdict word)")
