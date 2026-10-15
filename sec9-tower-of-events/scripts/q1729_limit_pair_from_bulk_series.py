"""q1729 (10 Oct 2026, READING, no verdict word): route to the limit pair from the bulk series alone. Reading 187 (q1728): the one-order step of the self-consistent pair of the exact block is minus the two-parameter fit (a + b s_k on the window k = 30..60) of the added bulk term D_NG = formal(NG + 1) - formal(NG) at the pair of the solved state (q1708_z_NG14_N90.json): step in nu0 = -b, in d0 = -a. Here the formal laws are rebuilt in mpmath (50 digits) from the closed forms of F_1..F_15 (q1583, q1557_stdout_ORD30.txt), the coefficients beyond f30 are extended by a model of their ratio F_j / F_(j-1) = rho_15 + (j - 15) inc, and the steps for NG = 15 .. 24 are summed onto the c30 pair of the exact block (q1705_res_NG15_N90.json).
Two models of the increment: A, inc = rho_15 - rho_14 (the last observed increment); B, inc = 0.0318 (reading 151). The series is asymptotic, so the sum is stopped at the term with the smallest step (reported).
NOT BLIND: readings 177 and 187 were read. Predictions (registration text of 12:52, written before this script existed):
 P1 control: the steps 12 -> 13, 13 -> 14 and 14 -> 15 rebuilt in mpmath agree with the observed steps of the ladder pairs (-2.084e-10, -8.424e-11, -3.673e-11 in nu0) within 1 percent;
 P2 the summed tail from NG = 15 on is between -6e-11 and -1e-11 in nu0 for both models, and the limit pair lies within 1e-11 of the c30 births pair (-0.0937736508428, +0.0011599160135) in both parameters;
 P3 the two models give limits in nu0 that differ by less than 3e-12.
Usage: py q1729_limit_pair_from_bulk_series.py (a few minutes)."""

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
for name, inc in (("A", None), ("B", 0.0318)):
    G = extend(0.0) if False else None
    base = list(F)
    rr = [None] + [F[j] / F[j - 1] for j in range(1, 16)]
    incA = rr[15] - rr[14]
    incx = incA if inc is None else inc
    G = list(F)
    for j in range(16, 27):
        G.append(G[-1] * (rr[15] + (j - 15) * incx))
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
p1 = all(abs(st[ng][0] / obs[ng] - 1) < 0.01 for ng in (13, 14, 15))
print(
    "control: rebuilt steps 12->13, 13->14, 14->15: %s against observed %s (ratios %s)"
    % (
        ", ".join("%+.4e" % st[ng][0] for ng in (13, 14, 15)),
        ", ".join("%+.4e" % obs[ng] for ng in (13, 14, 15)),
        ", ".join("%.4f" % (st[ng][0] / obs[ng]) for ng in (13, 14, 15)),
    )
)
lim = {}
for name in ("A", "B"):
    steps = res[name][0]
    mags = [abs(steps[ng][0]) for ng in range(16, 25)]
    imin = int(np.argmin(mags))
    last = 16 + imin
    tail_nu = sum(steps[ng][0] for ng in range(16, last + 1))
    tail_d0 = sum(steps[ng][1] for ng in range(16, last + 1))
    lim[name] = (P[15][0] + tail_nu, P[15][1] + tail_d0, tail_nu, tail_d0, last)
    print(
        "model %s: steps 16..%d summed (smallest step at NG = %d): tail nu0 %+.3e, d0 %+.3e; limit pair (%.13f, %.13f); minus the c30 births pair %+.3e, %+.3e"
        % (
            name,
            last,
            last,
            tail_nu,
            tail_d0,
            lim[name][0],
            lim[name][1],
            lim[name][0] - BIRTHS[0],
            lim[name][1] - BIRTHS[1],
        )
    )
p2 = all(
    -6e-11 < lim[n][2] < -1e-11 and abs(lim[n][0] - BIRTHS[0]) < 1e-11 and abs(lim[n][1] - BIRTHS[1]) < 1e-11
    for n in ("A", "B")
)
p3 = abs(lim["A"][0] - lim["B"][0]) < 3e-12
print(
    "P1 control within 1 percent: %s; P2 tail in [-6e-11, -1e-11] and limit within 1e-11 of the births pair: %s; P3 models differ by less than 3e-12: %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1729 done (labels by the script; no verdict word)")
