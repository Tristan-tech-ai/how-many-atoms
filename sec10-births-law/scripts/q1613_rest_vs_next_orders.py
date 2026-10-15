"""q1613 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 67): is the smooth rest of the interior of the exact blocks the truncation of the bulk law at c20?
q1612 found that the rest of 5e-11 (after the edge pair is removed on k = 30..60) is smooth in k (cubic remainder 13 to 17 percent of the rest, almost no sign changes between consecutive atoms) and is the same function for the blocks N = 90 and N = 120 (correlation 0.9487). The formal law of the blocks is the bulk law through c20 (NG = 10 in q1583_ball_rows.py). The exact coefficients f22 = -2.420924146814851e-10 and f24 = -8.973800969707301e-11 are known (q1586_f24_prereg.txt). This script builds the formal depths d_k = dlaw(rho_k) + d0 with rho_k from Nlaw(rho_k) = k + nu0 for the orders NG = 10, 11 and 12 at the formal pair (-0.093773586, +0.001159841), forms e_k = d^(12)_k - d^(10)_k, subtracts it from the block deviations (the deviation from the law through c24 is delta_k - e_k), refits the pair on k = 30..60 and compares the rest.
Predictions written before the run:
 P1 control: the float formal depths with NG = 10 agree with the arb formal depths of q1598 (formal_tail) for k = 1..120 to 1e-12;
 P2 the rest of the deviation from the law through c24 has an rms below 0.5 of the rest from the law through c20, for both blocks N = 90 and N = 120 (k = 30..60);
 P3 the pair fitted against the law through c24 has a combination d0 + 0.95 nu0 that is closer to that of the Aitken limit pair of q1564 (nu0 = -0.093773592, d0 = +0.001159849) than the combination of the pair fitted against the law through c20, for both blocks;
 P4 the rms of e_k after its own pair fit on k = 30..60 lies between 1e-11 and 3e-10 (the neglected orders are of the size of the rest).
Usage: py q1613_rest_vs_next_orders.py (about 1 minute)."""

import os, sys, json, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
spec = importlib.util.spec_from_file_location("q1598m", os.path.join(here, "q1598_krawczyk_free90_120.py"))
q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(q)
from flint import arb

PI = np.pi
NU, D0 = -0.093773586, 0.001159841
k1 = 1 / (8 * PI**2)
k4 = 1 / (64 * PI**4) - 1 / (96 * PI**2)
c6 = 1 / (512 * PI**6) - 13 / (768 * PI**4)
c8 = 1 / (4096 * PI**8) - 17 / (1536 * PI**6) - 1 / (3072 * PI**4)
c10 = 1 / (32768 * PI**10) - 29 / (6144 * PI**8) - 109 / (368640 * PI**6)
f12 = -37 / (442368 * PI**6) - 617 / (393216 * PI**10) + 1 / (262144 * PI**12) + 229 / (327680 * PI**8)
f14 = (
    -33223 / (123863040 * PI**8)
    - 1393 / (3145728 * PI**12)
    + 1 / (2097152 * PI**14)
    + 3819 / (2621440 * PI**10)
)
f16 = (
    -16507 / (424673280 * PI**8)
    - 43261 / (123863040 * PI**10)
    - 175 / (1572864 * PI**14)
    + 1 / (16777216 * PI**16)
    + 42631 / (31457280 * PI**12)
)
f18 = (
    -54007 / (293601280 * PI**10)
    - 53549 / (165150720 * PI**12)
    - 215 / (8388608 * PI**16)
    + 1 / (134217728 * PI**18)
    + 71767 / (83886080 * PI**14)
)
f20 = (
    -64764727 / (190253629440 * PI**12)
    - 395711 / (13589544960 * PI**10)
    - 11659519 / (31708938240 * PI**14)
    - 2963 / (536870912 * PI**18)
    + 1 / (1073741824 * PI**20)
    + 848207 / (2013265920 * PI**16)
)
F = [1.0, k1, k4, c6, c8, c10, f12, f14, f16, f18, f20, -2.420924146814851e-10, -8.973800969707301e-11]


def make(NG):
    g = [1.0]
    for n in range(1, NG + 1):
        g.append(-sum(F[j] * g[n - j] for j in range(1, n + 1)))
    dl = (
        lambda r: 16
        * PI**2
        * (g[0] * r**3 / 3 + sum(g[n] * r ** (3 - 2 * n) / (3 - 2 * n) for n in range(1, NG + 1)))
    )
    Nl = (
        lambda r: 16
        * PI**2
        * (
            g[0] * r**4 / 4
            + g[1] * r**2 / 2
            + g[2] * np.log(r)
            + sum(g[n] * r ** (4 - 2 * n) / (4 - 2 * n) for n in range(3, NG + 1))
        )
    )
    dN = lambda r: 16 * PI**2 * r * sum(g[n] * r ** (2 - 2 * n) for n in range(0, NG + 1))

    def depths(K):
        t = np.arange(1, K + 1) + NU
        r = (np.maximum(t, 1.0) / (4 * PI**2)) ** 0.25 + 0.3
        for _ in range(100):
            r = r - (Nl(r) - t) / dN(r)
        return dl(r) + D0

    return depths


dep = {NG: make(NG)(121) for NG in (10, 11, 12)}
q.NU, q.D0 = NU, D0
dt, wt = q.formal_tail(0, 121)
farb = np.array([float(t) for t in dt])
ctl = float(np.abs(dep[10] - farb).max())
print(
    "control: largest difference of the float NG = 10 depths from the arb formal depths for k = 1..121: %.2e"
    % ctl
)
e = dep[12] - dep[10]
print(
    "e_k = d^(12) - d^(10) at k = 1, 10, 30, 45, 60, 90, 120: "
    + " ".join("%+.2e" % e[k - 1] for k in (1, 10, 30, 45, 60, 90, 120))
)


def pairfit(delta, s, lo, hi):
    idx = np.arange(lo - 1, hi)
    A = np.vstack([np.ones(len(idx)), s[idx]]).T
    coef, *_ = np.linalg.lstsq(A, delta[idx], rcond=None)
    return coef, delta[idx] - A @ coef


LIM = (-0.093773592, 0.001159849)
comb = lambda nu, d0: d0 + 0.95 * nu
res = {}
for N in (90, 120):
    z = [arb(s) for s in json.load(open(os.path.join(here, "q1601_z_N%d.json" % N)))]
    d = np.array([float(z[k]) for k in range(N)])
    s = np.array([(farb[k + 1] - farb[k - 1]) / 2 if k > 0 else farb[1] - farb[0] for k in range(N)])
    delta20 = d - dep[10][:N]
    delta24 = d - dep[12][:N]
    c20, r20 = pairfit(delta20, s, 30, 60)
    c24, r24 = pairfit(delta24, s, 30, 60)
    ce, re_ = pairfit(e[:N], s, 30, 60)
    rm = lambda x: float(np.sqrt((x**2).mean()))
    n20 = (NU + c20[1], D0 + c20[0])
    n24 = (NU + c24[1], D0 + c24[0])
    g20 = abs(comb(*n20) - comb(*LIM))
    g24 = abs(comb(*n24) - comb(*LIM))
    print(
        "N = %d: law c20: pair (%+.9f, %+.9f), rest rms %.3e; law c24: pair (%+.9f, %+.9f), rest rms %.3e (ratio %.3f); e_k own-pair rest rms %.3e; combination distance to the limit pair %.3e (c20) and %.3e (c24)"
        % (N, n20[0], n20[1], rm(r20), n24[0], n24[1], rm(r24), rm(r24) / rm(r20), rm(re_), g20, g24)
    )
    res[N] = dict(ratio=rm(r24) / rm(r20), g20=g20, g24=g24, erms=rm(re_))
lab = lambda p: "holds" if p else "does not hold"
p1 = ctl < 1e-12
p2 = all(res[N]["ratio"] < 0.5 for N in res)
p3 = all(res[N]["g24"] < res[N]["g20"] for N in res)
p4 = all(1e-11 <= res[N]["erms"] <= 3e-10 for N in res)
print(
    "P1 control below 1e-12: %s; P2 rest ratio below 0.5 for both blocks: %s; P3 combination closer to the limit pair with c24: %s; P4 rms of e_k after its own pair fit in [1e-11, 3e-10]: %s"
    % (lab(p1), lab(p2), lab(p3), lab(p4))
)
print("READING q1613 done (labels by the script; no verdict word)")
