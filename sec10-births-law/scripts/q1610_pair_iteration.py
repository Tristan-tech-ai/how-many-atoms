"""q1610 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 64): a self-consistent edge pair from the exact blocks.
Reading 105: the interior atoms of the solved clean 120-atom block deviate from the formal bulk law by a change of the edge pair, delta d0 = +7.3e-08 and delta nu0 = -6.3e-08, which reduces the largest deviation in k = 30..70 by a factor of about 90. The frozen atoms beyond the block pin the deviation to zero at the block end, so the pair of the interior is the pair the exact ladder wants. This script iterates: solve the clean block N = 90 (NOUT = 170, level atom 140, 300 bits) with the law atoms 91..170 on the current pair, fit the deviations of the atoms k = 30..60 from the formal law of the current pair by (delta d0, delta nu0) as in q1609b, add the fit to the pair, and solve again from the previous solution. The start is the stored clean N = 90 solution (q1601_z_N90.json) with the pair of the formal law (nu0 = -0.093773586, d0 = +0.001159841).
Predictions written before the run:
 P1 after four rounds the fitted corrections are below 1e-10 in absolute value for both parameters;
 P2 the rms of the fit residual over k = 30..60 in the last round is below 1e-11;
 P3 the final pair differs from the limit pair of reading 74 (nu0 = -0.093773592, d0 = +0.001159849) by between 1e-08 and 2e-07 in each parameter;
 P4 every Newton solve of a round converges to a residual below 1e-60 within 8 updates.
Usage: py q1610_pair_iteration.py (about 25 minutes in the background, one process), the labels are printed by the same script.
"""

import os, sys, json, time, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
spec = importlib.util.spec_from_file_location("q1598m", os.path.join(here, "q1598_krawczyk_free90_120.py"))
q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(q)
m = q.m
from flint import arb, arb_mat

N = 90
n = 2 * N + 1
LO, HI = 30, 60


def set_block_clean(NN):
    m.NIN = NN
    m.NOUT = NN + 80
    m.KC = NN + 50


def fit(z, nu, d0):
    q.NU, q.D0 = nu, d0
    dt, wt = q.formal_tail(0, N + 1)
    d = np.array([float(z[k]) for k in range(N)])
    f = np.array([float(t) for t in dt])
    delta = d - f[:N]
    s = np.array([(f[k + 1] - f[k - 1]) / 2 if k > 0 else f[1] - f[0] for k in range(N)])
    idx = np.arange(LO - 1, HI)
    A = np.vstack([np.ones(len(idx)), s[idx]]).T
    coef, *_ = np.linalg.lstsq(A, delta[idx], rcond=None)
    res = delta[idx] - A @ coef
    return coef[0], coef[1], float(np.sqrt((res**2).mean())), float(np.abs(delta[idx]).max())


def run():
    t0 = time.time()
    z = [arb(s) for s in json.load(open(os.path.join(here, "q1601_z_N90.json")))]
    nu, d0 = q.NU, q.D0
    hist = []
    conv = True
    for r in range(5):
        dd0, dnu, rms, mx = fit(z, nu, d0)
        hist.append(dict(round=r, nu0=nu, d0=d0, fit_dd0=dd0, fit_dnu=dnu, rms=rms, maxdev=mx))
        print(
            "round %d: pair (%+.10f, %+.10f); fit corrections delta d0 %+.3e, delta nu0 %+.3e; rms of the rest %.2e; largest deviation %.2e [%.0fs]"
            % (r, nu, d0, dd0, dnu, rms, mx, time.time() - t0),
            flush=True,
        )
        nu, d0 = float(nu + dnu), float(d0 + dd0)
        if r == 4:
            break
        q.NU, q.D0 = nu, d0
        set_block_clean(N)
        WA = m.WallB(nu, d0, q.C0)  # the frozen atoms come from WallB itself
        z, h = q.newton_own(WA, z, n, 1e-70, 12)
        conv = conv and h[-1] < 1e-60 and len(h) - 1 <= 8
        hist[-1]["newton"] = [float(x) for x in h]
        print(
            "   Newton for the new pair: %d updates, residuals %s"
            % (len(h) - 1, ", ".join("%.1e" % x for x in h)),
            flush=True,
        )
    out = dict(hist=hist, final_pair=[nu, d0], newton_ok=conv)
    json.dump(out, open(os.path.join(here, "q1610_res.json"), "w"), indent=1)
    c = [abs(hist[-1]["fit_dd0"]), abs(hist[-1]["fit_dnu"])]
    p1 = max(c) < 1e-10
    p2 = hist[-1]["rms"] < 1e-11
    p3 = all(1e-8 < abs(a - b) < 2e-7 for a, b in ((nu, -0.093773592), (d0, 0.001159849)))
    p4 = conv

    def lab(x):
        return "holds" if x else "does not hold"

    print(
        "final pair (nu0, d0) = (%.12f, %.12f); differences to the limit pair of reading 74: %+.3e, %+.3e"
        % (nu, d0, nu + 0.093773592, d0 - 0.001159849)
    )
    print("P1 %s; P2 %s; P3 %s; P4 %s" % tuple(lab(x) for x in (p1, p2, p3, p4)))
    print("READING q1610 done (labels by the script; no verdict word)")


if __name__ == "__main__":
    run()
