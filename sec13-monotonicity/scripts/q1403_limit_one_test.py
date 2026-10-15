"""q1403 reading (9 Oct 2026, EXPLORATORY, no verdict; py q1403_limit_one_test.py from temporal_within/small_gap_q206/): can the stage-f0.97 sequences (K = 62..82) of P/N, of the edge-only ratio R_e and of rho be described with the limit 1 (for R_e and P/N) , i.e. 1 + a K^-p with a and p free? Compare the rms with the free-limit fit L + a K^-p (three parameters) and report p and a. The motivation is the earlier finding (MARGIN_PROOF_PLAN.md, amendments 1841) that the margin t_1 - 1 of the diagonal factor falls like 1/m. Expected before the run: the forced-limit fit has a clearly larger rms (a factor 5 or more) than the free fit, i.e. the data prefer L > 1; if the factor is below 3 the limit 1 cannot be excluded. Control: synthetic 1.0 + 3/K^0.5 is fitted by the forced-limit fit with rms below 1e-8 and a synthetic 1.04 + 1/K^0.7 gives a forced-limit rms above 1e-4."""

import json
import numpy as np
from scipy.optimize import least_squares
import importlib.util

spec = importlib.util.spec_from_file_location("q1401", "q1401_edge_fit_sensitivity.py")
src = open("q1401_edge_fit_sensitivity.py").read().split("Kc = np.arange")[0]
ns = {}
exec(src, ns)
quantities = ns["quantities"]


def forced(K, Y, L):
    best = None
    for p0 in (0.1, 0.3, 0.6, 1.0, 2.0):
        f = lambda q: L + q[0] * K ** (-q[1]) - Y
        r = least_squares(f, [Y[0] - L if Y[0] > L else 0.05, p0] if False else [(Y[0] - L) * K[0] ** p0, p0])
        rms = float(np.sqrt(np.mean(r.fun**2)))
        if best is None or rms < best[0]:
            best = (rms, r.x[0], r.x[1])
    return best


def free(K, Y):
    best = None
    for p0 in (0.3, 0.7, 1.5):
        for L0 in (0.9 * Y.min(), 1.0, Y.min() - 0.01):
            r = least_squares(
                lambda q: q[0] + q[1] * K ** (-abs(q[2])) - Y, [L0, (Y[0] - L0) * K[0] ** p0, p0]
            )
            rms = float(np.sqrt(np.mean(r.fun**2)))
            if best is None or rms < best[0]:
                best = (rms, r.x[0], r.x[1], abs(r.x[2]))
    return best


Kc = np.arange(62, 83, 2, float)
c1 = forced(Kc, 1.0 + 3 / Kc**0.5, 1.0)[0] < 1e-8
c2 = forced(Kc, 1.04 + 1 / Kc**0.7, 1.0)[0] > 1e-4
print(
    "control forced-limit recovers L = 1 sequence:",
    "OK" if c1 else "FAILED",
    "; rejects L = 1.04 sequence:",
    "OK" if c2 else "FAILED",
)
data = [quantities(f"q1388_jb_K{K}_f0.97.json") for K in range(62, 83)]
# quantities returns (K, R_e, rho, odd); P/N is read from the q1389 output
import re

pn = {}
cur = None
for line in open("q1389_out_fixed.txt", encoding="utf-8"):
    m = re.match(r"K (\d+) \S+ A \S+ \((odd|even)", line)
    if m:
        cur = int(m.group(1))
        continue
    m = re.match(r"\s+D1\s+: .*ratio (\S+);", line)
    if m and cur is not None and cur not in pn:
        pn[cur] = float(m.group(1))
for odd in (False, True):
    sel = [t for t in data if t[3] == odd]
    K = np.array([t[0] for t in sel], float)
    for nm, Y in (("P/N", np.array([pn[t[0]] for t in sel])), ("R_e", np.array([t[1] for t in sel]))):
        rf, a, p = forced(K, Y, 1.0)[0], forced(K, Y, 1.0)[1], forced(K, Y, 1.0)[2]
        fr = free(K, Y)
        print(
            f"{'odd' if odd else 'even'} {nm}: limit 1 forced: a {a:.4f}, p {p:.4f}, rms {rf:.2e} | free limit: L {fr[1]:.5f}, p {fr[3]:.3f}, rms {fr[0]:.2e} | rms ratio {rf / fr[0]:.1f}"
        )
print("READING q1403 done")
