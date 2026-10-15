"""q1738 (10 Oct 2026, READING, no verdict word): selection swap for the fit-free prediction of readings 191 to 194. In q1736 the pair (nmax, NG) was chosen on the lower births K_b 61..82 and checked on K_b 83..143 (q1737). Here the roles are exchanged: every pair nmax in {16, 18, 20}, NG in 14..24 is scored on K_b 83..143 alone (score = rms of a_pred - A over K_b 83..130 plus the largest |a_pred - A| over K_b 131..143), the best pair is taken, and its size on the lower births K_b 61..82 is the check. Same setup as q1737: limit pair of model B, bulk coefficient model B (increment 0.0318), stored centre series to order nmax, re-measured lower births (q1712, q1717), no fitted number.
NOT BLIND: readings 191 to 194 were read, and eight cells of this table (nmax 18 and 20, NG 17, 18, 19, 24) were printed by q1737 before these predictions were written to the registration text; the other cells were not seen.
Predictions (written at 13:05):
 P1 for nmax = 20 the NG with the smallest rms over K_b 83..130 lies in 17..20;
 P2 for nmax = 18 the NG with the smallest rms over K_b 83..130 lies in 16..19;
 P3 at the pair chosen on K_b 83..143 the largest size over K_b 61..82 is below 1e-12.
Extra facts printed without a label: the size at K_b 144 for the chosen pair (K_b 144 is no part of the score) when the event exists in the stored measurements.
Usage: py q1738_selection_swap.py (from small_gap_q206/; about 2 minutes)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
E = {n: float(S20["e%d" % n]) for n in range(2, 21)}


def parse(fn, ns):
    t = open(os.path.join(H, fn), encoding="utf-8").read()
    out = {}
    for n in ns:
        m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, t, re.M)
        out[n] = float(
            mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30))
        )
    return out


f = parse("q1557_stdout_ORD30.txt", (26, 28, 30))
F15 = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28], f[30]]


def extended(inc, top=24):
    G = list(F15)
    r15 = G[15] / G[14]
    for j in range(16, top + 1):
        G.append(G[-1] * (r15 + (j - 15) * inc))
    return G


def setlaw(G, NG):
    B.F = G[: NG + 1]
    B.NG = NG
    B.g = [1.0]
    for n in range(1, NG + 1):
        B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))


R = json.load(open(os.path.join(H, "q1712_res.json")))
R.update(json.load(open(os.path.join(H, "q1717_res.json"))))
NEW = {int(k): float(v["new"]) for k, v in R.items() if v.get("status") == "ok"}


def meas(K):
    return B.measured(K) if K >= 83 else NEW[K]


ex = [E[n] for n in range(6, 19)]
Klow = list(range(61, 83))
Kmid = [K for K in range(83, 131) if B.measured(K) is not None]
Kout = [K for K in range(131, 144) if B.measured(K) is not None]
rms = lambda d, Ks: float(np.sqrt(np.mean([d[K] ** 2 for K in Ks])))


def diffs(G, NG, pair, Ks):
    setlaw(G, NG)
    return {K: B.a_pred(K, pair[0], pair[1], ex) - meas(K) for K in Ks}


GB = extended(0.0318)
PB = (-0.0937736508483, 0.0011599160185)
print(
    "events used: K_b 61..82 re-measured (%d), K_b 83..130 (%d events), K_b 131..143 (%d events)"
    % (len(Klow), len(Kmid), len(Kout))
)
tab = {}
for nmax in (16, 18, 20):
    ex[:] = [E[n] for n in range(6, nmax + 1)]
    for NG in range(14, 25):
        d = diffs(GB, NG, PB, Klow + Kmid + Kout)
        tab[(nmax, NG)] = dict(
            mid=rms(d, Kmid), out=max(abs(d[K]) for K in Kout), low=max(abs(d[K]) for K in Klow), d=d
        )
        print(
            "nmax %d NG %2d: rms K_b 83..130 %.2e; largest K_b 131..143 %.2e; score %.2e; largest K_b 61..82 %.2e"
            % (
                nmax,
                NG,
                tab[(nmax, NG)]["mid"],
                tab[(nmax, NG)]["out"],
                tab[(nmax, NG)]["mid"] + tab[(nmax, NG)]["out"],
                tab[(nmax, NG)]["low"],
            )
        )
best_rms = {nm: min(range(14, 25), key=lambda NG: tab[(nm, NG)]["mid"]) for nm in (16, 18, 20)}
print(
    "NG with the smallest rms over K_b 83..130: nmax 16 -> %d, nmax 18 -> %d, nmax 20 -> %d"
    % (best_rms[16], best_rms[18], best_rms[20])
)
pair = min(tab, key=lambda k: tab[k]["mid"] + tab[k]["out"])
t = tab[pair]
print(
    "pair chosen on K_b 83..143 alone: nmax %d, NG %d (score %.2e); its largest size over K_b 61..82: %.2e"
    % (pair[0], pair[1], t["mid"] + t["out"], t["low"])
)
try:
    ex[:] = [E[n] for n in range(6, pair[0] + 1)]
    setlaw(GB, pair[1])
    m144 = B.measured(144)
    if m144 is not None:
        print(
            "extra fact: K_b 144 a_pred - A for the chosen pair: %+.3e"
            % (B.a_pred(144, PB[0], PB[1], ex) - m144)
        )
    else:
        print("extra fact: no stored measurement for K_b 144 in births_law.measured")
except Exception as e:
    print("extra fact: K_b 144 not evaluated (%s)" % type(e).__name__)
lab = lambda q: "holds" if q else "does not hold"
p1 = 17 <= best_rms[20] <= 20
p2 = 16 <= best_rms[18] <= 19
p3 = t["low"] < 1e-12
print(
    "P1 nmax 20 best NG in 17..20: %s; P2 nmax 18 best NG in 16..19: %s; P3 chosen pair on K_b 61..82 below 1e-12: %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1738 done (labels by the script; no verdict word)")
