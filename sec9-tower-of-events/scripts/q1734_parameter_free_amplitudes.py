"""q1734 (repair of q1733, written after its first output, kept as q1733_out_first_run_control_failed.txt, showed the control failing by the letter: the control compared the c28 law at nmax = 18 with the cell (14, 14) of q1687, which uses nmax = 14, so it was mis-specified. The repaired control compares with q1665 (law c28, pair of the c28 fit at nmax = 18, in-sample rms 1.518e-13 over K_b 83..130) within 20 percent. Predictions P1 to P4 and everything else are unchanged.)
Original docstring of q1733: (10 Oct 2026, READING, no verdict word): parameter-free births amplitudes. The wall pair is NOT fitted to any measured amplitude: it is the limit pair of the exact block, from the bulk series alone (q1729/q1732: model A (-0.0937736508471, +0.0011599160174), model B (-0.0937736508483, +0.0011599160185)). The bulk law is extended beyond f30 to NG = 24 with the coefficient model B of q1729 (ratio rho_15 + 0.0318 (j - 15)); the centre series e_6..e_18 is the stored one. The predicted amplitudes a_pred(K) are compared with the measured amplitudes: K_b 61..82 (arb re-measured, q1712_res.json, q1717_res.json), K_b 83..130 and K_b 131..142 (the events in fw_cap_par).
NOT BLIND: the measured amplitudes of the events were compared with fitted-pair predictions before, and the limit pair was computed from the bulk series (q1729) after readings 177 and 187 had been read; no amplitude entered the limit pair.
Predictions:
 P1 |a_pred - A| < 1.0e-12 for every K_b 61..82 (the Aitken extrapolation with a fitted pair gave 0.21e-12);
 P2 the rms of a_pred - A over K_b 83..130 is below 3e-13 (the fitted pair gives 3.1e-14);
 P3 |a_pred - A| < 5e-13 for every K_b 131..142 that exists;
 P4 the results of model A and model B differ by less than 3e-13 for every K_b.
Control (repaired): the same code with the births pair of the c28 fit (-0.0937736508344, +0.0011599160061) and the law NG = 14 reproduces the in-sample rms 1.518e-13 of q1665 over K_b 83..130 within 20 percent.
Usage: py q1733_parameter_free_amplitudes.py (from small_gap_q206/; about 2 minutes)."""

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
Kout = [K for K in range(131, 143) if B.measured(K) is not None]


def diffs(G, NG, pair, Ks):
    setlaw(G, NG)
    return {K: B.a_pred(K, pair[0], pair[1], ex) - meas(K) for K in Ks}


# control: c28 law with the c28 births pair against q1687 cell (14, 14)
dc = diffs(F15, 14, (-0.0937736508344, 0.0011599160061), Kmid + Kout)
rc = float(np.sqrt(np.mean([dc[K] ** 2 for K in Kmid])))
mc = max(abs(v) for v in dc.values())
PA = (-0.0937736508471, 0.0011599160174)
PB = (-0.0937736508483, 0.0011599160185)
GB = extended(0.0318)
GA = extended(15 and (F15[15] / F15[14] - F15[14] / F15[13]))
dA = diffs(GA, 24, PA, Klow + Kmid + Kout)
dB = diffs(GB, 24, PB, Klow + Kmid + Kout)
rms = lambda d, Ks: float(np.sqrt(np.mean([d[K] ** 2 for K in Ks])))
print(
    "control (law c28, births pair of the c28 fit): rms over K_b 83..130 %.3e, largest over all events %.3e (q1665 law c28: 1.518e-13)"
    % (rc, mc)
)
for name, d in (("model A", dA), ("model B", dB)):
    print(
        "%s: K_b 61..82 largest %.3e (at K_b %d); rms over K_b 83..130 %.3e; K_b 131..142 largest %.3e"
        % (
            name,
            max(abs(d[K]) for K in Klow),
            max(Klow, key=lambda K: abs(d[K])),
            rms(d, Kmid),
            max(abs(d[K]) for K in Kout),
        )
    )
print(
    "model B a_pred - A (1e-12), K_b 61..70: "
    + ", ".join("%d %+.2f" % (K, dB[K] * 1e12) for K in range(61, 71))
)
print("model B a_pred - A (1e-12), K_b 131..142: " + ", ".join("%d %+.2f" % (K, dB[K] * 1e12) for K in Kout))
lab = lambda q: "holds" if q else "does not hold"
p1 = all(max(abs(d[K]) for K in Klow) < 1.0e-12 for d in (dA, dB))
p2 = all(rms(d, Kmid) < 3e-13 for d in (dA, dB))
p3 = all(max(abs(d[K]) for K in Kout) < 5e-13 for d in (dA, dB))
p4 = max(abs(dA[K] - dB[K]) for K in dA) < 3e-13
ctl = abs(rc / 1.518e-13 - 1) < 0.20
print(
    "P1 K_b 61..82 below 1.0e-12: %s; P2 rms over K_b 83..130 below 3e-13: %s; P3 K_b 131..142 below 5e-13: %s; P4 models agree within 3e-13: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(p4), lab(ctl))
)
print("READING q1734 done (labels by the script; no verdict word)")
