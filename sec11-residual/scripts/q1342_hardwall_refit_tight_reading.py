"""q1342 (8 Oct 2026, EXPLORATORY reading, amendment 2116; no item, no verdict): refit of the hard-wall closure R / Da = c0(k) + c1 / v^2 (+ c2 / v^4) with the repaired prediction a_med (run through tight_run.py), on ALL located hard-wall events: q1304 (K_b 25-37; k = 0.02, 0.05, 0.1, 0.25), q1305 (K_b 29-36; k = 0.5, 0.75), q1306 (K_b 14-24, 39, 40; k = 0.1, 0.25, 0.5, 0.75), q1321 (k = 0.02, 0.05, 0.35), q1324 (k = 0.06, 0.075, 0.09), q1326 (k = 0.052 to 0.058). Inclusion as in q1307: located, R > 0, |a_quintic - a_septic| 1e9 <= 0.02 R.
Fits by linear least squares on the relative misfit (R_model / R - 1), as q1319 and q1320: per exponent two- and three-term; one common c1 with c0 per exponent (two terms); the rms of ln R.
Control (first): synthetic recovery of (c1 common, c0 per k) = (-148, [1.9, 2.0, 2.2, 2.7, 3.3]) to 1e-8 at the real v^2; with the ORIGINAL instrument (not repaired here) the q1320 values are not reproduced, so the control is run on synthetic data only.
Usage: py tight_run.py q1342_hardwall_refit_tight_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, sys
import numpy as np
import q1304_hardwall_family_blind as H4
import q1321_hardwall_common_c1_blind as B

QQ, Z, Da, H, nhw = B.QQ, B.Z, B.Da, B.H, B.nhw
TAGS = {
    0.02: "hw002",
    0.05: "hw005",
    0.052: "hw0052",
    0.054: "hw0054",
    0.056: "hw0056",
    0.058: "hw0058",
    0.06: "hw006",
    0.075: "hw0075",
    0.09: "hw009",
    0.1: "hw01",
    0.25: "hw025",
    0.35: "hw035",
    0.5: "hw05",
    0.75: "hw075",
}
SRC = {
    "u1304": (list(range(25, 38)), (0.02, 0.05, 0.1, 0.25)),
    "u1305": (list(range(29, 37)), (0.5, 0.75)),
    "u1306": (list(range(14, 25)) + [39, 40], (0.1, 0.25, 0.5, 0.75)),
    "u1321": (list(range(14, 41)), (0.02, 0.05, 0.35)),
    "u1324": ([38, 39, 40, 41], (0.06, 0.075, 0.09)),
    "u1326": ([39], (0.052, 0.054, 0.056, 0.058)),
}
data = {}
for k, tag in TAGS.items():
    sj = f"c52_{tag}c.json"
    N5, _ = nhw(sj, 92.0, 5)
    N7, _ = nhw(sj, 92.0, 7)
    rows = {}
    for src, (Ks, ks) in SRC.items():
        if k not in ks:
            continue
        for K in Ks:
            A = H.locr(f"c61_{src}_{tag}_K{K}")
            if A is None or K in rows:
                continue
            am5, am7 = QQ.root(N5, K), QQ.root(N7, K)
            R = (A - am5) * 1e9
            if R > 0 and abs(am5 - am7) * 1e9 <= 0.02 * R:
                rows[K] = (R, Z.v2f(N5, A))
    data[k] = sorted((K, r, v) for K, (r, v) in rows.items())


def fit(R, v, nt):
    R = np.array(R)
    v = np.array(v)
    d = np.array([Da(x) for x in v])
    x = 1.0 / v
    A = np.array([[(x[i] ** j) * d[i] / R[i] for j in range(nt)] for i in range(len(R))])
    c, *_ = np.linalg.lstsq(A, np.ones(len(R)), rcond=None)
    mod = sum(c[j] * x**j for j in range(nt)) * d
    return c, float(np.sqrt(np.mean(np.log(mod / R) ** 2)))


def joint(sets):
    ns = len(sets)
    rowsA = []
    rhs = []
    for i, (R, v) in enumerate(sets):
        v = np.array(v)
        R = np.array(R)
        d = np.array([Da(x) for x in v])
        x = 1.0 / v
        for j in range(len(R)):
            a = np.zeros(ns + 1)
            a[i] = d[j] / R[j]
            a[ns] = x[j] * d[j] / R[j]
            rowsA.append(a)
            rhs.append(1.0)
    c, *_ = np.linalg.lstsq(np.array(rowsA), np.array(rhs), rcond=None)
    out = []
    for i, (R, v) in enumerate(sets):
        v = np.array(v)
        R = np.array(R)
        d = np.array([Da(x) for x in v])
        mod = (c[i] + c[ns] / v) * d
        out.append(float(np.sqrt(np.mean(np.log(mod / R) ** 2))))
    return c, out


# control: synthetic recovery
ks_c = [0.02, 0.05, 0.1, 0.5, 0.75]
c0t = [1.9, 2.0, 2.2, 2.7, 3.3]
syn = []
for k, c0 in zip(ks_c, c0t):
    v = np.array([e[2] for e in data[k]])
    d = np.array([Da(x) for x in v])
    syn.append(((c0 - 148.0 / v) * d, v))
cs, _ = joint(syn)
ctl = bool(np.all(np.abs(cs[:5] / np.array(c0t) - 1) < 1e-8) and abs(cs[5] / -148.0 - 1) < 1e-8)
print(f"control: synthetic recovery of (c1 common, c0 per k): {ctl}")
if not ctl:
    print("READING q1342: VOID (control)")
    sys.exit(2)
print("per exponent: n, K_b range, v^2 range | two terms c0, c1, rms | three terms c0, c1, c2, rms")
for k in sorted(data):
    R = [e[1] for e in data[k]]
    v = [e[2] for e in data[k]]
    c2, r2 = fit(R, v, 2)
    c3, r3 = fit(R, v, 3) if len(R) >= 5 else (np.full(3, np.nan), np.nan)
    print(
        f"  k {k:5.3f}: n {len(R):3d}, K_b {data[k][0][0]}-{data[k][-1][0]}, v^2 {v[0]:6.1f}-{v[-1]:6.1f} | c0 {c2[0]:6.3f} c1 {c2[1]:8.2f} rms {r2:.4f} | c0 {c3[0]:6.3f} c1 {c3[1]:8.1f} c2 {c3[2]:9.1f} rms {r3:.4f}"
    )
ks_all = [k for k in sorted(data) if len(data[k]) >= 3]
sets = [([e[1] for e in data[k]], [e[2] for e in data[k]]) for k in ks_all]
cj, rj = joint(sets)
print(
    "one common c1 for all exponents: c1 = %.3f; c0 per k: " % cj[-1]
    + ", ".join(f"{k}: {cj[i]:.4f}" for i, k in enumerate(ks_all))
)
print("   rms of ln R with the common c1: " + ", ".join(f"{k}: {r:.4f}" for k, r in zip(ks_all, rj)))
sub = [k for k in ks_all if k in (0.02, 0.05, 0.1, 0.25, 0.5, 0.75)]
cs6, rs6 = joint([sets[ks_all.index(k)] for k in sub])
print(
    "common c1 for the six old exponents only: c1 = %.3f; c0: " % cs6[-1]
    + ", ".join(f"{k}: {cs6[i]:.4f}" for i, k in enumerate(sub))
    + "; rms "
    + ", ".join(f"{r:.4f}" for r in rs6)
)
c1j = cj[-1]
print(
    "the single-event exponents (k = 0.052 to 0.058, K_b 39) against the common-c1 form with c0(k) from a cubic through the common-fit c0 of k = 0.02, 0.05, 0.1, 0.25 (R / Da measured - predicted):"
)
cub = np.polyfit([0.02, 0.05, 0.1, 0.25], [cj[ks_all.index(k)] for k in (0.02, 0.05, 0.1, 0.25)], 3)
for k in (0.052, 0.054, 0.056, 0.058):
    K_, R_, v_ = data[k][0]
    print(
        f"  k {k}: R/Da {R_ / Da(v_):+.4f}, predicted {np.polyval(cub, k) + c1j / v_:+.4f}, difference {R_ / Da(v_) - (np.polyval(cub, k) + c1j / v_):+.4f}"
    )
print("READING q1342 (exploratory, no verdict)")
