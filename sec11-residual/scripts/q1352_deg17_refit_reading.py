"""q1352 (8 Oct 2026, EXPLORATORY reading, amendment 2121; no item, no verdict): the closure fits of the hard wall and the capacity with the prediction a_med from a spline of degree 17 (converged: degrees 15, 17, 19 agree to 1e-12 at K_b >= 26, q1351) instead of degree 5 (run through tight_run.py). Events and inclusion as in q1342 (hard wall: q1304-q1306, q1321, q1324, q1326 files; inclusion R > 0 and |a_17 - a_19| 1e9 <= 0.02 R) and q1344 (capacity K_b 20-59).
Printed: per-exponent two-term fits R / Da = c0 + c1 / v^2 (rms of ln R) for degree 17 next to degree 5; the common-c1 fit over the exponents with at least three events; the capacity fits on K_b 26-59 and 20-59; the relative misfit of the common form per K_b for the capacity and the hard-wall exponents 0.1, 0.5, 0.75.
Control (first): synthetic recovery of (c1 common, c0 per k) to 1e-8 at the real v^2 (joint fit); degree 5 reproduces the q1342 per-exponent two-term c0 and c1 of k = 0.5 (2.687, -147.25) within 2e-3.
Usage: py tight_run.py q1352_deg17_refit_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import q1342_hardwall_refit_tight_reading as P
B = P.B
QQ, Z, Da, H, nhw = B.QQ, B.Z, B.Da, B.H, B.nhw
C = Z.C
from scipy.optimize import brentq as _bq


def root2(Nf, K):
    z = lambda a: 2 * Nf(a) - K - 1 / 6
    a0 = _bq(z, 5.0, 55.0, xtol=1e-12)
    f = lambda a: z(a) - float(QQ.T3(1.0 / (4 * np.pi * (Nf(a + QQ.H) - Nf(a - QQ.H)) / (2 * QQ.H)) ** 2))
    return _bq(f, a0 - 2.5, a0 + 2.5, xtol=1e-14)


def events(deg):
    data = {}
    for k, tag in P.TAGS.items():
        sj = f"c52_{tag}c.json"
        Na, _ = nhw(sj, 92.0, deg)
        Nb, _ = nhw(sj, 92.0, deg + 2)
        rows = {}
        for src, (Ks, ks) in P.SRC.items():
            if k not in ks:
                continue
            for K in Ks:
                A = H.locr(f"c61_{src}_{tag}_K{K}")
                if A is None or K in rows:
                    continue
                am_a, am_b = root2(Na, K), root2(Nb, K)
                R = (A - am_a) * 1e9
                if R > 0 and abs(am_a - am_b) * 1e9 <= 0.02 * R:
                    rows[K] = (R, Z.v2f(Na, A))
        data[k] = sorted((K, r, v) for K, (r, v) in rows.items())
    return data


d5, d17 = events(5), events(17)
ks_c = [0.02, 0.05, 0.1, 0.5, 0.75]
c0t = [1.9, 2.0, 2.2, 2.7, 3.3]
syn = []
for k, c0 in zip(ks_c, c0t):
    v = np.array([e[2] for e in d17[k]])
    syn.append(((c0 - 148.0 / v) * np.array([Da(x) for x in v]), v))
cs, _ = P.joint(syn)
k5 = P.fit([e[1] for e in d5[0.5]], [e[2] for e in d5[0.5]], 2)[0]
ctl = bool(
    np.all(np.abs(cs[:5] / np.array(c0t) - 1) < 1e-8)
    and abs(cs[5] / -148.0 - 1) < 1e-8
    and abs(k5[0] - 2.687) < 2e-3
    and abs(k5[1] + 147.25) < 2e-2
)
print(
    f"control (synthetic joint recovery; degree 5 reproduces q1342 k = 0.5: c0 {k5[0]:.3f}, c1 {k5[1]:.2f}): {ctl}"
)
if not ctl:
    print("READING q1352: VOID (control)")
    sys.exit(2)
print("per exponent, two terms (c0, c1, rms of ln R): degree 5 -> degree 17; n events")
for k in sorted(d17):
    if len(d17[k]) < 4:
        continue
    f5 = P.fit([e[1] for e in d5[k]], [e[2] for e in d5[k]], 2)
    f17 = P.fit([e[1] for e in d17[k]], [e[2] for e in d17[k]], 2)
    print(
        f"  k {k:5.3f}: n {len(d5[k])}/{len(d17[k])}: {f5[0][0]:.3f}, {f5[0][1]:8.2f}, {f5[1]:.4f} -> {f17[0][0]:.3f}, {f17[0][1]:8.2f}, {f17[1]:.4f}"
    )
for lab, d in (("degree 5", d5), ("degree 17", d17)):
    ks_all = [k for k in sorted(d) if len(d[k]) >= 3]
    cj, rj = P.joint([([e[1] for e in d[k]], [e[2] for e in d[k]]) for k in ks_all])
    print(
        f"common c1, {lab}: {cj[-1]:.3f}; c0: "
        + ", ".join(f"{k}: {cj[i]:.4f}" for i, k in enumerate(ks_all))
        + "; rms: "
        + ", ".join(f"{r:.4f}" for r in rj)
    )
q16 = __import__("q1216_derived_series_lower_kb")


def cap(deg):
    Na = QQ.make(lambda a: C.Q.N_of(a, *C.CAPL), C.X, deg)
    Nb = QQ.make(lambda a: C.Q.N_of(a, *C.CAPL), C.X, deg + 2)
    rows = []
    for K, f in sorted(q16.CAP_FILES.items()):
        if 20 <= K <= 59:
            j = json.load(open(f + ".json"))
            a = j.get("A_t") or j.get("a_t")
            rows.append((K, (a - root2(Na, K)) * 1e9, Z.v2f(Na, a)))
    return rows


for deg in (5, 17):
    rows = cap(deg)
    K_ = np.array([r[0] for r in rows])
    R_ = np.array([r[1] for r in rows])
    V_ = np.array([r[2] for r in rows])
    D_ = np.array([Da(v) for v in V_])
    for lab, m in ((f"K_b 26-59", K_ >= 26), ("K_b 20-59", K_ >= 20)):
        c, r = P.fit(R_[m], V_[m], 2)
        print(
            f"capacity degree {deg}, {lab}: n {int(m.sum())}, c0 {c[0]:.4f}, c1 {c[1]:.2f}, rms of ln R {r:.4f}"
        )
    c, r = P.fit(R_[K_ >= 26], V_[K_ >= 26], 2)
    mis = (R_ / D_) / (c[0] + c[1] / V_) - 1
    print(
        f"   relative misfit of the free two-term fit (K_b 26-59), degree {deg}: "
        + " ".join(f"{K}:{x:+.3f}" for K, x in zip(K_, mis) if K >= 26)
    )
json.dump(
    {
        "hard_wall_deg5": {str(k): v for k, v in d5.items()},
        "hard_wall_deg17": {str(k): v for k, v in d17.items()},
        "capacity_deg5": cap(5),
        "capacity_deg17": cap(17),
    },
    open("q1352_events.json", "w"),
)
print("events written to q1352_events.json")
print("READING q1352 (exploratory, no verdict)")
