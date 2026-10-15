"""q1252 (8 Oct 2026, registered in amendment 2037): the order-16 statements of Section 6.5 recomputed with the cubic (k = 3) and the quintic (k = 5) spline of the half-line counting function.
dev(16) = A_stored - a_16; bar = 0.3e-9 + 3 |a_16 - a_14| (item L1 of amendment 1973); roots bracketed by +-2.5 around the zeroth-order root as q1216.
Controls (first): the cubic variant reproduces dev(16) of q1220_out.txt within 0.001e-9 at capacity K_b 24 (-0.7939e-9) and K_b 40 (-0.1410e-9) and NPMLE K_b 24 (+121.4170e-9); the quintic variant differs from the cubic by more than 0.001e-9 at capacity K_b 24.
Items: V1 capacity |dev16| < bar at all 47 events K_b 10-59; V2 NPMLE |dev16| < bar at all stored transitions K_b 31-52; V3 NPMLE dev16 > 0 at all stored events K_b 6-28; V4 capacity K_b 40-59 max |dev16| < 0.4e-9.
Usage: py q1252_order16_quintic.py  (from temporal_within/centre_scripts/)"""

import json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.optimize import brentq
import q1250_quintic_residual as Q
import q1209_births_law_derived_series as G

C, N = Q.C, Q.N
m = __import__("q1216_derived_series_lower_kb")


def root_n(Nf, K, nmax):
    z = lambda a: 2 * Nf(a) - K - 1 / 6
    a0 = brentq(z, 1.0, 140.0, xtol=1e-12)
    f = lambda a: z(a) - G.delta((Nf(a + G.H) - Nf(a - G.H)) / (2 * G.H), nmax)
    return brentq(f, a0 - 2.5, a0 + 2.5, xtol=1e-14)


def main():
    gv = lambda f: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))
    stored = {
        "capacity": {K: gv(f) for K, f in m.CAP_FILES.items() if 10 <= K <= 59},
        "npmle": {K: gv(f) for K, f in m.NPM_FILES.items() if 6 <= K <= 52},
    }
    capN = lambda a: C.Q.N_of(a, *C.CAPL)
    NL = {"capacity": (capN, C.X), "npmle": (lambda a: Q.N4(a, N.LAW), N.pn)}
    res = {}
    for name in NL:
        for k in (3, 5):
            Nf = Q.make(NL[name][0], NL[name][1], k)
            r = {}
            for K, A in stored[name].items():
                a16, a14 = root_n(Nf, K, 16), root_n(Nf, K, 14)
                r[K] = ((A - a16) * 1e9, (0.3e-9 + 3 * abs(a16 - a14)) * 1e9)
            res[(name, k)] = r
    c1 = (
        abs(res[("capacity", 3)][24][0] + 0.7939) < 0.001
        and abs(res[("capacity", 3)][40][0] + 0.1410) < 0.001
        and abs(res[("npmle", 3)][24][0] - 121.4170) < 0.001
    )
    c2 = abs(res[("capacity", 3)][24][0] - res[("capacity", 5)][24][0]) > 0.001
    print(
        f"control: cubic dev16 capacity K_b 24 {res[('capacity', 3)][24][0]:+.4f}, K_b 40 {res[('capacity', 3)][40][0]:+.4f}, NPMLE K_b 24 {res[('npmle', 3)][24][0]:+.4f} (q1220: -0.7939, -0.1410, +121.4170): {c1}; quintic differs from cubic at capacity K_b 24 by {abs(res[('capacity', 3)][24][0] - res[('capacity', 5)][24][0]):.4f}: {c2}"
    )
    if not (c1 and c2):
        print("READING q1252: VOID (control)")
        sys.exit(2)
    out = {}
    for k in (3, 5):
        cap, npm = res[("capacity", k)], res[("npmle", k)]
        v1 = all(abs(d) < b for d, b in cap.values())
        out_cap = sorted(K for K, (d, b) in cap.items() if abs(d) >= b)
        v2 = all(abs(npm[K][0]) < npm[K][1] for K in npm if 31 <= K <= 52)
        n2 = sum(1 for K in npm if 31 <= K <= 52)
        low = {K: npm[K] for K in npm if 6 <= K <= 28}
        v3 = all(d > 0 for d, b in low.values())
        above = [K for K, (d, b) in low.items() if abs(d) > b]
        mx_hi = max(abs(cap[K][0]) for K in cap if 40 <= K <= 59)
        mx_mid = max(abs(cap[K][0]) for K in cap if 24 <= K <= 39)
        v4 = mx_hi < 0.4
        ratios = [abs(low[K][0]) / low[K][1] for K in above]
        print(
            f"k = {k}: capacity events {len(cap)}, inside bar {len(cap) - len(out_cap)}, outside at K_b {out_cap}; max |dev16| K_b 40-59 {mx_hi:.4f}e-9, K_b 24-39 {mx_mid:.4f}e-9; NPMLE K_b 31-52 events {n2}, all inside {v2}; NPMLE K_b 6-28 events {len(low)}, all positive {v3}, above bar {len(above)}"
            + (f", |dev|/bar {min(ratios):.2f} to {max(ratios):.2f}" if ratios else "")
        )
        out[k] = (v1, v2, v3, v4)
    print(
        f"item V1 (capacity all inside bar): k=3 {out[3][0]}, k=5 {out[5][0]}; V2 (NPMLE K_b 31-52 inside): k=3 {out[3][1]}, k=5 {out[5][1]}; V3 (NPMLE K_b 6-28 positive): k=3 {out[3][2]}, k=5 {out[5][2]}; V4 (capacity K_b 40-59 max < 0.4e-9): k=3 {out[3][3]}, k=5 {out[5][3]}"
    )
    json.dump(
        {f"{n}_{k}": {str(K): v for K, v in d.items()} for (n, k), d in res.items()},
        open("q1252_dev16.json", "w"),
    )
    print(
        "READING q1252: for k = 5, V1 "
        + str(out[5][0])
        + ", V2 "
        + str(out[5][1])
        + ", V3 "
        + str(out[5][2])
        + ", V4 "
        + str(out[5][3])
        + " (no verdict)"
    )


if __name__ == "__main__":
    main()
