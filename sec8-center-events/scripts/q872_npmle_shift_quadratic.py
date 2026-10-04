"""q872 (1582): NPMLE centre-event shifts against the 1D NPMLE transitions; quadratic form from d = 2, 3 predicts d = 4.
1D: q644b_da_temperatures_full.json, entry K holds a_t where the 1D count goes K -> K + 1, so A_g(K_eff) = a_t[K_eff - 1].
d-ball events: q668_npmle2d.json (+ q668_npmle2d_from23.json), q679_npmle_d3.json, q679_npmle_d4.json (DIM = 4 run of q679_npmle_nd.py).
K_eff before an event = 1 per origin mass + 2 per ring (types 'o', 'r'); births and lift-offs only.
Usage: py q872_npmle_shift_quadratic.py [ratio]   ('ratio' prints only the d = 2, 3 shifts and their ratio)
"""

import sys, json
import numpy as np

T = json.load(open("q644b_da_temperatures_full.json"))
AG = {int(k) + 1: v["a_t"] for k, v in T.items()}


def shifts(files):
    out = {}
    for f in files:
        try:
            ev = json.load(open(f))["events"]
        except FileNotFoundError:
            continue
        for e in ev:
            if e["kind"] not in ("birth", "liftoff"):
                continue
            Kb = sum(1 if t == "o" else 2 for t in e["types"])
            Ka = Kb + 1
            A = 0.5 * (e["A_lo"] + e["A_hi"])
            if Ka in AG:
                out.setdefault(Ka, (e["kind"], A - AG[Ka], A))
    return out


S2 = shifts(["q668_npmle2d.json", "q668_npmle2d_from23.json"])
S3 = shifts(["q679_npmle_d3.json"])
if len(sys.argv) > 1 and sys.argv[1] == "ratio":
    for K in sorted(set(S2) & set(S3)):
        print(
            f"K_eff {K:2d} {S2[K][0]:8s}/{S3[K][0]:8s} s2 {S2[K][1]:.5f} s3 {S3[K][1]:.5f} ratio {S3[K][1]/S2[K][1]:.4f}"
        )
    sys.exit()
S4 = shifts(["q679_npmle_d4.json"])
rows = []
for K in sorted(S4):
    if K not in S2 or K not in S3:
        continue
    kind, s4, A = S4[K]
    s2 = S2[K][1]
    s3 = S3[K][1]
    if S2[K][0] != kind or S3[K][0] != kind:
        print(f"K {K}: kind mismatch")
        continue
    p = 3 * (s3 - s2)
    rows.append((K, kind, A, s4, p, s4 / p - 1, s4 / (3 * s2) - 1, s4 / (1.5 * s3) - 1))
nin = sum(abs(r[5]) <= 0.01 for r in rows)
print(
    f"d = 4 NPMLE: {len(rows)} matched events; within 1 %: {nin}; verdict {'PASS' if rows and nin >= 0.9*len(rows) else 'FAIL'}"
)
for K, kind, A, s4, p, rq, r2, r3 in rows:
    print(
        f"   K_eff {K:2d} {kind:8s} A {A:8.4f} shift {s4:.5f} quadratic {p:.5f} ({rq:+.4f})  linear-2D {r2:+.4f}  linear-3D {r3:+.4f}"
    )
if rows:
    a = np.array([r[5:] for r in rows])
    print("   rms rel: quadratic %.4f, linear-2D %.4f, linear-3D %.4f" % tuple(np.sqrt((a**2).mean(0))))
