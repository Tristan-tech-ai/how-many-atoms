"""q870 (1581): capacity centre-event shifts in d = 4, 5, 8 predicted from d = 2, 3 by shift_d = (d - 1)(alpha + beta d) at matched K_eff.
alpha = 3 s2 - s3, beta = s3/2 - s2 (s_d = A_d(event to K_eff) - A_g(K_eff), A_g from the 1D capacity transitions as in q657).
K_eff before an event = 1 per origin mass + 2 per ring/shell (from the event's 'types'); only births and lift-offs (step 1) are used.
Usage: py q870_cap_shift_quadratic.py
"""

import json, glob
import numpy as np

AG = {
    3: 1.66593,
    4: 2.90783,
    5: 4.02128,
    6: 5.05816,
    7: 6.03983,
    8: 6.9789594281,
    9: 7.8834798596,
    10: 8.7590077552,
    11: 9.6096906207,
    12: 10.4387115429,
    13: 11.2485854341,
    14: 12.042209243774412,
}  # copied as text from q657
for pat in ("c60_K*.json", "c60_small_K*.json", "c60_f3_K*.json", "c60_d38_K*.json"):
    for f in glob.glob(pat):
        try:
            d = json.load(open(f))
            if "A_t" in d and "_p192" not in f:
                AG.setdefault(int(d["K"]) + 1, float(d["A_t"]))
        except Exception:
            pass


def shifts(files):
    out = {}
    for f in files:
        for e in json.load(open(f))["events"]:
            if e["kind"] not in ("birth", "liftoff"):
                continue
            Kb = sum(1 if t == "o" else 2 for t in e["types"])
            Ka = Kb + 1
            A = 0.5 * (e["A_lo"] + e["A_hi"])
            if Ka in AG:
                out.setdefault(Ka, (e["kind"], A - AG[Ka], A))
    return out


S = {
    2: shifts(["q659_rings2d_evstates.json"]),
    3: shifts(["q659_shells3d_evstates.json"]),
    4: shifts(["q662_ball_d4.json"]),
    5: shifts(["q662_ball_d5.json"]),
    8: shifts(["q662_ball_d8.json", "q666_ball_d8.json"]),
}
band = {4: 0.01, 5: 0.02, 8: 0.05}
for d in (4, 5, 8):
    rows = []
    nin = 0
    for K in sorted(S[d]):
        if K not in S[2] or K not in S[3]:
            continue
        kind, sd, A = S[d][K]
        s2 = S[2][K][1]
        s3 = S[3][K][1]
        if S[2][K][0] != kind or S[3][K][0] != kind:
            print(f"  d {d} K {K}: kind mismatch {kind} / {S[2][K][0]} / {S[3][K][0]}")
            continue
        al = 3 * s2 - s3
        be = s3 / 2 - s2
        p = (d - 1) * (al + be * d)
        rq = sd / p - 1
        r2 = sd / ((d - 1) * s2) - 1
        r3 = sd / ((d - 1) * s3 / 2) - 1
        nin += abs(rq) <= band[d]
        rows.append((K, kind, A, sd, p, rq, r2, r3))
    print(
        f"d = {d}: {len(rows)} matched events; in band ({band[d]:.0%}) {nin}; verdict {'PASS' if rows and nin >= 0.9*len(rows) else 'FAIL'}"
    )
    for K, kind, A, sd, p, rq, r2, r3 in rows:
        print(
            f"   K_eff {K:2d} {kind:8s} A {A:8.4f}  shift {sd:.5f}  quadratic {p:.5f} ({rq:+.4f})  linear-2D {r2:+.4f}  linear-3D {r3:+.4f}"
        )
    if rows:
        a = np.array([r[5:] for r in rows])
        print(
            f"   rms rel: quadratic {np.sqrt((a[:, 0]**2).mean()):.4f}, linear-2D {np.sqrt((a[:, 1]**2).mean()):.4f}, "
            f"linear-3D {np.sqrt((a[:, 2]**2).mean()):.4f}"
        )
