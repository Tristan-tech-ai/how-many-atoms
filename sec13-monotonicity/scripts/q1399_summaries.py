"""q1399 (9 Oct 2026, summary by script, no verdict; py q1399_summaries.py from temporal_within/small_gap_q206/): ranges and drifts from the printed outputs q1395_out.txt and q1398_out.txt (21 states f0.97, K = 62..82), per parity."""

import re
import numpy as np

r95 = []
for l in open("q1395_out.txt", encoding="utf-8"):
    t = l.split()
    if len(t) >= 15 and t[0].isdigit() and t[1] in ("even", "odd"):
        r95.append(
            dict(
                K=int(t[0]),
                par=t[1],
                s=[float(t[3]), float(t[4]), float(t[5])],
                amp=float(t[7]),
                v=[float(x) for x in t[8:13]],
                R=float(t[14]),
            )
        )
r98 = []
for l in open("q1398_out.txt", encoding="utf-8"):
    t = l.split()
    if len(t) >= 9 and t[0].isdigit() and t[1] in ("even", "odd") and t[-1] == "OK":
        r98.append(
            dict(
                K=int(t[0]),
                par=t[1],
                A=float(t[2]),
                flow=float(t[3]),
                c1=float(t[4]),
                E=float(t[5]),
                rho=float(t[6]),
                ef=float(t[7]),
            )
        )
print("q1395 rows", len(r95), "; q1398 rows", len(r98), "(first 21 are the stage f0.97 states)")
fixed = r98[:21]
assert [r["K"] for r in fixed] == list(range(62, 83))
for par in ("even", "odd"):
    a = [r for r in fixed if r["par"] == par]
    print(
        f"{par} f0.97: {len(a)} states; c_1/E from {min(r['rho'] for r in a):.5f} to {max(r['rho'] for r in a):.5f} (K {a[0]['K']}: {a[0]['rho']:.5f}, K {a[-1]['K']}: {a[-1]['rho']:.5f}); E/flow from {min(r['ef'] for r in a):.4f} to {max(r['ef'] for r in a):.4f}; flow all positive {all(r['flow'] > 0 for r in a)}; c_1 all negative {all(r['c1'] < 0 for r in a)}"
    )
    b = [r for r in r95 if r["par"] == par]
    names = [
        "c_D(m)/c_D'(m)",
        "c_D'(m-1)/c_D'(m)",
        "c_D(m-1)/c_D'(m)",
        "c_D'(m-2)/c_D'(m)",
        "c_D(m-2)/c_D'(m)",
    ]
    for i, nm in enumerate(names):
        print(
            f"   {nm}: K {b[0]['K']}: {b[0]['v'][i]:.5f}, K {b[-1]['K']}: {b[-1]['v'][i]:.5f}, relative change {b[-1]['v'][i] / b[0]['v'][i] - 1:+.4f}"
        )
for par in ("even", "odd"):
    b = [r for r in r95 if r["par"] == par]

    def edge(r):
        v = r["v"]
        return (1 + v[1] + v[3]) / (-(v[0] + v[2] + v[4]))

    print(
        f"{par}: edge-only ratio P_edge/N_edge from the three outermost atoms (rows D', D of m, m-1, m-2): K {b[0]['K']}: {edge(b[0]):.5f}, K {b[-1]['K']}: {edge(b[-1]):.5f}; minimum {min(edge(r) for r in b):.5f}, maximum {max(edge(r) for r in b):.5f}; full P/N K {b[0]['K']}: {b[0]['R']:.5f}, K {b[-1]['K']}: {b[-1]['R']:.5f}"
    )
print(
    "depths s_1..s_3 identical in all 21 states to 5 digits:",
    len({tuple(r["s"]) for r in r95}) == 1,
    sorted({tuple(r["s"]) for r in r95}),
)
fam = r98[21:26]
assert all(r["K"] == 62 for r in fam)
print(
    "K = 62 family (5 states, A from %.4f to %.4f): c_1/E from %.5f to %.5f, E/flow from %.4f to %.4f"
    % (
        min(r["A"] for r in fam),
        max(r["A"] for r in fam),
        min(r["rho"] for r in fam),
        max(r["rho"] for r in fam),
        min(r["ef"] for r in fam),
        max(r["ef"] for r in fam),
    )
)
print("small A:", [(r["K"], r["A"], round(r["rho"], 5), round(r["ef"], 4)) for r in r98[26:]])
print("READING q1399 done")
