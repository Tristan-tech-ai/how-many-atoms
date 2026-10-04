"""I_centre_scan (03 Oct 2026): for boundary-ring states of the curve-restricted problem (G_cont states.jsonl, double precision),
D - C at the centre and the largest D - C on an interior grid of the filled ellipse; D at one atom as a check (should equal C).
Gauss-Hermite in each axis (NGH nodes), D(q) = |q|^2/2 - E log S(q + Z). Usage: py I_centre_scan.py states.jsonl [NGH]
"""

import sys, json, math
import numpy as np

NGH = int(sys.argv[2]) if len(sys.argv) > 2 else 120
xg, wg = np.polynomial.hermite_e.hermegauss(NGH)
wg = wg / np.sqrt(2 * np.pi)
Z1, Z2 = np.meshgrid(xg, xg, indexing="ij")
W = np.outer(wg, wg)


def atoms(st):
    rp, rm = st["rp"], st["rm"]
    ts = []
    ms = []
    k = 0
    if st["X"]:
        ts += [0.0, math.pi]
        ms += [st["m"][k]] * 2
        k += 1
    if st["Y"]:
        ts += [math.pi / 2, 3 * math.pi / 2]
        ms += [st["m"][k]] * 2
        k += 1
    for g in st["G"]:
        ts += [g, -g, math.pi - g, math.pi + g]
        ms += [st["m"][k]] * 4
        k += 1
    xs = np.array([[rp * math.cos(t), rm * math.sin(t)] for t in ts])
    return xs, np.array(ms)


def D(q, xs, ms):
    y1 = q[0] + Z1
    y2 = q[1] + Z2
    e = y1[..., None] * xs[:, 0] + y2[..., None] * xs[:, 1] - 0.5 * (xs**2).sum(1)
    mx = e.max(-1, keepdims=True)
    L = np.log((ms * np.exp(e - mx)).sum(-1)) + mx[..., 0]
    return 0.5 * (q[0] ** 2 + q[1] ** 2) - (W * L).sum()


seen = set()
for line in open(sys.argv[1]):
    st = json.loads(line)
    key = (round(st["rm"], 6), st.get("label"))
    if key in seen:
        continue
    seen.add(key)
    xs, ms = atoms(st)
    C = st["C"]
    d0 = D((0.0, 0.0), xs, ms) - C
    datom = D(tuple(xs[0]), xs, ms) - C
    best = -1e9
    for s in (0.2, 0.4, 0.6, 0.8, 0.9):
        for t in np.linspace(0, math.pi / 2, 7):
            v = D((s * st["rp"] * math.cos(t), s * st["rm"] * math.sin(t)), xs, ms) - C
            best = max(best, v)
    print(
        f"rm {st['rm']:.5f} K {st.get('K')} {st.get('label')}: D-C centre {d0:+.4e}; interior max {max(best, d0):+.4e}; check D-C at atom {datom:+.1e}",
        flush=True,
    )
