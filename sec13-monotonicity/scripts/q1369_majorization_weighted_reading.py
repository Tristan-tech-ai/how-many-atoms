"""q1369 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1369_majorization_weighted_reading.py from temporal_within/centre_scripts/): majorization of the sorted mass vectors and the Robin Hood conditions (a), (b) of q1367 and q1368 on the stored rows of the two weighted problems (immune results_q584_ext.json, NPMLE results_q601_ext.json; uniform data on [-a, a]; pair masses v_j split in two atoms of mass v_j / 2, centre mass c = 1 - sum v). Expected before the run: unknown (the mass density is flat for these problems, so the masses are nearly equal and condition (a) may fail while the majorization may still hold). Reported: for each file the number of rows, the number of consecutive pairs (a increasing) with max_r S_r(a2) - S_r(a1) above 1e-12 (and the largest value), the number of rows whose masses are nondecreasing outward, and the number of consecutive same-family pairs (same K and centre structure) whose mass flow has exactly one sign change with the inner end positive.
Controls: the checker of q1367 (synthetic majorized chain accepted, reversed rejected)."""

import json, os, sys
import numpy as np


def die(msg):
    print(msg, flush=True)
    os._exit(2)


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def partial(m, n):
    p = np.zeros(n)
    p[: len(m)] = m
    return np.cumsum(p)


def worst(m1, m2):
    n = max(len(m1), len(m2))
    d = partial(m2, n) - partial(m1, n)
    i = int(np.argmax(d))
    return float(d[i]), i + 1


def changes(d, tol):
    s = [np.sign(x) for x in d if abs(x) > tol]
    return int(sum(1 for a, b in zip(s[:-1], s[1:]) if a != b)), s


rng = np.random.default_rng(3)
v = np.sort(rng.dirichlet(np.ones(8)))[::-1]
chain = [v]
for _ in range(6):
    u = chain[-1].copy()
    i, k = 0, len(u) - 1
    t = 0.2 * (u[i] - u[k]) / 2
    u[i] -= t
    u[k] += t
    chain.append(np.sort(u)[::-1])
c1 = (
    all(worst(chain[i], chain[i + 1])[0] <= 1e-15 for i in range(6))
    and sum(1 for i in range(6) if worst(chain[i + 1], chain[i])[0] > 1e-6) >= 5
)
print(f"control c1 (majorization checker): {c1}", flush=True)
if not c1:
    die("READING q1369: VOID (control)")
for kind, fn in (("immune", "results_q584_ext.json"), ("npmle", "results_q601_ext.json")):
    rows = sorted(json.load(open(os.path.join(ROOT, fn)))["rows"], key=lambda r: float(r["a"]))
    data = []
    for r in rows:
        v = [float(x) for x in r["v"]]
        centre = bool(r["centre"])
        c = 1 - sum(v) if centre else None
        pos = ([c] if centre else []) + [x / 2 for x in v for _ in (0, 1)]
        order = ([c] if centre else []) + [
            x / 2 for x in v
        ]  # position order of the right-half atoms, centre first (each non-centre entry stands for the mirror pair)
        data.append((float(r["a"]), int(r["K"]), centre, np.sort(np.array(pos))[::-1], np.array(order)))
    nv = 0
    wmax = -1.0
    npairs = 0
    cross = 0
    for (a1, K1, c1_, m1, o1), (a2, K2, c2_, m2, o2) in zip(data[:-1], data[1:]):
        if a2 - a1 < 1e-12:
            continue
        d, r = worst(m1, m2)
        npairs += 1
        wmax = max(wmax, d)
        nv += int(d > 1e-12)
        cross += int(K1 != K2)
    mono = sum(1 for d in data if np.all(np.diff(d[4][(1 if d[2] else 0) :]) >= -1e-14))
    one = 0
    tot = 0
    bad = []
    for (a1, K1, c1_, m1, o1), (a2, K2, c2_, m2, o2) in zip(data[:-1], data[1:]):
        if K1 == K2 and c1_ == c2_ and 1e-6 < a2 - a1 < 0.2:
            dd = (o2 - o1) / (a2 - a1)
            n, sg = changes(dd, 1e-14 / (a2 - a1))
            tot += 1
            ok = n == 1 and sg[0] > 0 and sg[-1] < 0
            one += int(ok)
            if not ok and len(bad) < 3:
                bad.append((K1, round(a1, 4), n, sg[0] if sg else 0, sg[-1] if sg else 0))
    print(
        f"{kind}: rows {len(data)}, consecutive pairs {npairs} (across a change of K: {cross}); pairs with max_r (S_r(a2) - S_r(a1)) > 1e-12: {nv}; largest value {wmax:.3e}; rows with masses nondecreasing outward (centre excluded): {mono} of {len(data)}; same-family pairs {tot} with one sign change, inner +, outer -: {one}; examples of other pairs {bad}",
        flush=True,
    )
print("READING q1369 (exploratory, no verdict)")
