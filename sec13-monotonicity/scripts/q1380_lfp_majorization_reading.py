"""q1380 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1380_lfp_majorization_reading.py from temporal_within/centre_scripts/): the structure of amendments 2130-2137 on a fourth functional, the symmetric least-favourable prior (LFP) of the bounded normal mean under squared error, theta in [-m, m] (states of c80_lfp_states.json and c80_lfp_states_b.json: 182 distinct states, m = 40.9 to 59.0 in steps of 0.1, K = 69 to 109, certified in ball arithmetic). Masses: right-half atoms a_1..a_J (a_J = m the wall), the centre atom if present; the LFP mass profile is expected to be largest at the centre and small at the wall (cos^2-like, T43), so condition (a) of amendment 2130 (masses nondecreasing outward) is expected to FAIL here. Question: does the majorization p(m2) majorized by p(m1) (m1 < m2) still hold, and what is the sign structure of the mass flow? Expected before the run: unknown; if majorization holds although (a) fails, the reduction through (a) and (b) is not the whole mechanism.
Printed: (a) the fraction of states with masses nondecreasing outward and the direction of the trend; majorization over consecutive pairs (largest max_r S_r(m2) - S_r(m1) and the number of pairs above 1e-12, pairs across a change of K listed); for consecutive same-family pairs (same K, same centre structure, dm = 0.1) the number of sign changes of dp_j/dm in position order and the sign pattern at both ends.
"""

import json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
allst = {}
for fn in ("c80_lfp_states.json", "c80_lfp_states_b.json"):
    for s in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fn))):
        allst[(round(float(s["m"]), 6), int(s["K"]), bool(s["centre"]))] = s
keys = sorted(allst)
st = []
for m, K, c in keys:
    s = allst[(m, K, c)]
    ps = [float(x) for x in s["masses"]]
    cm = float(s["centre_mass"]) if c and s.get("centre_mass") not in (None, "None") else None
    order = ([cm] if c else []) + ps
    full = np.sort(np.array(([cm] if c else []) + ps + ps))[::-1]
    st.append((m, K, c, np.array(order), full, abs(float(s["maxF"]))))


def partial(v, n):
    p = np.zeros(n)
    p[: len(v)] = v
    return np.cumsum(p)


def worst(a, b):
    n = max(len(a), len(b))
    d = partial(b, n) - partial(a, n)
    i = int(np.argmax(d))
    return float(d[i]), i + 1


tot = [abs(s[4].sum() - 1) for s in st]
print(
    f"states {len(st)}; largest |total mass - 1| {max(tot):.1e}; largest residual maxF {max(s[5] for s in st):.1e}",
    flush=True,
)
inc = sum(1 for s in st if np.all(np.diff(s[3][(1 if s[2] else 0) :]) >= -1e-14))
dec = sum(1 for s in st if np.all(np.diff(s[3][(1 if s[2] else 0) :]) <= 1e-14))
print(
    f"(a) masses (centre excluded) nondecreasing outward in {inc} of {len(st)} states, nonincreasing outward in {dec}; example m {st[0][0]} K {st[0][1]}: first masses {np.round(st[0][3][:3], 5)} ... last {np.round(st[0][3][-3:], 5)}",
    flush=True,
)
viol = []
wmax = -1
cross = 0
npairs = 0
for a, b in zip(st[:-1], st[1:]):
    if b[0] - a[0] < 1e-9:
        continue
    d, r = worst(a[4], b[4])
    npairs += 1
    wmax = max(wmax, d)
    cross += int(a[1] != b[1])
    if d > 1e-12:
        viol.append((a[0], a[1], b[0], b[1], d, r))
print(
    f"majorization: consecutive pairs {npairs} (across a change of K: {cross}); pairs with max_r (S_r(m2) - S_r(m1)) > 1e-12: {len(viol)}; largest value {wmax:.3e}",
    flush=True,
)
for v in viol[:10]:
    print(f"   violation m {v[0]} K {v[1]} -> m {v[2]} K {v[3]}: {v[4]:.3e} at r = {v[5]}")
cnt = {}
ends = {}
for a, b in zip(st[:-1], st[1:]):
    if a[1] == b[1] and a[2] == b[2] and 0 < b[0] - a[0] < 0.25:
        dd = (b[3] - a[3]) / (b[0] - a[0])
        sg = [np.sign(x) for x in dd if abs(x) > 1e-14]
        n = sum(1 for x, y in zip(sg[:-1], sg[1:]) if x != y)
        cnt[n] = cnt.get(n, 0) + 1
        ends[(sg[0], sg[-1])] = ends.get((sg[0], sg[-1]), 0) + 1
print(
    f"mass flow dp/dm in position order (same family, dm = 0.1): sign changes by count {sorted(cnt.items())}; (first, last) signs {sorted(ends.items())}",
    flush=True,
)
print("READING q1380 (exploratory, no verdict)")
