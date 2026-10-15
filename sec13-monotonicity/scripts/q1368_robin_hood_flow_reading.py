"""q1368 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1368_robin_hood_flow_reading.py from temporal_within/centre_scripts/): is the flow of the probability masses of the capacity-achieving input a Robin Hood flow, which would imply the majorization of q1367 and, through it, the monotonicity of K(A)?
Sufficient condition (to be tested here, then proved if it holds). Order the atoms by position (centre or innermost pair first, edge atom last). (a) The masses increase with the position (the edge atom is the heaviest). (b) Along a family (fixed K, no event), the derivative dp_j/dA of the mass of the j-th atom from the edge has exactly one sign change in position order: negative for the outer atoms, positive for the inner ones. Then for every r the sum of the r outermost masses, S_r, has dS_r/dA = sum of the r outermost dp_j <= 0 (the partial sums of a vector with zero total and one sign change from minus to plus never turn positive), which is majorization of p(A2) by p(A1) inside a family; the events (a new atom with mass growing from zero at the origin, a pair splitting from the centre atom) need their own check, which q1367 covers numerically.
Test. For every pair of consecutive OPTIMAL states of the stored capacity files (K = 40, 60-83; the same K and the same centre/no-centre structure, A2 - A1 between 1e-6 and 0.2): (a) fraction of states whose masses, in position order, are nondecreasing outward; (b) the number of sign changes of (p2 - p1) in position order and the sign pattern at the two ends (outer negative, inner positive). Expected before the run (the condition): (a) in all states, (b) exactly one sign change with outer negative and inner positive in all pairs, up to masses changing by less than 1e-14 (noise: such entries are treated as zero).
Controls (first): c1 the counter on a synthetic one-sign-change vector and on a vector with three; c2 shuffling the mass differences of one pair gives more than one sign change; c3 the masses of a state with two entries swapped break (a).
Usage: py q1368_robin_hood_flow_reading.py"""

import glob, json, os, re, sys
import numpy as np


def die(msg):
    print(msg, flush=True)
    os._exit(2)


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


def vec(
    j,
):  # masses in position order: centre atom (if any) first, then the right-half atoms inward to outward, the edge atom last
    w = [float(numstr(x)) for x in j["w"]]
    we = float(numstr(j["we"]))
    wc = None if j.get("wc") in (None, "None") else float(numstr(j["wc"]))
    return ([wc] if wc is not None else []) + w + [we], wc is not None


def changes(d, tol):
    s = [np.sign(x) for x in d if abs(x) > tol]
    return int(sum(1 for a, b in zip(s[:-1], s[1:]) if a != b)), s


rng = np.random.default_rng(5)
c1 = (
    changes(np.array([1, 2, 1, -1, -2, -3.0]), 1e-14)[0] == 1
    and changes(np.array([1, -1, 1, -1.0]), 1e-14)[0] == 3
)
print(f"control c1 (counter): {c1}", flush=True)
if not c1:
    die("READING q1368: VOID (control c1)")
states = []
for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
    try:
        j = json.load(open(f))
        res = float(numstr(j["residual"]))
        K = int(j["K"])
        A = float(numstr(j["A"]))
    except Exception:
        continue
    if j.get("verdict") == "OPTIMAL" and res < 1e-20:
        m, c = vec(j)
        states.append((K, c, A, np.array(m), f))
states.sort(key=lambda t: (t[0], t[1], t[2]))
mono = [(s[0], s[2], bool(np.all(np.diff(s[3]) >= -1e-14))) for s in states if len(s[3]) > 3]
c3 = not bool(np.all(np.diff(np.array([5, 3, 4, 6.0])) >= -1e-14))
print(f"control c3 (a swapped mass vector breaks monotonicity): {c3}", flush=True)
ref = [
    p
    for p in zip(states[:-1], states[1:])
    if p[0][0] == p[1][0] and p[0][1] == p[1][1] and 1e-6 < p[1][2] - p[0][2] < 0.2
]
dd = ref[len(ref) // 2][1][3] - ref[len(ref) // 2][0][3]
sh = dd.copy()
rng.shuffle(sh)
c2 = changes(dd, 1e-14)[0] <= 1 and changes(sh, 1e-14)[0] > 1
print(
    f"control c2 (a shuffled difference vector has more than one sign change; the original has {changes(dd, 1e-14)[0]}): {c2}",
    flush=True,
)
if not (c2 and c3):
    die("READING q1368: VOID (control)")
nm = sum(1 for m in mono if m[2])
print(
    f"(a) states with masses nondecreasing outward: {nm} of {len(mono)}; first failures: {[m[:2] for m in mono if not m[2]][:5]}",
    flush=True,
)
cnt = {}
bad = []
ends = {"outer-": 0, "outer+": 0}
for s1, s2 in ref:
    d = (s2[3] - s1[3]) / (s2[2] - s1[2])
    n, sg = changes(d, 1e-14 / (s2[2] - s1[2]))
    cnt[n] = cnt.get(n, 0) + 1
    if sg:
        ends["outer-" if sg[-1] < 0 else "outer+"] += 1
    if n != 1 or (sg and (sg[-1] > 0 or sg[0] < 0)):
        bad.append((s1[0], s1[2], s2[2], n, sg[0] if sg else 0, sg[-1] if sg else 0))
print(
    f"(b) consecutive same-family pairs {len(ref)}: sign changes of dp_j/dA in position order by count {sorted(cnt.items())}; outer end negative in {ends['outer-']}, positive in {ends['outer+']}; pairs not of the form (inner +, one change, outer -): {len(bad)}",
    flush=True,
)
for b in bad[:12]:
    print(f"  K {b[0]} A {b[1]:.6f} -> {b[2]:.6f}: changes {b[3]}, first sign {b[4]}, last sign {b[5]}")
print("READING q1368 (exploratory, no verdict)")
