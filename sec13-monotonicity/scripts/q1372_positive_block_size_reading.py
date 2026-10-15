"""q1372 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1372_positive_block_size_reading.py from temporal_within/centre_scripts/): size of the inner block of atoms with positive mass flow dp_j/dA (the part of the one-sign-change structure of q1368 that is positive). For every consecutive same-family pair of the stored capacity states (same K, same centre structure, 1e-6 < A2 - A1 < 0.2) the number of right-half atoms (centre atom included, edge atom included) whose mass difference exceeds 1e-14. Printed: the distribution over pairs per centre structure, and per family the range. Expected before the run: unknown; the inner block should be small compared with K / 2 and grow with the age of the family."""

import glob, json, os, re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


def vec(j):
    w = [float(numstr(x)) for x in j["w"]]
    we = float(numstr(j["we"]))
    wc = None if j.get("wc") in (None, "None") else float(numstr(j["wc"]))
    return ([wc] if wc is not None else []) + w + [we], wc is not None


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
        states.append((K, c, A, np.array(m)))
states.sort(key=lambda t: (t[0], t[1], t[2]))
cnt = {}
byk = {}
age = []
fam_first = {}
for s in states:
    fam_first.setdefault((s[0], s[1]), s[2])
for s1, s2 in zip(states[:-1], states[1:]):
    if s1[0] == s2[0] and s1[1] == s2[1] and 1e-6 < s2[2] - s1[2] < 0.2 and s1[0] >= 60:
        d = s2[3] - s1[3]
        npos = int(np.sum(d > 1e-14))
        key = (s1[1], npos)
        cnt[key] = cnt.get(key, 0) + 1
        byk.setdefault((s1[0], s1[1]), set()).add(npos)
        age.append((s1[2] - fam_first[(s1[0], s1[1])], npos))
print("(centre atom, number of right-half atoms with positive mass flow) -> pairs:", sorted(cnt.items()))
print(
    "per family (K, centre): range of the block size:",
    {k: (min(v), max(v)) for k, v in sorted(byk.items())},
    flush=True,
)
a = np.array(age)
print(
    "correlation between the age of the family (A - A_first) and the block size:",
    float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
    "; pairs",
    len(a),
)
print("READING q1372 (exploratory, no verdict)")
