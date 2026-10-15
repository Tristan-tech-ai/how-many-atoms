"""q1388 (9 Oct 2026, EXPLORATORY data export, no item, no verdict; py q1388_dump_jb.py K LABEL from temporal_within/small_gap_q206/): J (40-digit, q206 residual_and_jacobian), b = dF/dA (central difference in A at fixed grid, delta 1e-12) and the state of the stored OPTIMAL capacity state of family K at LABEL = start | middle | end (same selection as q1387) or fFRACTION (the state at that fraction of the centred triples) written to q1388_jb_K{K}_{LABEL}.json for offline sign analysis."""

import sys, os, json, glob, time
import q206_gauss_mp as G
from mpmath import mp, mpf, log, pi

mp.dps = 40
G.mp.dps = 40
G.L2P = log(2 * pi) / 2
K = int(sys.argv[1])
label = sys.argv[2]
out = {}
for f in glob.glob(os.path.join("..", "**", f"results_q2*_gauss_K{K}_A*.json"), recursive=True):
    if "eps" in f:
        continue
    try:
        j = json.load(open(f))
        res = float(j["residual"])
    except Exception:
        continue
    if j.get("verdict", "OPTIMAL") != "OPTIMAL" or res > 1e-20 or int(j["K"]) != K:
        continue
    out[round(float(j["A"]), 6)] = j
st = [out[k] for k in sorted(out)]
A = [float(s["A"]) for s in st]
trip = [i for i in range(1, len(st) - 1) if A[i + 1] - A[i - 1] <= 0.0101] or list(range(1, len(st) - 1))
i = (
    {"start": trip[0], "middle": trip[len(trip) // 2], "end": trip[-1]}[label]
    if not label.startswith("f")
    else trip[min(len(trip) - 1, int(float(label[1:]) * len(trip)))]
)
j = st[i]
Av = mpf(j["A"])
u = [mpf(x) for x in j["u"]]
w = [mpf(x) for x in j["w"]]
we = mpf(j["we"])
wc = None if j["wc"] is None else mpf(j["wc"])
y = G.grid(Av)
t0 = time.time()
r0, J = G.residual_and_jacobian(Av, y, u, w, we, wc)
dl = mpf("1e-12")
F = lambda a: G.residual_and_jacobian(a, y, u, w, we, wc, want_J=False)[0]
rp, rm = F(Av + dl), F(Av - dl)
b = [(rp[k] - rm[k]) / (2 * dl) for k in range(len(r0))]
rec = dict(
    K=K,
    label=label,
    A=j["A"],
    m=len(u),
    odd=wc is not None,
    u=j["u"],
    w=j["w"],
    we=j["we"],
    wc=j["wc"],
    r0max=mp.nstr(max(abs(t) for t in r0), 4),
    b=[mp.nstr(t, 40) for t in b],
    J=[[mp.nstr(J[a, c], 40) for c in range(J.cols)] for a in range(J.rows)],
    neighbours=[st[i - 1]["A"], st[i + 1]["A"]],
)
json.dump(rec, open(f"q1388_jb_K{K}_{label}.json", "w"))
print("written", K, label, j["A"], "J", J.rows, "x", J.cols, f"[{time.time() - t0:.0f}s]")
