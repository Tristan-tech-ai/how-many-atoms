"""q1388b (9 Oct 2026, EXPLORATORY data export, no item, no verdict; py q1388b_dump_gap.py from temporal_within/small_gap_q206/): J, b = dF/dA and the state for each of the 27 certified gap states of q1384_states.json (A = 16.3 to 19.9, K = 19 to 24), written to gapjb/q1388_jb_K{K}_A{A}.json in the format of q1388 (label = 'gap')."""

import sys, os, json, time
import q206_gauss_mp as G
from mpmath import mp, mpf, log, pi

mp.dps = 40
G.mp.dps = 40
G.L2P = log(2 * pi) / 2
os.makedirs("gapjb", exist_ok=True)
st = json.load(open("q1384_states.json"))
st.sort(key=lambda s: s["A"])
t0 = time.time()
for s in st:
    Av = mpf(s["A"])
    u = [mpf(x) for x in s["u"]]
    w = [mpf(x) for x in s["w"]]
    we = mpf(s["we"])
    wc = None if s["wc"] is None else mpf(s["wc"])
    y = G.grid(Av)
    r0, J = G.residual_and_jacobian(Av, y, u, w, we, wc)
    dl = mpf("1e-12")
    F = lambda a: G.residual_and_jacobian(a, y, u, w, we, wc, want_J=False)[0]
    rp, rm = F(Av + dl), F(Av - dl)
    b = [(rp[k] - rm[k]) / (2 * dl) for k in range(len(r0))]
    rec = dict(
        K=s["K"],
        label="gap",
        A=str(s["A"]),
        m=len(u),
        odd=wc is not None,
        u=s["u"],
        w=s["w"],
        we=s["we"],
        wc=s["wc"],
        r0max=mp.nstr(max(abs(t) for t in r0), 4),
        b=[mp.nstr(t, 40) for t in b],
        J=[[mp.nstr(J[a, c], 40) for c in range(J.cols)] for a in range(J.rows)],
    )
    json.dump(rec, open(os.path.join("gapjb", f"q1388_jb_K{s['K']}_A{s['A']}.json"), "w"))
    print("written", s["K"], s["A"], "residual", rec["r0max"], f"[{time.time() - t0:.0f}s]", flush=True)
print("done", len(st))
