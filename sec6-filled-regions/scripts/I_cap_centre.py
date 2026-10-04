"""I_cap_centre (03 Oct 2026, test of 1451): TEXT COPY of I_rd_centre for FUNC cap (LAM from the state). Centre excess D(0) - C of a solved rd boundary ring (a helper script state json from D_solver,
FUNC lfp), with D_core2.Ring's own risk function (_D_lfp, the record's q786 text) at q = (0, 0); also r - C on a small interior grid
(s z(t), s in {0.25, 0.5, 0.75, 0.9}, t in 0..90 deg step 15) to check the region.
Usage: py I_lfp_centre.py STATE.json [NGH] [PREC]"""

import sys, json
from flint import arb, ctx
from D_core2 import Ring

d = json.load(open(sys.argv[1]))
NGH = int(sys.argv[2]) if len(sys.argv) > 2 else int(d.get("ngh", 120))
PREC = int(sys.argv[3]) if len(sys.argv) > 3 else int(d.get("prec", 128))
R = Ring("cap", d["rp"], d["X"], d["Y"], len(d["G"]), NGH=NGH, prec=PREC, lam=d.get("lam", "0"))
G = [arb(x) for x in d["G"]]
m = [arb(x) for x in d["m"]]
C = arb(d["C"])
rm = arb(d["rm"])
th, w = R.atoms(G, m)
xs = [R.pos(t, rm) for t in th]
r0 = R.D_grad((arb(0), arb(0)), xs, w, grad=False)
out = [
    f"rp {d['rp']} rm {d['rm']} K {len(th)}: D(0) - C = {float((r0 - C).mid()):+.10e} (C {float(C.mid()):.12f}, NGH {NGH}, prec {PREC})"
]
worst = None
for s in ["0.25", "0.5", "0.75", "0.9"]:
    for deg in range(0, 91, 15):
        t = arb(deg) * R.pi / 180
        p = R.pos(t, rm)
        q = (arb(s) * p[0], arb(s) * p[1])
        v = float((R.D_grad(q, xs, w, grad=False) - C).mid())
        if worst is None or v > worst[0]:
            worst = (v, s, deg)
out.append(f"   interior grid max r - C {worst[0]:+.4e} at s {worst[1]}, {worst[2]} deg")
print("\n".join(out), flush=True)
