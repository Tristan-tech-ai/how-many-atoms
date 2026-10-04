"""q814 (03 Oct 2026, my check of the K 20 event on lam 0.02, rp 1.2): D - C at chosen angles under capfix (FIX_N, FIX_B)
and Gauss-Hermite (NGH), using the D_core2 (unchanged). Usage (cwd ): py q814 STATE.json "capfix:40:20,capfix:56:24,cap:360" PREC
"""

import sys, os, time, json

sys.path.insert(0, ".")
from flint import arb, ctx
import D_core2 as DC

d = json.load(open(sys.argv[1]))
prec = int(sys.argv[3])
ctx.prec = prec
rm = arb(d["rm"])
pi = arb.pi()
v = [arb(g) for g in d["G"]] + [arb(x) for x in d["m"]] + [arb(d["C"])]
degs = ["0", "0.5", "0.85", "1.2", "2", "10", "45", "89.5", "90"]
ts = [arb(x) * pi / 180 for x in degs]
print("state", sys.argv[1], "rm", d["rm"], "X Y nG", d["X"], d["Y"], len(d["G"]), "angles", degs, flush=True)
for spec in sys.argv[2].split(","):
    p = spec.split(":")
    t0 = time.time()
    if p[0] == "capfix":
        os.environ["FIX_N"] = p[1]
        os.environ["FIX_B"] = p[2]
        R = DC.Ring("capfix", "1.2", d["X"], d["Y"], len(d["G"]), prec=prec, lam="0.02")
    else:
        R = DC.Ring("cap", "1.2", d["X"], d["Y"], len(d["G"]), NGH=int(p[1]), prec=prec, lam="0.02")
    vals = R.kkt_many(v, rm, ts)
    print(spec, f"{time.time()-t0:.0f}s", " ".join(f"{float(x.mid()):+.4e}" for x in vals), flush=True)
