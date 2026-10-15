import sys, time

sys.path.insert(0, ".")
from flint import arb, ctx
import q920_cap_flint as F
import q921_cap_tools as T

t0 = time.time()
ctx.prec = 600
F.L2P = (arb.pi() * 2).log() / 2
A_str, u, w, we, wc, d = F.load_state("tower/K65/results_q220_gauss_K65_A43.5803.json")
m = len(u)
odd = wc is not None
cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
r0, _, _ = cap.res_jac(v, False, False)
print("start residual", F.maxabs(r0).str(3, radius=False), flush=True)
v, res, ok = T.newton_full(cap, v, lambda *a: None, tol=arb("1e-120"), maxit=8)
print("polished", ok, F.maxabs(res).str(3, radius=False), f"{time.time() - t0:.0f}s")
