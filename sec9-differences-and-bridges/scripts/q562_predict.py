"""Q562 part 2: compute the blind predictions (see q562_prereg.txt part 1). One case per call; bisection on a where the paper fixes D
or I. Output: q562_pred_<case>.json. Usage: py q562_predict.py CASE"""

import sys, json, math
import numpy as np
import q562_rdlib as R

case = sys.argv[1]
say = lambda *a: print(*a, flush=True)
T = [
    1.732051,
    2.979939,
    4.099981,
    5.143980,
    6.132694,
    7.078065,
    7.988062,
    8.868382,
    9.723280,
    10.556038,
    11.369250,
    12.165018,
]


def bisect(f, lo, hi, it=14):
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


out = {"case": case}
if case in ("A9", "A10"):
    beta = {"A9": 0.1, "A10": 0.2}[case]
    a = 8 * math.sqrt(2 * beta)
    mode = "unif"
elif case in ("A11", "A12", "A8"):
    D = {"A11": 4.0, "A12": 3.0, "A8": 8.0}[case]
    mode = "unif"
    a = bisect(lambda a: R.solve(a)[4] * 64 / a**2 - D, 1.8, 12.0)
elif case in ("A13", "A3", "A4"):
    Itar = {"A13": math.log(2), "A3": 1.0, "A4": 1.2}[case]
    mode = "unif"
    a = bisect(lambda a: R.solve(a)[5] - Itar, 1.8, 14.0)
elif case in ("A1", "A2"):
    Itar = {"A1": 0.33, "A2": 0.39}[case]
    mode = "tnorm"
    a = bisect(lambda a: R.solve(a, "tnorm")[5] - Itar, 1.0, 14.0)
elif case == "A5":
    a = 3 * math.sqrt(2)
    mode = "tnorm"
elif case == "A5alt":
    a = 6.0
    mode = "tnorm"
K, ok, ys, ws, Dn, I = R.solve(a, mode)
near = min(abs(a - t) for t in T) if mode == "unif" else None
out.update(
    {
        "a": a,
        "mode": mode,
        "K": K,
        "certified": ok,
        "D_noise": Dn,
        "I_nats": I,
        "atoms_noise_units": ys.tolist(),
        "weights": ws.tolist(),
        "nearest_uniform_transition_distance": near,
    }
)
json.dump(out, open(f"q562_pred_{case}.json", "w"), indent=1)
say(
    f"{case}: a = {a:.4f} ({mode}), K = {K}, certified {ok}, D_noise = {Dn:.5f}, I = {I:.5f} nats"
    + (f", nearest transition {near:.3f}" if near is not None else "")
)
