"""q1726 (10 Oct 2026, READING, no verdict word): does a geometric tail of the bulk terms close the inner deviation? At the pair of q1708_z_NG14_N90.json the formal laws c24, c26, c28, c30 (NG = 12..15) give the terms D_n(k) = formal(NG = n + 1) - formal(NG = n). With r_k = D_14 / D_13 (c30 minus c28 over c28 minus c26), the geometric remainder after c28 is D_14 / (1 - r_k). It is compared with delta_k = d_k - formal(c28) of the solved state.
NOT BLIND: readings 182 and 184 were read.
Predictions:
 P1 r_k lies between 0.4 and 0.6 for every k = 16..26;
 P2 D_14 / (1 - r_k) differs from delta_k by less than 20 percent for every k = 16..26;
 P3 at k = 34..45 the geometric remainder does not close delta_k (the relative difference exceeds 100 percent for every k).
Control: the formal depths of NG = 14 from the two modules agree to 1e-50 (as in q1725).
Usage: py q1726_bulk_remainder_geometric.py (seconds)."""

import os, sys, json, io, contextlib, importlib.util
import numpy as np

sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
D = json.load(open(os.path.join(here, "q1708_z_NG14_N90.json")))
N = D["N"]
nu, d0 = D["nu_used"], D["d0_used"]
from flint import arb


def formal(modfile, ng):
    os.environ["Q1696_NG"] = str(ng)
    os.environ["Q1703_NG"] = str(ng)
    sys.argv = ["x"]
    spec = importlib.util.spec_from_file_location(
        "m%d_%s" % (ng, os.path.basename(modfile)), os.path.join(here, modfile)
    )
    q = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(q)
    q.NU, q.D0 = nu, d0
    return q.formal_tail(0, N + 1)[0]


f12 = formal("q1697_krawczyk_ng.py", 12)
f13 = formal("q1697_krawczyk_ng.py", 13)
f14 = formal("q1697_krawczyk_ng.py", 14)
f14b = formal("q1704_krawczyk_ng15.py", 14)
f15 = formal("q1704_krawczyk_ng15.py", 15)
fl = lambda f_: np.array([float(t.mid()) for t in f_])
z = [arb(s) for s in D["z"]]
F12, F13, F14, F15 = fl(f12), fl(f13), fl(f14), fl(f15)
delta = np.array([float((z[k] - f14[k]).mid()) for k in range(N)])
D12, D13, D14 = F13 - F12, F14 - F13, F15 - F14
r = D14 / D13
rem = D14 / (1 - r)
ctl = max(float((a - b).abs_upper()) for a, b in zip(f14, f14b)) < 1e-50
print("k   delta_k       D(c24->c26)   D(c26->c28)   D(c28->c30)   r=D14/D13   rem=D14/(1-r)   rem/delta")
for k in list(range(12, 31)) + [34, 36, 38, 40, 45]:
    print(
        "%2d  %+.3e  %+.3e  %+.3e  %+.3e  %+.3f   %+.3e   %+.3f"
        % (
            k,
            delta[k - 1],
            D12[k - 1],
            D13[k - 1],
            D14[k - 1],
            r[k - 1],
            rem[k - 1],
            rem[k - 1] / delta[k - 1],
        )
    )
ks = np.arange(16, 27)
k2 = np.arange(34, 46)
lab = lambda q_: "holds" if q_ else "does not hold"
p1 = bool(np.all((r[ks - 1] >= 0.4) & (r[ks - 1] <= 0.6)))
p2 = bool(np.all(np.abs(rem[ks - 1] / delta[ks - 1] - 1) < 0.20))
p3 = bool(np.all(np.abs(rem[k2 - 1] / delta[k2 - 1] - 1) > 1.0))
print(
    "r over k = 16..26: %.3f to %.3f; rem/delta over k = 16..26: %.3f to %.3f"
    % (
        r[ks - 1].min(),
        r[ks - 1].max(),
        (rem[ks - 1] / delta[ks - 1]).min(),
        (rem[ks - 1] / delta[ks - 1]).max(),
    )
)
print(
    "P1 r in [0.4, 0.6]: %s; P2 remainder within 20 percent: %s; P3 not closed at k = 34..45: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1726 done (labels by the script; no verdict word)")
