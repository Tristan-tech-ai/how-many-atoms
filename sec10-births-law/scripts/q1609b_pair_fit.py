"""q1609b (10 Oct 2026, READING, post hoc, written after q1609 was read, no prediction registered, no verdict word): the interior deviations of the exact 120-atom block from the formal law are fitted by a change of the edge pair, delta_k = d(delta d0) + d(nu0) * s_k with s_k = d d_k / d k of the formal law (central difference of the formal depths). The fit window is k = 30..70, with the checks k = 30..60 and 40..80. Usage: py q1609b_pair_fit.py"""

import os, sys, json, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
spec = importlib.util.spec_from_file_location("q1598m", os.path.join(here, "q1598_krawczyk_free90_120.py"))
q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(q)
from flint import arb

N = 120
z = [arb(s) for s in json.load(open(os.path.join(here, "q1601_z_N120.json")))]
dt, wt = q.formal_tail(0, N + 1)
d = np.array([float(z[k]) for k in range(N)])
f = np.array([float(t) for t in dt])
delta = d - f[:N]
s = np.array([(f[k + 1] - f[k - 1]) / 2 if k > 0 else f[1] - f[0] for k in range(N)])
for lo, hi in ((30, 70), (30, 60), (40, 80), (50, 90)):
    idx = np.arange(lo - 1, hi)
    A = np.vstack([np.ones(len(idx)), s[idx]]).T
    coef, *_ = np.linalg.lstsq(A, delta[idx], rcond=None)
    res = delta[idx] - A @ coef
    print(
        "window k = %d..%d: delta d0 = %+.4e, delta nu0 = %+.4e, rms of the rest %.2e, largest %.2e (largest |delta| in the window %.2e)"
        % (lo, hi, coef[0], coef[1], np.sqrt((res**2).mean()), np.abs(res).max(), np.abs(delta[idx]).max())
    )
print(
    "pair used: nu0 = -0.093773586, d0 = +0.001159841; spacing s_k at k = 30, 50, 70: %.4f %.4f %.4f"
    % (s[29], s[49], s[69])
)
print("READING q1609b done (post hoc; no verdict word)")
