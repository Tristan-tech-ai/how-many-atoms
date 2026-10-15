"""q1709 (10 Oct 2026, READING, no verdict word): the deviation delta_k = d_k - formal_k of the solved atoms of the clean N = 90 block from the formal law through c28 at the pair used for the state (q1708_z_NG14_N90.json), for k = 1..60.
Predictions written before q1708_z_NG14_N90.json exists (the run q1708 was started at 12:06:27 by the clock):
 P1 |delta_k| falls monotonically in k over k = 10..45;
 P2 delta_k has one sign over k = 10..45;
 P3 log|delta_k| is linear in k over k = 20..40 (correlation of the straight-line fit above 0.98) with a slope between -0.5 and -0.05 per atom (the signature of an exponentially small sector);
 P4 control: the two-parameter fit of q1698 (delta_k = a + b s_k, s_k = half the difference of the neighbouring formal depths) on k = 20..40 leaves an rms of 4.05e-11, reproduced to 1 percent.
If the file is missing the script prints 'VOID: state file missing' and exits with code 2. Usage: py q1709_inner_deviation_profile.py (seconds).
"""

import os, sys, json, io, contextlib, importlib.util
import numpy as np

sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
f = os.path.join(here, "q1708_z_NG14_N90.json")
if not os.path.exists(f):
    print("VOID: state file missing")
    sys.exit(2)
D = json.load(open(f))
N = D["N"]
nu, d0 = D["nu_used"], D["d0_used"]
os.environ["Q1696_NG"] = str(D["NG"])
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("q1697m", os.path.join(here, "q1697_krawczyk_ng.py"))
q = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(q)
from flint import arb

q.NU, q.D0 = nu, d0
dt, wt = q.formal_tail(0, N + 1)
form = np.array([float(t) for t in dt])
z = [arb(s) for s in D["z"]]
delta = np.array([float((z[k] - dt[k]).mid()) for k in range(N)])
s = np.array([(form[k + 1] - form[k - 1]) / 2 if k > 0 else form[1] - form[0] for k in range(N)])
idx = np.arange(19, 40)
A = np.vstack([np.ones(len(idx)), s[idx]]).T
coef, *_ = np.linalg.lstsq(A, delta[idx], rcond=None)
res = delta[idx] - A @ coef
rms = float(np.sqrt((res**2).mean()))
print(
    "pair used (%.12f, %.12f); delta_k at k = 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60: %s"
    % (nu, d0, " ".join("%+.2e" % delta[k - 1] for k in (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60)))
)
k1 = np.arange(10, 46)
a1 = np.abs(delta[k1 - 1])
mono = bool(np.all(np.diff(a1) < 0))
sign = bool(np.all(delta[k1 - 1] > 0) or np.all(delta[k1 - 1] < 0))
k2 = np.arange(20, 41)
l2 = np.log(np.abs(delta[k2 - 1]))
sl, ic = np.polyfit(k2, l2, 1)
cor = float(np.corrcoef(k2, l2)[0, 1])
print(
    "monotone fall over k = 10..45: %s; one sign: %s; log-linear fit over k = 20..40: slope %+.4f per atom, correlation %+.4f; window fit rest rms on 20..40: %.3e"
    % (mono, sign, sl, cor, rms)
)
lab = lambda q_: "holds" if q_ else "does not hold"
print(
    "P1 %s; P2 %s; P3 %s; P4 %s"
    % (
        lab(mono),
        lab(sign),
        lab(abs(cor) > 0.98 and -0.5 <= sl <= -0.05),
        lab(abs(rms / 4.05e-11 - 1) < 0.01),
    )
)
print("READING q1709 done (labels by the script; no verdict word)")
