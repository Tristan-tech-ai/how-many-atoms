"""q1709b (10 Oct 2026, READING, POST HOC, no predictions; follows q1709 whose P1 to P3 did not hold): the deviation delta_k of q1708_z_NG14_N90.json from the formal law c28, k = 1..60, as signed values and as successive ratios, to see where the exponential fall ends and the floor begins. Written after the output of q1709 had been read, so nothing here is a test."""

import os, sys, json, io, contextlib, importlib.util
import numpy as np

sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
D = json.load(open(os.path.join(here, "q1708_z_NG14_N90.json")))
N = D["N"]
os.environ["Q1696_NG"] = str(D["NG"])
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("q1697m", os.path.join(here, "q1697_krawczyk_ng.py"))
q = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(q)
from flint import arb

q.NU, q.D0 = D["nu_used"], D["d0_used"]
dt, wt = q.formal_tail(0, N + 1)
z = [arb(s) for s in D["z"]]
delta = np.array([float((z[k] - dt[k]).mid()) for k in range(N)])
for k in range(8, 46):
    print("k %2d  delta %+.3e  ratio to previous %+.3f" % (k, delta[k - 1], delta[k - 1] / delta[k - 2]))
k1 = np.arange(12, 31)
sl, ic = np.polyfit(k1, np.log(np.abs(delta[k1 - 1])), 1)
cor = np.corrcoef(k1, np.log(np.abs(delta[k1 - 1])))[0, 1]
print(
    "log-linear fit over k = 12..30: slope %+.4f per atom, correlation %+.5f; largest relative departure of the fit %.3f"
    % (sl, cor, float(np.max(np.abs(np.exp(np.polyval([sl, ic], k1)) / np.abs(delta[k1 - 1]) - 1))))
)
sg = np.where(np.diff(np.sign(delta[9:60])) != 0)[0] + 10
print("sign changes of delta_k between k = 10 and 60 after k = %s" % list(sg))
print(
    "rms of delta_k over k = 35..60: %.3e; over k = 45..75: %.3e"
    % (float(np.sqrt((delta[34:60] ** 2).mean())), float(np.sqrt((delta[44:75] ** 2).mean())))
)
print("READING q1709b done (post hoc; no labels)")
