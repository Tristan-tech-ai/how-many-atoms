"""c36 (8 Oct 2026, amendment 2052): driver of the half-line stage for the functional W P^-k, built from c34_halfline.py in four runs, as done by hand for k = 3 (amendments 2046, 2047):
 (1) a stage run to D = 36 on the guessed law numbers LAW0 (tag <tag>a), (2) a fit of the law numbers (d0, nu0) to its last converged stage, window d = 14 to 30 (least squares of N_law(d_j) = j - 1/2), (3) a stage run to D = 76 on the fitted law (tag <tag>b), (4) a fit of the offsets OFF1, OFF2 of the derived bulk law to its D = 76 atoms, window d = 15 to 40, (5) the stage run on the derived bulk law (DERIVED_LAW = 1, step 6, buffer 20, 192 bits) to D = 92 (tag <tag>c) whose file c33_<tag>c.json is the half-line input of the chain solver c35.
The driver only runs scripts and fits; no event, no residual. Prints the fitted numbers, G of every stage of the last run and the time.
Environment SKIP_AB = 1 reuses the existing files c33_<TAG>a.json and c33_<TAG>b.json (steps 1 to 3 are not run), and the offsets are fitted to the last converged stage of b if it reached D >= 60 (added 8 Oct, amendment 2058, for small k where the stage D = 76 on the fitted law does not converge).
Usage: py c36_kstage_pipeline.py K LAW0_d0 LAW0_nu0 TAG   (from temporal_within/centre_scripts/)"""

import json, os, subprocess, sys, time
import numpy as np
from scipy.integrate import cumulative_simpson as cs
from scipy.interpolate import CubicSpline as CSp
from scipy.optimize import brentq, least_squares

K, d0, nu0, TAG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pi = np.pi
s0 = 1 / (16 * pi**2)
t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


def run(env, args, log):
    e = dict(os.environ)
    e.update(env)
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable] + args, stdout=f, stderr=subprocess.STDOUT, env=e)
    return r.returncode


def rho_of(d, a):
    return brentq(lambda r: r**3 - 3 * r / (8 * pi**2) - 1 / (32 * pi**2 * r) - 3 * s0 * (d + a), 0.05, 400)


def N_of(d, a, n):
    r = rho_of(d, a)
    return 4 * pi**2 * r**4 - r**2 + np.log(r) / 6 + n


def last_conv(jf):
    st = [s for s in json.load(open(jf))["stages"] if s["conv"]]
    return st[-1]


def fit_law(st, win):
    p = np.array(st["p"][: st["nfree"]])
    j = np.arange(1, len(p) + 1) - 0.5
    m = (p >= win[0]) & (p <= win[1])
    r = least_squares(lambda q: np.array([N_of(x, q[0], q[1]) for x in p[m]]) - j[m], [0.3, 0.05])
    return r.x


_k = 1 / (8 * pi**2)
_k4 = 1 / (64 * pi**4) - 1 / (96 * pi**2)
_c6 = 1 / (512 * pi**6) - 13 / (768 * pi**4)
_c8 = 1 / (4096 * pi**8) - 17 / (1536 * pi**6) - 1 / (3072 * pi**4)
_c10 = 1 / (32768 * pi**10) - 29 / (6144 * pi**8) - 109 / (368640 * pi**6)
_R = np.linspace(0.45, 2.6, 500001)
_f = 1 + _k / _R**2 + _k4 / _R**4 + _c6 / _R**6 + _c8 / _R**8 + _c10 / _R**10
_D = cs(_R**2 / (s0 * _f), x=_R, initial=0)
_N = cs(_R**3 / (s0 * _f), x=_R, initial=0)
_rinv = CSp(_D, _R)
_Nr = CSp(_R, _N)


def Nd(d, o1, o2):
    return float(_Nr(_rinv(d + o1))) + o2


def fit_off(st, win):
    p = np.array(st["p"][: st["nfree"]])
    j = np.arange(1, len(p) + 1) - 0.5
    m = (p >= win[0]) & (p <= win[1])
    r = least_squares(lambda q: np.array([Nd(x, q[0], q[1]) for x in p[m]]) - j[m], [-3.0, 1.2])
    return r.x, float(np.sqrt(np.mean(r.fun**2)))


SKIP = os.environ.get("SKIP_AB") == "1"
if not SKIP:
    rc = run(
        {"LAW0": f"{d0},{nu0}"},
        ["c34_halfline.py", K, "36", "192", "8", "14", f"{TAG}a", "20"],
        f"c33_{TAG}a.log",
    )
    say("run a exit", rc)
    sa = last_conv(f"c33_{TAG}a.json")
    say("last converged stage of a: D", sa["D"])
    law = fit_law(sa, (14, 30))
    say("fitted law numbers", law)
    rc = run(
        {"LAW0": f"{law[0]:.5f},{law[1]:.5f}"},
        ["c34_halfline.py", K, "76", "192", "8", "14", f"{TAG}b", "20"],
        f"c33_{TAG}b.log",
    )
    say("run b exit", rc)
else:
    law = np.array([float(d0), float(nu0)])
sb = [s for s in json.load(open(f"c33_{TAG}b.json"))["stages"] if s["conv"]][-1]
say("last converged stage of b: D", sb["D"])
if sb["D"] < (60 if SKIP else 76):
    say("STOP: run b did not reach the required D")
    sys.exit(1)
off, rms = fit_off(sb, (15, 40))
say("derived-law offsets OFF1, OFF2", off, "rms", rms)
rc = run(
    {"DERIVED_LAW": "1", "OFF1": f"{off[0]:.5f}", "OFF2": f"{off[1]:.5f}", "LAW0": "0,0"},
    ["c34_halfline.py", K, "92", "192", "6", "20", f"{TAG}c", "20"],
    f"c33_{TAG}c.log",
)
say("run c exit", rc)
sc = json.load(open(f"c33_{TAG}c.json"))["stages"]
say(
    "run c stages",
    len(sc),
    "last D",
    sc[-1]["D"],
    "conv",
    sc[-1]["conv"],
    "G",
    [round(s["Gl"], 6) for s in sc[-5:]],
)
print(
    "RESULT c36:",
    TAG,
    "k",
    K,
    "OFF1",
    round(float(off[0]), 5),
    "OFF2",
    round(float(off[1]), 5),
    "law",
    [round(float(x), 5) for x in law],
    "last D",
    sc[-1]["D"],
)
