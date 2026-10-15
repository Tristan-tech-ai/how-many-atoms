"""q1727 (10 Oct 2026, READING, no verdict word): is the plateau of about -2e-12 at k = 34..45 (reading 170, 184, 185) the compensation of the pair fit? The pair of the solved state was fitted on the window 30..60 (q1698), so the deviation from the formal law c28 at the used pair is delta_k = T_k - (a + b s_k) + rest, with T_k the neglected terms of the bulk series and (a, b) the two-parameter fit of T_k on the same window with s_k as in q1698. T_k is taken as the geometric remainder D14 / (1 - r_k) of q1726.
NOT BLIND: delta_k, the remainder and the plateau were read (readings 184 and 185); the compensation had not been computed.
Predictions:
 P1 |delta_k - (T_k - fit(T_k))| is below 20 percent of |delta_k| for every k = 17..24;
 P2 |delta_k - (T_k - fit(T_k))| is below 0.5e-12 for every k = 34..45 (the plateau closes);
 P3 the rms of delta_k - (T_k - fit(T_k)) over k = 30..60 is below 1e-12.
Control: the same fit applied to delta_k alone on the window 30..60 returns corrections below 1e-14 (the state used was fitted on this window).
Usage: py q1727_plateau_from_pair_fit.py (seconds)."""

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


fl = lambda f_: np.array([float(t.mid()) for t in f_])
F13 = fl(formal("q1697_krawczyk_ng.py", 13))
f14 = formal("q1697_krawczyk_ng.py", 14)
F14 = fl(f14)
F15 = fl(formal("q1704_krawczyk_ng15.py", 15))
z = [arb(s) for s in D["z"]]
delta = np.array([float((z[k] - f14[k]).mid()) for k in range(N)])
F14full = F14.copy()
F13, F14, F15 = F13[:N], F14[:N], F15[:N]
D13, D14 = F14 - F13, F15 - F14
r = D14 / D13
T = D14 / (1 - r)
s = np.array([(F14full[k + 1] - F14full[k - 1]) / 2 if k > 0 else F14full[1] - F14full[0] for k in range(N)])
idx = np.arange(29, 60)


def fitcorr(y):
    A = np.vstack([np.ones(len(idx)), s[idx]]).T
    coef, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
    return coef[0] + coef[1] * s


cd = fitcorr(delta)
ctl = float(np.max(np.abs(cd[idx]))) < 1e-14
model = T - fitcorr(T)
resid = delta - model
print(
    "control: fit of delta_k on k = 30..60 gives corrections of at most %.2e" % float(np.max(np.abs(cd[idx])))
)
print("k    delta_k       T_k         T_k - fit(T_k)   residual   residual/delta")
for k in list(range(14, 31, 2)) + [32, 34, 36, 38, 40, 42, 45, 50, 55, 60]:
    print(
        "%2d  %+.3e  %+.3e  %+.3e   %+.3e  %+.3f"
        % (k, delta[k - 1], T[k - 1], model[k - 1], resid[k - 1], resid[k - 1] / delta[k - 1])
    )
ks = np.arange(17, 25)
k2 = np.arange(34, 46)
lab = lambda q_: "holds" if q_ else "does not hold"
p1 = bool(np.all(np.abs(resid[ks - 1]) < 0.2 * np.abs(delta[ks - 1])))
p2 = bool(np.all(np.abs(resid[k2 - 1]) < 0.5e-12))
p3 = float(np.sqrt((resid[29:60] ** 2).mean())) < 1e-12
print(
    "rms of the residual over k = 30..60: %.3e; largest |residual| over k = 34..45: %.3e"
    % (float(np.sqrt((resid[29:60] ** 2).mean())), float(np.max(np.abs(resid[k2 - 1]))))
)
print("P1 %s; P2 %s; P3 %s; control: %s" % (lab(p1), lab(p2), lab(p3), lab(ctl)))
print("READING q1727 done (labels by the script; no verdict word)")
