"""q1728 (10 Oct 2026, READING, no verdict word): are the one-order steps of the ladder pair the compensation of the added bulk term on the fit window? The solved state is fixed (q1708_z_NG14_N90.json); changing the formal law from NG to NG + 1 changes the deviation of the atoms by -D_NG(k), D_NG = formal(NG + 1) - formal(NG), so the two-parameter fit of the deviation on the window 30..60 (a + b s_k, s_k as in q1698) changes by -(a_D + b_D s_k), with (a_D, b_D) the fit of D_NG. Predicted step of the pair: delta d0 = -a_D, delta nu0 = -b_D. Observed steps: p13 - p12 (q1714 minus q1698 NG = 12), p14 - p13, p15 - p14 of the self-consistent pairs (q1698, q1714, q1705 result files).
NOT BLIND: the observed steps were read (readings 177, 168); the fit of D_NG had not been computed.
Predictions:
 P1 for the steps 12 -> 13, 13 -> 14 and 14 -> 15 the predicted delta nu0 and delta d0 agree with the observed ones within 15 percent each;
 P2 the ratios of successive predicted steps (13 -> 14 over 12 -> 13 and 14 -> 15 over 13 -> 14) agree with the observed ratios 0.404 and 0.436 within 0.03.
Control: the fit of the deviation delta_k of the state itself on the window 30..60 returns corrections below 1e-14.
Usage: py q1728_pair_steps_from_bulk_terms.py (seconds)."""

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
f = {
    12: formal("q1697_krawczyk_ng.py", 12),
    13: formal("q1697_krawczyk_ng.py", 13),
    14: formal("q1697_krawczyk_ng.py", 14),
    15: formal("q1704_krawczyk_ng15.py", 15),
}
F = {k: fl(v) for k, v in f.items()}
z = [arb(s) for s in D["z"]]
delta = np.array([float((z[k] - f[14][k]).mid()) for k in range(N)])
s = np.array([(F[14][k + 1] - F[14][k - 1]) / 2 if k > 0 else F[14][1] - F[14][0] for k in range(N)])
idx = np.arange(29, 60)


def fit(y):
    A = np.vstack([np.ones(len(idx)), s[idx]]).T
    coef, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
    return coef


ctl = float(np.max(np.abs(fit(delta)))) < 1e-14
P = {
    ng: tuple(json.load(open(os.path.join(here, fn)))["final_pair"])
    for ng, fn in (
        (12, "q1698_res_NG12_N90.json"),
        (13, "q1714_res_NG13_N90.json"),
        (14, "q1698_res_NG14_N90.json"),
        (15, "q1705_res_NG15_N90.json"),
    )
}
pred = {}
obs = {}
for ng in (12, 13, 14):
    Dn = F[ng + 1][:N] - F[ng][:N]
    a, b = fit(Dn)
    pred[ng + 1] = (-b, -a)  # (delta nu0, delta d0)
    obs[ng + 1] = (P[ng + 1][0] - P[ng][0], P[ng + 1][1] - P[ng][1])
for ng in (13, 14, 15):
    print(
        "step %d -> %d: predicted delta nu0 %+.3e, delta d0 %+.3e; observed %+.3e, %+.3e; ratios %.3f, %.3f"
        % (ng - 1, ng, *pred[ng], *obs[ng], pred[ng][0] / obs[ng][0], pred[ng][1] / obs[ng][1])
    )
pr = [pred[14][0] / pred[13][0], pred[15][0] / pred[14][0]]
orr = [obs[14][0] / obs[13][0], obs[15][0] / obs[14][0]]
print("ratios of successive steps in nu0: predicted %.3f, %.3f; observed %.3f, %.3f" % (*pr, *orr))
lab = lambda q_: "holds" if q_ else "does not hold"
p1 = all(abs(pred[ng][i] / obs[ng][i] - 1) < 0.15 for ng in (13, 14, 15) for i in (0, 1))
p2 = all(abs(a - b) < 0.03 for a, b in zip(pr, orr))
print(
    "P1 steps within 15 percent: %s; P2 step ratios within 0.03: %s; control: %s"
    % (lab(p1), lab(p2), lab(ctl))
)
print("READING q1728 done (labels by the script; no verdict word)")
