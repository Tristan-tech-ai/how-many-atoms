"""q1443 = q1440 with the walk and the event location removed (9 Oct 2026, INSTRUMENT): converges ONE state of the family K near the margin stage A_target = A_(K-1) + 0.97 (A_K - A_(K-1)) (events K_b - 1 and K_b read from OUTDIR_EVENTS / fw_prod / fw_cap_par) from the stage seed, with starting amplitudes A_target + offset, and saves it in the file format of the walker (results_q1411_gauss_K<K>_A<A>.json) in OUTDIR so that q1416 can take it as the nearest state. Original docstring: q1440 (9 Oct 2026, INSTRUMENT / COMPUTATION, no verdict): locate ONE capacity transition K_b by itself (many in parallel on separate cores), the capacity counterpart of q1437_imm_event_at.py. The state is seeded on the capacity half-line stage D = 134 of the chain with B1 = 20 (centre_scripts/c50_capc_D140_B20.json: depths d and masses m of the atoms, the atom 1 being the wall atom) at A0 = a_pred(K) - BACK, where a_pred is the stage-134 prediction of the registered table q1439_cap_b20_table.json. Interior atoms j = 2 .. K//2 (j counted from the edge) sit at x with N(A - x) - dM(x/v) = j - 1/2 (N the counting function of the stage, dM the centre correction of the bulk comb, v = 4 pi rho), masses m_j, wall mass m_1, centre mass 1/rho for odd K, all divided by the total mass. The seed is converged by the Newton of q1414 (192 bits, band-limited, H = 0.05, tol 1e-30, 25 steps, then up to 60 more when it stalls between 1e-30 and 1e-2), then walked up with the walker of q1411 to a_pred + FWD and the event is located with Walker.locate. The seed only chooses the starting point. Output OUTDIR/event_Kb<K>.json in the format of q1411.
Usage: py q1440_cap_event_at.py K OUTDIR [BACK] [FWD]; env Q1437_BACKS (comma list of offsets) replaces the default list (from temporal_within/small_gap_q206/).
"""

import sys, os, json, time, re

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("Q1411_BAND", "1")
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares
from scipy.integrate import quad
from flint import arb
import q1411_fast_walk as WK

K = int(sys.argv[1])
out = sys.argv[2]
CHAIN = False
KS = K  # chain mode: the seed is that of K_b - 1 (an odd chain starts easily), and the walker passes through its split into the family K and on to the event K_b
TAB = json.load(open(os.path.join(HERE, "..", "centre_scripts", "q1439_cap_b20_table.json")))["stages"]["134"]
pred = TAB[str(K)]
predS = pred
pi = np.pi
s0 = 1 / (16 * pi**2)
st = [
    s
    for s in json.load(open(os.path.join(HERE, "..", "centre_scripts", "c50_capc_D140_B20.json")))["stages"]
    if s["D"] == 134.0
][0]
d = np.array(st["d"][: 1 + st["nfree"]])
mlist = np.array(st["m"])
law = tuple(st["law"]) if False else None
# the derived-law offsets of the chain (OFF1, OFF2) are used by q1183.N_of through the plain law fit, as in q1439: refit the plain law on atoms 4..30
sys.path.insert(0, os.path.join(HERE, ".."))
cwd = os.getcwd()
os.chdir(os.path.join(HERE, ".."))
import q1183_births_residual_halfline_depth_reading as Q

os.chdir(cwd)
jj = np.arange(1, len(d) + 1) - 0.5
mm = (d >= 4) & (d <= 30)
law = tuple(
    least_squares(lambda q: np.array([Q.N_of(x, q[0], q[1]) for x in d[mm]]) - jj[mm], [0.0558, 0.5482]).x
)
Nf, _ = Q.spline(d[d <= 90.0], law)


def rho_of(dd):
    return Q.rho_of(dd, law[0])


def dM(xi):
    xi = abs(xi)
    return 1 / 12 if xi == 0 else quad(lambda t: t * (1 / np.tanh(np.pi * t) - 1), xi, np.inf, limit=200)[0]


def seed(A):
    npos = KS // 2
    v = 4 * pi * rho_of(A)
    odd = KS % 2 == 1
    u = []
    w = []
    for j in range(npos, 1, -1):  # edge index j = npos (innermost) ... 2
        d_j = brentq(lambda t: Nf(t) - (j - 0.5), 1e-6, 120.0, xtol=1e-13)
        x0 = A - d_j
        try:
            x = brentq(
                lambda x: Nf(A - x) - dM(x / v) - (j - 0.5),
                max(1e-6, x0 - 1.0),
                min(x0 + 1.0, A - 1e-6),
                xtol=1e-13,
            )
        except ValueError:
            x = x0
        u.append(x)
        w.append(float(mlist[j - 1]))
    we = float(mlist[0])
    wc = (1 / rho_of(A)) if odd else None
    M = 2 * (sum(w) + we) + (wc or 0)
    return u, [t / M for t in w], we / M, (None if wc is None else wc / M)


from decimal import Decimal


def event_A(Kb):
    for d_ in ("fw_prod", "fw_cap_par", "fw_cap_par_merged"):
        f = os.path.join(HERE, d_, f"event_Kb{Kb}.json")
        if os.path.exists(f):
            return Decimal(json.load(open(f))["A_event"])
    raise FileNotFoundError(f"event K_b {Kb}")


a0 = event_A(K - 1)
a1 = event_A(K)
At = float(a0 + Decimal("0.97") * (a1 - a0))
W = WK.Walker(192, out, H=0.05)
F = W.F
t0 = time.time()
ok = False
OFFS = [
    0.0,
    0.0003,
    -0.0003,
    0.0006,
    -0.0006,
    0.0009,
    -0.0009,
    0.0012,
    -0.0012,
    0.0015,
    -0.0015,
    0.0018,
    -0.0018,
    0.0021,
    -0.0021,
    0.0024,
    -0.0024,
    0.0027,
    -0.0027,
    0.0030,
    -0.0030,
]
if os.environ.get("Q1443_OFFS"):
    OFFS = [
        float(t) for t in os.environ["Q1443_OFFS"].split(",")
    ]  # replacement list of offsets (seed procedure only)
for off in OFFS:
    A0 = At + off
    u0, w0, we0, wc0 = seed(A0)
    A = arb(A0)
    u = [arb(x) for x in u0]
    w = [arb(x) for x in w0]
    we = arb(we0)
    wc = None if wc0 is None else arb(wc0)
    try:
        u, w, we, wc, res, k = F.newton(A, u, w, we, wc, tol="1e-30", steps=25)
        if arb("1e-30") < res < arb("1e-2"):
            u, w, we, wc, res, k2 = F.newton(A, u, w, we, wc, tol="1e-30", steps=60)
            k += k2
    except Exception as e:
        WK.say(f"K = {K}: row seed at A0 = {A0:.6f} (offset {off:+.4f}); exception {e!r}")
        continue
    WK.say(
        f"K = {K}: row seed at A0 = {A0:.6f} (offset {off:+.4f}, A_target {At:.6f}); Newton residual {float(res):.2e} in {k} it [{time.time()-t0:.0f}s]; feasible {F.feasible(A, u, w, we, wc)}"
    )
    if res < arb("1e-30") and F.feasible(A, u, w, we, wc) and W.signal(A, u, w, we, wc, F.last_y) < 0:
        ok = True
        break
if not ok:
    WK.say(f"RESULT row state K {K}: seed failed for all {len(OFFS)} offsets")
    sys.exit(3)
sig = W.signal(A, u, w, we, wc, F.last_y)
W.save(K, A, u, w, we, wc, res, "OPTIMAL", sig)
WK.say(
    f"RESULT row state K {K}: state saved in {out} at A = {A0:.6f} (A_target {At:.6f}) [{time.time()-t0:.0f}s]"
)
