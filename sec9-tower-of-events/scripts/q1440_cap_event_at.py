"""q1440 (9 Oct 2026, INSTRUMENT / COMPUTATION, no verdict): locate ONE capacity transition K_b by itself (many in parallel on separate cores), the capacity counterpart of q1437_imm_event_at.py. The state is seeded on the capacity half-line stage D = 134 of the chain with B1 = 20 (centre_scripts/c50_capc_D140_B20.json: depths d and masses m of the atoms, the atom 1 being the wall atom) at A0 = a_pred(K) - BACK, where a_pred is the stage-134 prediction of the registered table q1439_cap_b20_table.json. Interior atoms j = 2 .. K//2 (j counted from the edge) sit at x with N(A - x) - dM(x/v) = j - 1/2 (N the counting function of the stage, dM the centre correction of the bulk comb, v = 4 pi rho), masses m_j, wall mass m_1, centre mass 1/rho for odd K, all divided by the total mass. The seed is converged by the Newton of q1414 (192 bits, band-limited, H = 0.05, tol 1e-30, 25 steps, then up to 60 more when it stalls between 1e-30 and 1e-2), then walked up with the walker of q1411 to a_pred + FWD and the event is located with Walker.locate. The seed only chooses the starting point. Output OUTDIR/event_Kb<K>.json in the format of q1411.
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
back = float(sys.argv[3]) if len(sys.argv) > 3 else 0.003
fwd = float(sys.argv[4]) if len(sys.argv) > 4 else 0.012
CHAIN = os.environ.get("Q1440_CHAIN") == "1"
KS = (
    K - 1 if CHAIN else K
)  # chain mode: the seed is that of K_b - 1 (an odd chain starts easily), and the walker passes through its split into the family K and on to the event K_b
TAB = json.load(open(os.path.join(HERE, "..", "centre_scripts", "q1439_cap_b20_table.json")))["stages"]["134"]
TAB.update(
    json.load(open(os.path.join(HERE, "..", "centre_scripts", "q1451_cap_b20_table.json")))["stages"]["134"]
)
TAB.update(
    json.load(open(os.path.join(HERE, "..", "centre_scripts", "q1527_cap_b20_table.json")))["stages"]["134"]
)
_LT = os.path.join(HERE, "..", "centre_scripts", "q1694_cap_law_seed_table.json")
if os.path.exists(_LT):
    _T2 = json.load(open(_LT))["stages"]["134"]
    [
        TAB.setdefault(k, v) for k, v in _T2.items()
    ]  # 10 Oct 10:5x: seeds for K_b 139..150 from the registered law; keys already in the older tables are not changed
pred = TAB[str(K)]
predS = TAB[
    str(KS)
]  # 9 Oct 16:3x: the table q1451 (K_b 115 to 130, registered in amendment 2179) is added for the seeds; q1439 values for K_b <= 114 are unchanged
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


W = WK.Walker(192, out, H=0.05)
F = W.F
t0 = time.time()
ok = False
BACKS = [back] + [0.0030 + 0.0003 * i for i in range(-5, 11) if abs(0.0030 + 0.0003 * i - back) > 1e-9]
if os.environ.get("Q1437_BACKS"):
    BACKS = [float(t) for t in os.environ["Q1437_BACKS"].split(",")]
for bk in BACKS:
    A0 = predS - bk
    u0, w0, we0, wc0 = seed(A0)
    A = arb(A0)
    u = [arb(x) for x in u0]
    w = [arb(x) for x in w0]
    we = arb(we0)
    wc = None if wc0 is None else arb(wc0)
    try:
        u, w, we, wc, res, k = F.newton(A, u, w, we, wc, tol="1e-30", steps=25)
        if arb("1e-30") < res < arb("1e-2"):
            u, w, we, wc, res, k2 = F.newton(
                A, u, w, we, wc, tol="1e-30", steps=int(os.environ.get("Q1440_SEED_STEPS", "60"))
            )
            k += k2
    except Exception as e:
        WK.say(f"K = {K}: seed at A0 = {A0:.6f} (back {bk:.4f}); exception {e!r}")
        continue
    WK.say(
        f"K = {K}: seed at A0 = {A0:.6f} (back {bk:.4f}, a_pred {predS:.9f}, seed K {KS}); Newton residual {float(res):.2e} in {k} it [{time.time()-t0:.0f}s]; feasible {F.feasible(A, u, w, we, wc)}"
    )
    if res < arb("1e-30") and F.feasible(A, u, w, we, wc) and W.signal(A, u, w, we, wc, F.last_y) < 0:
        ok = True
        break
if not ok:
    WK.say(f"RESULT K_b {K}: seed failed for all {len(BACKS)} starting amplitudes")
    sys.exit(3)
if CHAIN:
    o = W.run(A, (u, w, we, wc), pred + fwd, hmax=0.02)
    WK.say(
        f"RESULT chain K_b {K}: run returned {o[0]} [{time.time()-t0:.0f}s]; event file exists: {os.path.exists(os.path.join(out, f'event_Kb{K}.json'))}"
    )
    sys.exit(0)
o = W.walk_family(A, u, w, we, wc, pred + fwd, h=2e-4, hmax=0.01)
if o[0] != "event":
    WK.say(f"RESULT K_b {K}: no event before {pred + fwd:.6f} ({o[0]})")
    sys.exit(4)
_, A_lo, st_lo, s_lo, A_hi, st_hi, s_hi = o
A_lo2, st_lo2, s_lo2, A_hi2, st_hi2, s_hi2, A_ev = W.locate((A_lo, st_lo, s_lo), (A_hi, st_hi, s_hi))
kind = "split" if st_lo2[3] is not None else "birth"
WK.say(
    f"EVENT located: K_b = {K} ({kind}) A_event = {A_ev.str(20, radius=False)} (bracket {float(A_hi2 - A_lo2):.2e})"
)
os.makedirs(out, exist_ok=True)
json.dump(
    {
        "K_b": K,
        "kind": kind,
        "A_event": A_ev.str(20, radius=False),
        "A_lo": WK.S(A_lo2, 30),
        "A_hi": WK.S(A_hi2, 30),
        "s_lo": float(s_lo2.mid()),
        "s_hi": float(s_hi2.mid()),
        "seed_A0": A0,
        "a_pred": pred,
    },
    open(os.path.join(out, f"event_Kb{K}.json"), "w"),
)
WK.say(
    f"RESULT K_b {K}: A_event {A_ev.str(20, radius=False)}; deviation from a_pred {(float(A_ev.mid()) - pred) * 1e9:+.4f}e-9 [{time.time()-t0:.0f}s]"
)
