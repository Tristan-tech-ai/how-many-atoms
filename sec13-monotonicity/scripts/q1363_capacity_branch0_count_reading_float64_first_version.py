"""q1363 (8 Oct 2026, EXPLORATORY reading, no item, no verdict; run from temporal_within/centre_scripts/ as py q1363_capacity_branch0_count_reading.py): the branch-0 signature (T12 of the 6 Oct list, q884, which covered only the immune rows up to K 39) on the certified CAPACITY states of the amplitude-constrained Gaussian channel, K = 40 and K = 60 to 83 (A up to 52).
Branch k = 0 means that D(x) - C, the KKT function of the capacity (D(x) = int phi(y - x) log(phi(y - x) / Q(y)) dy, Q the output density of the stored input), has no critical point on (0, A) other than the atoms (maxima, D = C) and one minimum between consecutive atoms (and between the last interior atom and the edge atom at A). Expected number of strict sign changes of D'(x) on the open interval (0, A): K - 2 for every K (J interior atoms give J maxima and J minima for an even K; with a centre atom the minimum between the centre atom and the first atom adds one and the count is again K - 2). Stated BEFORE the run: expected K - 2 at every state; any other count is a finding that has to be followed up (a higher branch, i.e. a chirped comb, is the wall of amendment 656).
Data: for each K one stored state with verdict OPTIMAL (residual below 1e-20) chosen by the rule: the middle one, by sorted A, among the OPTIMAL states of that K (files results_q2*_gauss_K*_A*.json found below temporal_within). D' is computed in float64 on the grid x = 0.005 + 0.01 i, y-grid step 0.02 as in the solver (q206_gauss_mp), and again with the y-grid shifted by 0.01 (quadrature and rounding noise: the largest difference of D' is printed per state; grid points with |D'| below ten times it are undecided; a state with undecided points other than within 0.03 of an atom is not counted as agreeing).
Controls (first): c1 the counter finds 2 sign changes of a synthetic derivative with two known zeros and ignores a zero-touching point; c2 float64 against the 40-digit mpmath of q206_gauss_mp on the K = 40 state: the same sign-change positions within 0.01 and D' within 1e-9 relative to its maximum on a subgrid; c3 negative control that must change the count: the K = 40 state with the interior atoms moved to an equally spaced comb gives a count different from 38; c4 negative control: one interior atom's weight set to zero (the atom removed) gives a count different from 38.
Usage: py q1363_capacity_branch0_count_reading.py  (from temporal_within/centre_scripts/)"""

import glob, json, os, re, sys
import numpy as np


def die(msg):
    print(msg, flush=True)
    os._exit(2)


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def fl(s):
    return float(re.search(r"[-+0-9.eE]+", str(s).replace("np.float64(", "")).group(0))


def load():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
        try:
            j = json.load(open(f))
            K = int(j["K"])
            res = fl(j["residual"])
        except Exception:
            continue
        if j.get("verdict") == "OPTIMAL" and res < 1e-20:
            out.setdefault(K, []).append((fl(j["A"]), f, j))
    return out


def state(j):
    u = np.array([float(x) for x in j["u"]])
    w = np.array([float(x) for x in j["w"]])
    we = float(j["we"])
    wc = None if j.get("wc") in (None, "None") else float(j["wc"])
    return fl(j["A"]), u, w, we, wc


def dprime(A, u, w, we, wc, shift=0.0, H=0.02, step=0.01, want_D=False, at=None):
    y = -A - 9 + shift + H * np.arange(int((2 * A + 18) / H) + 1)
    lp = lambda c: -((y - c) ** 2) / 2 - 0.5 * np.log(2 * np.pi)
    Q = np.zeros_like(y)
    for uj, wj in zip(u, w):
        Q += wj * (np.exp(lp(uj)) + np.exp(lp(-uj)))
    Q += we * (np.exp(lp(A)) + np.exp(lp(-A)))
    if wc is not None:
        Q += wc * np.exp(lp(0.0))
    lQ = np.log(Q)
    xs = 0.005 + step * np.arange(int(A / step)) if at is None else np.array(at, dtype=float)
    d1 = np.empty_like(xs)
    d0 = np.empty_like(xs)
    for i0 in range(0, len(xs), 400):
        xc = xs[i0 : i0 + 400, None]
        L = -((y[None, :] - xc) ** 2) / 2 - 0.5 * np.log(2 * np.pi)
        P = np.exp(L)
        d1[i0 : i0 + 400] = H * np.sum(P * (y[None, :] - xc) * (L - lQ[None, :] + 1), axis=1)
        if want_D:
            d0[i0 : i0 + 400] = H * np.sum(P * (L - lQ[None, :]), axis=1)
    return (xs, d1, d0) if want_D else (xs, d1)


def count(xs, d, noise, atoms):
    """strict sign changes of d over the grid; a run of points with |d| < 10 noise is bridged: opposite decided signs on its two sides count one change, equal signs count as ambiguous (und) unless the run lies within 0.03 of an atom; a leading run is ignored"""
    ch = []
    und = 0
    prev = None
    run = []
    for x, v in zip(xs, d):
        if abs(v) < 10 * noise:
            run.append(x)
            continue
        sg = 1 if v > 0 else -1
        if prev is not None:
            if sg != prev:
                ch.append(run[len(run) // 2] if run else x)
            elif run and min((abs(r - a) for r in run for a in atoms), default=1e9) > 0.03:
                und += 1
        prev = sg
        run = []
    return ch, und


def analyse(A, u, w, we, wc):
    xs, d1 = dprime(A, u, w, we, wc)
    _, d2 = dprime(A, u, w, we, wc, shift=0.01)
    noise = float(np.max(np.abs(d1 - d2)))
    atoms = [x for x, ww in zip(u, w) if ww > 0] + [A] + ([0.0] if wc is not None else [])
    ch, und = count(xs, d1, noise, atoms)
    return ch, und, noise, float(np.max(np.abs(d1)))


# controls
t = np.linspace(0, 1, 2001)
syn = (t - 0.3) * (t - 0.7) * t**2
ch, und = count(t, syn, 1e-12, [0.0])
c1 = len(ch) == 2 and und == 0
print(f"control c1 (counter on a synthetic derivative): {c1}", flush=True)
if not c1:
    die("READING q1363: VOID (control c1)")
ST = load()
k40 = sorted(ST[40])[0] if 40 in ST else None
if k40 is None:
    die("READING q1363: VOID (no K = 40 state)")
A40, u40, w40, we40, wc40 = state(k40[2])
ch40, und40, noise40, mx40 = analyse(A40, u40, w40, we40, wc40)
sys.path.insert(0, os.path.join(ROOT))
import q206_gauss_mp as G
from mpmath import mp, mpf

mp.dps = 40
G.mp.dps = 40
G.L2P = mp.log(2 * mp.pi) / 2
A_ = mpf(k40[2]["A"])
uu = [mpf(x) for x in k40[2]["u"]]
ww = [mpf(x) for x in k40[2]["w"]]
yg = G.grid(A_)
rows, wp, wm, cen, Qm, lQm = G.tables(A_, yg, uu, ww, mpf(k40[2]["we"]), None)
xsub = [3.0, 7.3, 11.1, 16.4, 21.9, 27.5]
dm = []
for x in xsub:
    Lr, Pr, Sr = G.phi_row(yg, mpf(x))
    dm.append(float(G.Dp_of(Lr, Pr, Sr, lQm)))
df = [float(v) for v in dprime(A40, u40, w40, we40, wc40, at=xsub)[1]]
c2 = len(ch40) == 38 and max(abs(a - b) for a, b in zip(dm, df)) <= 1e-9 * mx40 + 1e-12
print(
    f"control c2 (K = 40 float64 against 40-digit: sign changes {len(ch40)}, noise {noise40:.2e}, largest D' {mx40:.2e}, largest difference at six points {max(abs(a - b) for a, b in zip(dm, df)):.2e}): {c2}"
)
uq = np.linspace(0.5, A40 - 1.0, len(u40))
chq, undq, _, _ = analyse(A40, uq, w40, we40, wc40)
c3 = len(chq) != 38 or undq > 0
w0 = w40.copy()
w0[len(w0) // 2] = 0.0
chr_, undr, _, _ = analyse(A40, u40, w0, we40, wc40)
c4 = len(chr_) != 38 or undr > 0
print(
    f"control c3 (equally spaced comb: {len(chq)} sign changes, {undq} undecided): {c3}; c4 (one atom removed: {len(chr_)}, {undr} undecided): {c4}"
)
if not (c2 and c3 and c4):
    die("READING q1363: VOID (control)")
print("state K, A, file | sign changes (expected K - 2), undecided away from atoms, noise, largest |D'|")
allok = True
n = 0
for K in [40] + sorted(k for k in ST if 60 <= k <= 83):
    L = sorted(ST[K], key=lambda t: t[0])
    A, f, j = L[len(L) // 2]
    A, u, w, we, wc = state(j)
    ch, und, noise, mx = analyse(A, u, w, we, wc)
    ok = len(ch) == K - 2 and und == 0
    allok = allok and ok
    n += 1
    print(
        f"  K {K}: A {A:.4f} ({len(L)} OPTIMAL states, file {os.path.basename(f)}) | {len(ch)} (expected {K - 2}), undecided {und}, noise {noise:.1e}, max |D'| {mx:.2e} -> {'agrees' if ok else 'DIFFERS'}"
    )
print(f"states analysed {n}; all agree with branch 0: {allok}")
print("READING q1363 (exploratory, no verdict)")
