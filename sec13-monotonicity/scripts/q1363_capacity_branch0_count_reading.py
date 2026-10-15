"""q1363 (8 Oct 2026, EXPLORATORY reading, no item, no verdict; version 2 in ball arithmetic; run from temporal_within/centre_scripts/ as py q1363_capacity_branch0_count_reading.py [K ...]): the branch-0 signature (T12 of the 6 Oct list, q884, which covered only the immune rows up to K 39) on the certified CAPACITY states of the amplitude-constrained Gaussian channel, K = 40 and K = 60 to 83 (A up to 52.5).
Version 1 (float64, file q1363_capacity_branch0_count_reading_float64_first_version.py, output q1363_out_float64_first_version.txt) agreed at K = 40, 60, 61 and 63 and then saturated near 59 sign changes for K >= 62: the derivative of the KKT function between atoms falls below the double-precision noise (1e-15) in the deep bulk, and its ambiguity test was wrong (a run of undecided points counted as near an atom if any one point was). Those rows are not a finding about the states. Version 2 computes D'(x) in 220-bit ball arithmetic and treats a point as decided only if its ball excludes 0.
Branch k = 0 means that D(x) - C (D(x) = int phi(y - x) log(phi(y - x) / Q(y)) dy, Q the output density of the stored input, C the capacity) has no critical point on (0, A) other than the atoms (maxima) and one minimum between consecutive atoms (and between the last interior atom and the edge atom at A). Stated BEFORE the run: the number of strict sign changes of D' on the open interval (0, A) is K - 2 at every state, the pattern is alternating (+ to - at the atom j, - to + between atoms j and j + 1 and between the last interior atom and A), and every +- change lies within 0.05 of the corresponding atom. Any other count or pattern is a finding that has to be followed up (a higher branch, i.e. a chirped comb, is the wall of amendment 656).
Method. x on the solver grid y_j = -A - 9 + H j with H the double 0.02 (q206_gauss_mp creates H = mpf('0.02') at 53 bits before it sets its precision, so its grid spacing is 0.0200000000000000004163...; the states were solved on that grid and the ball code uses the same H); D'(x_i) = H sum_j P(j - i)(m H)(1 - c - (m H)^2 / 2 - log Q_j), m = j - i, the sum over the grid points only (the discrete sum of the solver), as the difference of two correlations done by arb_poly products (kernel truncated at |m H| <= 14, below 1e-42). Resolution limit, stated in advance: two extra critical points closer than 0.04 would be missed. A state is analysed only if its stored residual is below 1e-20 and verdict OPTIMAL; for each K the middle one by sorted A.
Controls (first): c1 the counter on a synthetic derivative with two known zeros, a double zero at the start and an exact zero on a grid point; c2 the ball-arithmetic D' against the plain 40-digit mpmath sum of q206_gauss_mp (Dp_of) at seven grid points of the K = 40 state: difference below 1e-30 in absolute value; c3 negative control: the K = 40 state with the interior atoms moved to an equally spaced comb gives a count or a pattern different from the expected; c4 negative control: one interior atom's weight set to zero gives a count different from 38; c5 the K = 40 state itself gives 38 changes, an alternating pattern and no undecided point.
Usage: py q1363_capacity_branch0_count_reading.py [K ...]   (from temporal_within/centre_scripts/; without arguments K = 40 and 60 to 83)
"""

import glob, json, os, re, sys
import numpy as np
from flint import arb, arb_poly, ctx


def die(msg):
    print(msg, flush=True)
    os._exit(2)


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PREC = 220
RK = 14.0


def numstr(s):
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s))
    return m.group(0)


def load():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
        try:
            j = json.load(open(f))
            K = int(j["K"])
            res = float(numstr(j["residual"]))
            A = numstr(j["A"])
        except Exception:
            continue
        if j.get("verdict") == "OPTIMAL" and res < 1e-20:
            out.setdefault(K, []).append((float(A), f, j))
    return out


def state(j):
    wc = j.get("wc")
    return (
        numstr(j["A"]),
        [numstr(x) for x in j["u"]],
        [numstr(x) for x in j["w"]],
        numstr(j["we"]),
        (None if wc in (None, "None") else numstr(wc)),
    )


def dprime_ball(A_s, u_s, w_s, we_s, wc_s, w_override=None):
    """grid points inside (0, A) and the balls D'(x)"""
    ctx.prec = PREC
    A = arb(A_s)
    Hh = arb(0.02)
    n = int(float((2 * A + 18) / Hh)) + 1
    c = (2 * arb.pi()).log() / 2
    ys = [-A - 9 + Hh * j for j in range(n)]
    phi = lambda t: (-(t * t) / 2 - c).exp()
    Q = [arb(0)] * n
    wts = [arb(x) for x in (w_s if w_override is None else w_override)]
    for us, wk in zip(u_s, wts):
        u = arb(us)
        if wk == 0:
            continue
        for j in range(n):
            Q[j] += wk * (phi(ys[j] - u) + phi(ys[j] + u))
    we = arb(we_s)
    for j in range(n):
        Q[j] += we * (phi(ys[j] - A) + phi(ys[j] + A))
    if wc_s is not None:
        wc = arb(wc_s)
        for j in range(n):
            Q[j] += wc * phi(ys[j])
    lQ = [q.log() for q in Q]
    M = int(RK / 0.02)
    Km = [phi(Hh * (M - t)) * (Hh * (M - t)) for t in range(2 * M + 1)]  # coefficient t <-> m = M - t
    K1 = [phi(Hh * (M - t)) * (Hh * (M - t)) * (1 - c - (Hh * (M - t)) ** 2 / 2) for t in range(2 * M + 1)]
    S2 = (arb_poly(lQ) * arb_poly(Km)).coeffs()
    S1 = (arb_poly([arb(1)] * n) * arb_poly(K1)).coeffs()
    xs = []
    ds = []
    for i in range(n):
        x = float(ys[i])
        if not (0.0 < x < float(A)):
            continue
        ds.append(Hh * (S1[i + M] - S2[i + M]))
        xs.append(x)
    return np.array(xs), ds


def count(xs, balls, atoms):
    """strict sign changes of D' over the grid; points whose ball contains 0 form runs: opposite decided signs on both sides count one change at the middle of the run, equal signs are ambiguous unless every point of the run is within 0.03 of an atom; a leading run is ignored"""
    ch = []
    und = 0
    prev = None
    run = []
    for x, b in zip(xs, balls):
        if b.contains(0):
            run.append(x)
            continue
        sg = 1 if b > 0 else -1
        if prev is not None:
            if sg != prev:
                ch.append(run[len(run) // 2] if run else x)
            elif run and max(min(abs(r - a) for a in atoms) for r in run) > 0.03:
                und += 1
        prev = sg
        run = []
    return ch, und


def pattern_ok(ch, atoms_int):
    """the sign changes are, in order, atom 1 (+ to -), a minimum, atom 2, ..., atom J, a minimum; the odd ones within 0.05 of the atoms"""
    J = len(atoms_int)
    if len(ch) != 2 * J:
        return False
    return all(abs(ch[2 * j] - atoms_int[j]) <= 0.05 for j in range(J)) and all(
        atoms_int[j] < ch[2 * j + 1] < (atoms_int[j + 1] if j + 1 < J else 1e9) for j in range(J)
    )


def analyse(A_s, u_s, w_s, we_s, wc_s, w_override=None):
    xs, ds = dprime_ball(A_s, u_s, w_s, we_s, wc_s, w_override)
    wts = w_s if w_override is None else w_override
    atoms_int = sorted(float(a) for a, wk in zip(u_s, wts) if float(wk) > 0)
    atoms = atoms_int + [float(A_s)] + ([0.0] if wc_s is not None else [])
    ch, und = count(xs, ds, atoms)
    mx = max(abs(float(b.mid())) for b in ds)
    dec = [abs(float(b.mid())) for b in ds if not b.contains(0)]
    mn = min(dec) if dec else float("nan")
    rad = max(float(b.rad()) for b in ds)
    return ch, und, mx, mn, rad, (pattern_ok(ch, atoms_int) if wc_s is None else None)


class FakeBall:  # stand-in for the counter control
    def __init__(self, v):
        self.v = v

    def contains(self, k):
        return abs(self.v) < 1e-12

    def __gt__(self, k):
        return self.v > 0


t = np.linspace(0, 1, 2001)
syn = (t - 0.3) * (t - 0.7) * t**2
ch, und = count(t, [FakeBall(v) for v in syn], [0.0])
c1 = len(ch) == 2 and und == 0
print(f"control c1 (counter on a synthetic derivative): {c1}", flush=True)
if not c1:
    die("READING q1363: VOID (control c1)")
ST = load()
if 40 not in ST:
    die("READING q1363: VOID (no K = 40 state)")
L40 = sorted(ST[40], key=lambda x: x[0])
A40s, u40, w40, we40, wc40 = state(L40[0][2])
ch40, und40, mx40, mn40, rad40, pat40 = analyse(A40s, u40, w40, we40, wc40)
sys.path.insert(0, ROOT)
import q206_gauss_mp as G
from mpmath import mp, mpf

mp.dps = 40
G.mp.dps = 40
G.L2P = mp.log(2 * mp.pi) / 2
A_ = mpf(A40s)
uu = [mpf(x) for x in u40]
ww = [mpf(x) for x in w40]
yg = G.grid(A_)
rows, wp, wm, cen, Qm, lQm = G.tables(A_, yg, uu, ww, mpf(we40), None)
xs40, ds40 = dprime_ball(A40s, u40, w40, we40, wc40)
pick = [20, 100, 400, 700, 1000, 1200, 1400]
dif = 0.0
for i in pick:
    idx = int(round((float(xs40[i]) - float(yg[0])) / 0.02))
    Lr, Pr, Sr = G.phi_row(yg, yg[idx])
    dv = G.Dp_of(Lr, Pr, Sr, lQm)
    dif = max(dif, float(abs(ds40[i] - arb(mp.nstr(dv, 45))).upper()))
c2 = dif <= 1e-30
print(
    f"control c2 (ball arithmetic against the 40-digit mpmath sum at seven grid points, full precision: largest upper bound of the difference {dif:.2e}, max |D'| {mx40:.3e}; bar 1e-30): {c2}",
    flush=True,
)
uq = [str(x) for x in np.linspace(0.5, float(A40s) - 1.0, len(u40))]
chq, undq, _, _, _, patq = analyse(A40s, uq, w40, we40, wc40)
c3 = len(chq) != 38 or undq > 0 or not patq
w0 = list(w40)
w0[len(w0) // 2] = "0"
chr_, undr, _, _, _, patr = analyse(A40s, u40, w40, we40, wc40, w_override=w0)
c4 = len(chr_) != 38 or undr > 0
c5 = len(ch40) == 38 and und40 == 0 and bool(pat40)
print(
    f"control c3 (equally spaced comb: {len(chq)} sign changes, {undq} undecided, pattern {patq}): {c3}; c4 (one atom removed: {len(chr_)} sign changes, {undr} undecided): {c4}; c5 (K = 40 state: {len(ch40)} changes, {und40} undecided, pattern {pat40}, smallest decided |D'| {mn40:.2e}, largest ball radius {rad40:.1e}): {c5}",
    flush=True,
)
if not (c2 and c3 and c4 and c5):
    die("READING q1363: VOID (control)")
Ks = [int(a) for a in sys.argv[1:]] or [40] + sorted(k for k in ST if 60 <= k <= 83)
print(
    "state K, A, file | sign changes (expected K - 2), pattern, undecided, smallest decided |D'|, largest radius",
    flush=True,
)
allok = True
n = 0
for K in Ks:
    L = sorted(ST[K], key=lambda x: x[0])
    A, f, j = L[len(L) // 2]
    A_s, u_s, w_s, we_s, wc_s = state(j)
    ch, und, mx, mn, rad, pat = analyse(A_s, u_s, w_s, we_s, wc_s)
    ok = len(ch) == K - 2 and und == 0 and (pat is None or pat)
    allok = allok and ok
    n += 1
    print(
        f"  K {K}: A {A:.4f} ({len(L)} OPTIMAL states, {os.path.basename(f)}) | {len(ch)} (expected {K - 2}), pattern {pat}, undecided {und}, smallest decided |D'| {mn:.1e}, radius {rad:.1e} -> {'agrees' if ok else 'DIFFERS'}",
        flush=True,
    )
print(f"states analysed {n}; all agree with branch 0: {allok}")
print("READING q1363 (exploratory, no verdict)")
