"""q1365 (8 Oct 2026, EXPLORATORY reading, no item, no verdict; run from temporal_within/centre_scripts/ as py q1365_weighted_branch0_count_reading.py IDX, IDX = 0 (immune q584), 1 (immune q584_ext), 2 (NPMLE q601), 3 (NPMLE q601_ext)): the one-minimum lemma of TIER_A_NOTE section 3 on the weighted problems with uniform data on [-a, a].
Expected, written BEFORE the run. For these problems the KKT function is g(y) = (phi * (1_[-a,a] P^-k))(y) up to a constant, k = 1 (NPMLE) and k = 2 (immune), P = phi * (input) the output density. Its derivative g' = phi * (f') with f = 1_[-a,a] P^-k; f' is the distribution -k P^(-k-1) P' on (-a, a) plus a positive delta at -a and a negative delta at a, so S(g') <= 2 + S(P') <= 2 + (2K - 1) = 2K + 1 for K mass points (P has at most K modes). From below g = C at the K atoms (maxima, g <= C on the whole line) and g < C in the K - 1 gaps: 2K - 1 sign changes. g is even, so g' is odd and tends to 0 from below at infinity (g decreases to 0); extra sign changes would come in mirror pairs with even parity on each half line, and the slack is 2 in all, so none. Expected: exactly K - 1 sign changes of g' on (0, infinity), i.e. on the grid (0, a + 3]; in the order of y: with no centre atom the changes are the atom p_1 (+ to -), a minimum, p_2, ..., p_J (the odd-numbered changes are atoms, within 0.05); with a centre atom the first change is a minimum, then p_1, ... (the even-numbered changes are atoms). q884 (6 Oct) checked 8 immune rows to K = 39 with the same expectation (2J - 1 or 2J).
Rows: for each of the four stored files every 60th row and the last row (immune q584: K 30-39; q584_ext: K 30-45, a to 33; NPMLE q601: K 24-34; q601_ext: K 24-37, a to 28.9). Each row is polished to 200 bits by the arb code of q579 (the stored floats are the seed; the polished residual is printed), g' is evaluated by arb on the grid y = 0.01, 0.02, ..., a + 3 with the weights of the problem (immune: Q / (2 P^2) in the sum, NPMLE: Q / (2 P), the code of q601), a point is decided only if its ball excludes 0, a run of undecided points between opposite decided signs counts one change, a run between equal signs counts as ambiguous unless every point of the run is within 0.03 of an atom. Resolution limit, stated in advance: two extra critical points closer than 0.02 would be missed.
Controls (first, on the first selected row): c1 counter on a synthetic sequence; c2 the polished state has max |g'(p_j)| / C below 1e-30 at the atoms and positions within 1e-9 of the stored floats; c3 negative control: the same state with the atoms moved to an equally spaced comb gives another count or pattern; c4 negative control: one pair mass set to zero gives another count.
Usage: py q1365_weighted_branch0_count_reading.py IDX"""

import json, os, sys
import numpy as np

IDX = int(sys.argv[1])
sys.argv = [sys.argv[0]]
from flint import arb, ctx

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import q579_immune_mp as M


def die(msg):
    print(msg, flush=True)
    os._exit(2)


GS_IMM = M.IMP.gs


def gs_npmle(self, p, v, c, ys):  # text copy of _gs_npmle of q601_npmle_mpcont.py
    P = self.P(p, v, c)
    Cn = sum(self.Q)
    W = [q / (2 * Pi) for q, Pi in zip(self.Q, P)]
    out = []
    for y in ys:
        g0 = g1 = g2 = arb(0)
        for x, Wi in zip(self.x, W):
            dm, dp = x - y, x + y
            em, ep = M.f(dm), M.f(dp)
            g0 += Wi * (em + ep)
            g1 += Wi * (dm * em - dp * ep)
            g2 += Wi * ((dm * dm - 1) * em + (dp * dp - 1) * ep)
        out.append((g0 / Cn, g1 / Cn, g2 / Cn))
    return out, Cn


FILES = [
    ("immune", "results_q584.json"),
    ("immune", "results_q584_ext.json"),
    ("npmle", "results_q601.json"),
    ("npmle", "results_q601_ext.json"),
]
kind, fn = FILES[IDX]
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
rows = json.load(open(os.path.join(ROOT, fn)))["rows"]
sel = sorted(set(list(range(0, len(rows), 60)) + [len(rows) - 1]))


def count(xs, balls, atoms):
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


def pattern_ok(ch, atoms, centre):
    J = len(atoms)
    if centre:
        if len(ch) != 2 * J:
            return False
        return all(abs(ch[2 * j + 1] - atoms[j]) <= 0.05 for j in range(J))
    if len(ch) != 2 * J - 1:
        return False
    return all(abs(ch[2 * j] - atoms[j]) <= 0.05 for j in range(J))


def analyse(r, p_override=None, v_override=None):
    M.IMP.gs = GS_IMM if kind == "immune" else gs_npmle
    ctx.prec = 200
    a = float(r["a"])
    centre = bool(r["centre"])
    I = M.IMP(arb(a), 9)
    p0 = [float(t) for t in (p_override if p_override is not None else r["p"])]
    v0 = [float(t) for t in (v_override if v_override is not None else r["v"])]
    if p_override is None and v_override is None:
        pp, vv, res = I.polish(p0, v0, centre)
    else:
        pp = [arb(t) for t in p0]
        vv = [arb(t) for t in v0]
        res = None
    c = (arb(1) - sum(vv)) if centre else None
    ys = np.arange(0.01, a + 3, 0.01)
    gp, C = I.gs(pp, vv, c, [arb(float(y)) for y in ys])
    d1 = [t[1] for t in gp]
    atoms = [float(t.mid()) for t in pp]
    ch, und = count(ys, d1, atoms + ([0.0] if centre else []))
    st = I.gs(pp, vv, c, pp)[0]
    stat = max(abs(float(t[1].mid())) for t in st)
    mn = min([abs(float(b.mid())) for b in d1 if not b.contains(0)] or [float("nan")])
    return dict(
        ch=ch,
        und=und,
        atoms=atoms,
        centre=centre,
        K=2 * len(pp) + int(centre),
        a=a,
        res=(float(res) if res is not None else None),
        stat=stat,
        mn=mn,
        pos_diff=max(abs(float(x.mid()) - y) for x, y in zip(pp, r["p"])) if res is not None else None,
        rad=max(float(b.rad()) for b in d1),
    )


t = np.linspace(0, 1, 2001)
syn = (t - 0.3) * (t - 0.7) * t**2


class FB:
    def __init__(self, v):
        self.v = v

    def contains(self, k):
        return abs(self.v) < 1e-12

    def __gt__(self, k):
        return self.v > 0


ch, und = count(t, [FB(v) for v in syn], [0.0])
c1 = len(ch) == 2 and und == 0
print(f"control c1 (counter on a synthetic sequence): {c1}", flush=True)
if not c1:
    die("READING q1365: VOID (control c1)")
r0 = rows[sel[0]]
R0 = analyse(r0)
J0 = len(R0["atoms"])
K0 = R0["K"]
c2 = R0["stat"] < 1e-30 and R0["pos_diff"] < 1e-9 and len(R0["ch"]) == K0 - 1 and R0["und"] == 0
print(
    f"control c2 (first row {kind} K {K0}, a {R0['a']}: polished residual {R0['res']:.1e}, max |g'(p_j)| {R0['stat']:.1e}, position difference to the stored floats {R0['pos_diff']:.1e}, {len(R0['ch'])} sign changes, expected {K0 - 1}): {c2}",
    flush=True,
)
uq = list(np.linspace(0.3, R0["a"] - 0.5, J0))
Rq = analyse(r0, p_override=uq, v_override=[float(x) for x in r0["v"]])
c3 = len(Rq["ch"]) != K0 - 1 or Rq["und"] > 0 or not pattern_ok(Rq["ch"], sorted(Rq["atoms"]), Rq["centre"])
v0 = [float(x) for x in r0["v"]]
v0[len(v0) // 2] = 0.0
Rr = analyse(r0, p_override=r0["p"], v_override=v0)
c4 = len(Rr["ch"]) != K0 - 1 or Rr["und"] > 0
print(
    f"control c3 (equally spaced comb: {len(Rq['ch'])} sign changes, {Rq['und']} undecided): {c3}; c4 (one pair mass set to zero: {len(Rr['ch'])}, {Rr['und']} undecided): {c4}",
    flush=True,
)
if not (c2 and c3 and c4):
    die("READING q1365: VOID (control)")
print(
    f"{kind} rows of {fn}: {len(sel)} selected | K, a | sign changes (expected K - 1), pattern, undecided, polished residual, max |g'(p_j)|, smallest decided |g'|, radius",
    flush=True,
)
allok = True
for i in sel:
    R = analyse(rows[i])
    K = R["K"]
    pat = pattern_ok(R["ch"], sorted(R["atoms"]), R["centre"])
    ok = len(R["ch"]) == K - 1 and R["und"] == 0 and pat
    allok = allok and ok
    print(
        f"  row {i}: K {K}, a {R['a']:.4f}, centre {R['centre']} | {len(R['ch'])} (expected {K - 1}), pattern {pat}, undecided {R['und']}, residual {R['res']:.1e}, stat {R['stat']:.1e}, smallest decided {R['mn']:.1e}, radius {R['rad']:.1e} -> {'agrees' if ok else 'DIFFERS'}",
        flush=True,
    )
print(f"rows analysed {len(sel)}; all agree with the lemma: {allok}")
print("READING q1365 (exploratory, no verdict)")
