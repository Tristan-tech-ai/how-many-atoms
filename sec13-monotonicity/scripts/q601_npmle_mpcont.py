"""Q601 (30 Sep 21:05:36, date stamp): q584 v6 copied for the NPMLE functional (uniform data on [-a, a]): IMP.gs is replaced so that
W = Q/(2 Ptilde) and C = sum Q (KKT c(y) = int phi(x - y) q/Ptilde dx <= 1); every polish is arb-only (q580.run_polish is the immune
double solver and is never called); start from the q590 double state q508c_ckpt_npmle_q590.npz (a = 20.7806, K = 24); output
results_q601.json. Purpose: reach 823 (3)'s sixth transition (25 -> 26 predicted at 21.448937) and 845's position-model windows.
Q584 (source of the code below): 200-bit continuation of the uniform immune optimum with GENERAL events, written 30 Sep 18:22:00 (date stamp), for the event cluster
at a = 23.685-23.71 (816). Uses q580's run_polish (double predictor, then q579 arb corrector) and q579's IMP. At every row: (i) the
centre signal (s0 = g(0)/C - 1 without a centre atom, s2 = g''(0)/C with one), (ii) the off-support maximum of g/C - 1 on a 0.01 grid of
[0, a + 3] away from atoms, with its location, (iii) the largest g''/C over the pairs, with its index. A row is accepted when all three
are <= 0. At a sign change, the largest signal of the OLD family is bisected to 1e-6 in a, its type decides the event (centre appear,
centre split, pair birth at the located |y|, split of the located pair), and new-family candidates at a_t + 0.002 go through
run_polish; the first with all three signals <= 0 is taken. Steps start at 0.005 and double after three clean rows, up to 0.05.
The run records every event with its type and location: whether any event sits off the centre is itself a test of the birth rule.
Start: the certified post-split row a = 23.704778 (K = 30) of results_q580.json. Output: results_q601.json, log q584.log.
v3 (18:55): quadratic predictor through the last three rows (v2's secant limited steps to ~0.0012 at a - a_t ~ 0.03).
v2 (18:42): steps use an arb-only corrector from a secant predictor through the last two 200-bit rows (the double
predictor's soft-mode error, ~1e-10 / 1e-8, left it outside Newton's basin; v1 stopped at 23.708527); minimum step 2e-6.
RESUME: start from the last row of results_q601.json when it exists.
v4 (19:33): (a) the resume polish and every new-family candidate go arb-only first (run_polish's double pre-solve only as fallback);
(c) a centre split polishes the old family at a_hi + 0.02 and seeds the young pair at +-0.24 (pitchfork width
1.71 sqrt(a - a_t) from q580's post-split row), then 0.3, 0.2, 0.5, 0.7. (b) a centre appear is followed by q587's born_centre (fixed-c continuation in the centre mass, as q492) at a_hi + 0.01, which
certified K = 31 at 24.3272637 in q586 after v3's direct seeds (1e-3, 1e-2, 3e-2) and q585's arb-only direct seeds failed.
v5 (20:24): a centre split first tries q587.split_centre (young pair pinned at +-eps, regula falsi on g'(eps)/C), which
certified K = 32 at 24.9646401 in q596 after v4's five direct seeds failed.
Usage: py q584_general_cont.py [AMAX]"""

import sys, json, time
import numpy as np

AMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
sys.argv = [sys.argv[0]]
import os

DMAX = float(
    os.environ.get("DMAX", "0.05")
)  # v6 (20:52): optional step cap (0.021 steps kept failing past 25.1)
import q580_immune_mpcont as C
import q587_birthlib as B
from flint import arb

M = C.M
from flint import arb as _arb


def _gs_npmle(self, p, v, c, ys):
    P = self.P(p, v, c)
    Cn = sum(self.Q)
    W = [q / (2 * Pi) for q, Pi in zip(self.Q, P)]
    out = []
    for y in ys:
        g0 = g1 = g2 = _arb(0)
        for x, Wi in zip(self.x, W):
            dm, dp = x - y, x + y
            em, ep = M.f(dm), M.f(dp)
            g0 += Wi * (em + ep)
            g1 += Wi * (dm * em - dp * ep)
            g2 += Wi * ((dm * dm - 1) * em + (dp * dp - 1) * ep)
        out.append((g0 / Cn, g1 / Cn, g2 / Cn))
    return out, Cn


M.IMP.gs = _gs_npmle
say = lambda *a: print(*a, flush=True)
t0 = time.time()


def signals3(a, pp, vv, centre):
    I = M.IMP(a, 9)
    c = (arb(1) - sum(vv)) if centre else None
    g0c = I.gs(pp, vv, c, [arb(0)])[0][0]
    cs = float(((g0c[2]) if centre else (g0c[0] - 1)).mid())
    pf = [float(t.mid()) for t in pp]
    grid = [
        y
        for y in np.arange(0.01, a + 3, 0.01)
        if min(abs(y - q) for q in pf) > 0.05 and (not centre or y > 0.05)
    ]
    gg, _ = I.gs(pp, vv, c, [arb(y) for y in grid])
    vals = [float((t[0] - 1).mid()) for t in gg]
    j = int(np.argmax(vals))
    gp, _ = I.gs(pp, vv, c, pp)
    s2p = [float(t[2].mid()) for t in gp]
    k = int(np.argmax(s2p))
    return cs, vals[j], grid[j], s2p[k], k


def polish_sig(a, p, v, centre, arb_only=False):
    if arb_only:  # v2: arb-only corrector from a (secant) predictor
        I = M.IMP(a, 9)
        pp, vv, r = I.polish(list(p), list(v), centre)
        r = float(r)
        pf = [float(t.mid()) for t in pp]
        vf = [float(t.mid()) for t in vv]
        ok = r < 1e-35 and min(vf) > 0 and (not centre or 1 - sum(vf) > 0) and all(np.diff([0.0] + pf) > 0)
    else:
        return polish_sig(a, p, v, centre, arb_only=True)
    if not ok:
        return None
    cs, off, loc, sp, k = signals3(a, pp, vv, centre)
    return pf, vf, r, cs, off, loc, sp, k


if __name__ == "__main__":
    import os

    src = (
        "results_q601.json"
        if os.path.exists("results_q601.json") and json.load(open("results_q601.json"))["rows"]
        else "npz"
    )
    if src == "npz":
        z = np.load("q508c_ckpt_npmle_q590.npz")
        a = float(z["a"])
        p = list(map(float, z["p"]))
        v = list(map(float, z["v"]))
        centre = bool(z["centre"])
    else:
        last = json.load(open(src))["rows"][-1]
        a = float(last["a"])
        p = list(last["p"])
        v = list(last["v"])
        centre = bool(last["centre"])
    out = polish_sig(a, p, v, centre, arb_only=True) or polish_sig(a, p, v, centre)
    p, v = out[0], out[1]
    say(
        f"start a = {a}: K = {2*len(p) + int(centre)}, centre sig {out[3]:+.2e}, off {out[4]:+.2e} at {out[5]:.2f}, pair split {out[6]:+.2e}   [{time.time() - t0:.0f}s]"
    )
    rows = json.load(open("results_q601.json"))["rows"] if src == "results_q601.json" else []
    events = json.load(open("results_q601.json"))["events"] if src == "results_q601.json" else []
    da = float(os.environ.get("DA0", "0.0003"))
    clean = 0

    def save():
        json.dump({"rows": rows, "events": events}, open("results_q601.json", "w"), indent=1)

    while a < AMAX - 1e-9:
        an = round(min(a + da, AMAX), 7)
        same = lambda i, j: rows[i]["K"] == rows[j]["K"] and rows[i]["centre"] == rows[j]["centre"]
        if (
            len(rows) >= 3 and same(-1, -2) and same(-2, -3)
        ):  # v3: quadratic (Lagrange) predictor through the last three rows
            (a0, r0), (a1, r1), (a2, r2) = [(rows[i]["a"], rows[i]) for i in (-3, -2, -1)]
            L0 = (an - a1) * (an - a2) / ((a0 - a1) * (a0 - a2))
            L1 = (an - a0) * (an - a2) / ((a1 - a0) * (a1 - a2))
            L2 = (an - a0) * (an - a1) / ((a2 - a0) * (a2 - a1))
            pp_ = [L0 * x0 + L1 * x1 + L2 * x2 for x0, x1, x2 in zip(r0["p"], r1["p"], r2["p"])]
            vp_ = [L0 * x0 + L1 * x1 + L2 * x2 for x0, x1, x2 in zip(r0["v"], r1["v"], r2["v"])]
        elif len(rows) >= 2 and same(-1, -2):  # v2: secant predictor
            r1, r0 = rows[-1], rows[-2]
            lam = (an - r1["a"]) / (r1["a"] - r0["a"])
            pp_ = [x1 + lam * (x1 - x0) for x1, x0 in zip(r1["p"], r0["p"])]
            vp_ = [x1 + lam * (x1 - x0) for x1, x0 in zip(r1["v"], r0["v"])]
        else:
            pp_, vp_ = p, v
        res = polish_sig(an, pp_, vp_, centre, arb_only=True)
        if res is None:
            if da > 2e-6:
                da /= 2
                say(f"   polish failed at {an}; step halved to {da}")
                continue
            say(f"STOP at {an}: polish failed")
            break
        pf, vf, r, cs, off, loc, sp, k = res
        top = max(cs, off, sp)
        if top <= 0:
            a, p, v = an, pf, vf
            clean += 1
            K = 2 * len(p) + int(centre)
            rows.append(
                {
                    "a": a,
                    "K": K,
                    "centre": centre,
                    "p": p,
                    "v": v,
                    "cs": cs,
                    "off": off,
                    "off_at": loc,
                    "pairsplit": sp,
                    "res": r,
                }
            )
            save()
            say(
                f"a = {a}: K = {K}, centre {cs:+.1e}, off {off:+.1e}, pair split {sp:+.1e}, |F| {r:.0e}   [{time.time() - t0:.0f}s]"
            )
            if abs(a - round(a)) < 1e-9:
                say(f"   SCORE ROW a = {a}: K = {K}, offset {K - 0.4002*a**(4/3):.3f}")
            if clean >= 3 and da < DMAX:
                da = min(2 * da, DMAX)
                clean = 0
            if a + da > np.floor(a) + 1 + 1e-9 and np.floor(a) + 1 <= AMAX:
                da = round(np.floor(a) + 1 - a, 6) or da  # land on integers
            continue
        lo, hi, plo, vlo = a, an, p, v
        while hi - lo > 1e-6:
            mid = round(0.5 * (lo + hi), 9)
            rm = polish_sig(mid, plo, vlo, centre)
            if rm is None:
                say(f"   bisection polish failed at {mid}")
                break
            if max(rm[3], rm[4], rm[6]) > 0:
                hi = mid
            else:
                lo, plo, vlo = mid, rm[0], rm[1]
        rh = polish_sig(hi, plo, vlo, centre)
        cs, off, loc, sp, k = rh[3:] if rh else (cs, off, loc, sp, k)
        kind = ["centre", "pair birth", "pair split"][int(np.argmax([cs, off, sp]))]
        if kind == "centre":
            kind = "centre split" if centre else "centre appear"
        where = 0.0 if kind.startswith("centre") else (loc if kind == "pair birth" else plo[k])
        events.append(
            {"a_lo": lo, "a_hi": hi, "kind": kind, "at": where, "K_before": 2 * len(plo) + int(centre)}
        )
        save()
        say(f"EVENT {kind} at |y| = {where:.3f}, a in ({lo:.7f}, {hi:.7f}]   [{time.time() - t0:.0f}s]")
        a1 = round(hi + 0.002, 7)
        cands = []
        if kind == "centre appear":
            cands = [(list(plo), [x * (1 - m) for x in vlo], True) for m in (1e-3, 1e-2, 3e-2)]
        elif kind == "centre split":  # v4: old family polished at a_hi + 0.02 first; pitchfork width
            a1 = round(hi + 0.02, 7)
            I1 = M.IMP(a1, 9)
            po, vo, ro = I1.polish(list(plo), list(vlo), True)  # 1.71 sqrt(0.02) = 0.24 (q580)
            if float(ro) < 1e-35:
                plo, vlo = [float(t.mid()) for t in po], [float(t.mid()) for t in vo]
            vc = 1 - sum(vlo)
            cands = [([d] + list(plo), [vc] + list(vlo), False) for d in (0.24, 0.3, 0.2, 0.5, 0.7)]
            if float(ro) < 1e-35:  # v5: q492's fixed-eps young pair first (q587.split_centre)
                sc = B.split_centre(I1, plo, vlo, 0.05, say)
                if sc is not None:
                    cands = [(sc[0], sc[1], False)] + cands
        elif kind == "pair birth":
            for m in (1e-3, 1e-2):
                j = int(np.searchsorted(plo, loc))
                cands.append(
                    (list(np.insert(plo, j, loc)), list(np.insert(np.array(vlo) * (1 - m), j, m)), centre)
                )
        else:
            for d in (0.2, 0.1, 0.3):
                pj, vj = plo[k], vlo[k]
                pn = np.concatenate([np.delete(plo, k), [max(pj - d, 0.02), pj + d]])
                vn = np.concatenate([np.delete(vlo, k), [vj / 2, vj / 2]])
                o = np.argsort(pn)
                cands.append((list(pn[o]), list(vn[o]), centre))
        got = False
        if kind == "centre appear":  # v4: bordered continuation in the centre mass (q587)
            a1 = round(hi + 0.01, 7)
            I1 = M.IMP(a1, 9)
            po, vo, ro = I1.polish(list(plo), list(vlo), False)
            if float(ro) < 1e-35:
                bc = B.born_centre(I1, [float(t.mid()) for t in po], [float(t.mid()) for t in vo], say)
                if bc is not None:
                    cands = [(bc[0], bc[1], True)] + cands
        for pc, vc_, cc in cands:
            rc = polish_sig(a1, pc, vc_, cc, arb_only=True) or polish_sig(a1, pc, vc_, cc)
            if rc is None:
                say("   candidate: polish failed")
                continue
            if max(rc[3], rc[4], rc[6]) <= 0:
                a, p, v, centre = a1, rc[0], rc[1], cc
                K = 2 * len(p) + int(centre)
                got = True
                da = 0.005
                clean = 0
                rows.append(
                    {
                        "a": a,
                        "K": K,
                        "centre": centre,
                        "p": p,
                        "v": v,
                        "cs": rc[3],
                        "off": rc[4],
                        "off_at": rc[5],
                        "pairsplit": rc[6],
                        "res": rc[2],
                    }
                )
                save()
                say(f"   new family at a = {a}: K = {K}")
                break
            say(
                f"   candidate: signals centre {rc[3]:+.1e}, off {rc[4]:+.1e} at {rc[5]:.2f}, pair split {rc[6]:+.1e}"
            )
        if not got:
            say("STOP: no new family certified")
            break
    save()
    say(f"END at a = {a}   [{time.time() - t0:.0f}s]")
