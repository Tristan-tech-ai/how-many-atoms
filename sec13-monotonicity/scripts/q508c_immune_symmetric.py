"""Q508c: the immune-repertoire bridge by SYMMETRIC continuation. q508b v2 stopped at a = 4.124 (label B): after the 3 -> 4 split the
unconstrained Newton polish slid the new pair back onto one point (two coincident atoms at 0, i.e. the old 3-atom state). The problem is
symmetric and its optimum is unique, hence symmetric; so the polish works in the reduced space: mirror pairs +-p_i (pair mass v_i) and,
for odd K, a centre atom (mass v0 = 1 - sum v_i). Equations: g(p_i)/C - 1 = 0 and g'(p_i) = 0 (the centre's g(0) = C follows from
sum_k w_k g(y_k) = C); for even K the last pair mass is 1 - sum of the others and the last g-equation is dropped (implied).
Same model, same pre-registration as q508 (P1-P3), same change detection and bisection as q508b; new families seeded by a centre birth
(mass 0.2, 0.1, 0.3) or a split (centre atom -> pair at +-delta; pair -> two pairs), delta in 0.4, 0.25, 0.6, 0.15.
v17: RESTOL (default 1e-8) is the residual floor for stiff steps and new families; near a degenerate centre birth the
newborn's g''(0)/C ~ 1e-9 makes the Jacobian nearly singular (q508l: cold seeds do not converge); v16: COARSE=1 skips the bisection and lets new_family() choose by cost at each flagged change (for targets where the change
points are not needed, e.g. q508's state at a = 24); v15: after a change, every clean candidate family (K + 1 and, for a centre birth, K + 2) is built and the one with the
lowest cost is kept (the optimum is unique; at K ~ 25 the newborn centre atom has g''(0)/C ~ -2e-09, so the order of birth and split
cannot be read from signal signs); v14: after hybr, up to six Newton steps (central-difference Jacobian, least squares) take stiff residuals from ~1e-9 to
~1e-14 (hybr stops on a small relative step); v13: the change-detection threshold SIGTOL is an environment setting (default 1e-10; 1e-8 at K ~ 25, where residuals
of stiff families reach 5e-10 and a split was flagged on noise at 20.442482, label B); v12: new_family() accepts residual < 1e-8 as v11 does for steps (the immune run stopped after the 24 -> 25 birth at
20.437478); v11: a continuation step whose best residual is between 1e-10 and 1e-8 is accepted and reported (the immune run
stopped at 19.797 with residual 1.1e-09 right after the 23 -> 24 split); v10: for odd K the centre condition g(0) = C is an explicit equation (the last pair's g equation is the implied one);
v9: coincident atoms (spacing <= 1e-3) reject a step or a new family, and a positive signal with no crossing found
by bisection is discarded as a polish artefact (q508i: the NPMLE '15 -> 16' of v8 at 14.190840 was a K = 15 state with its centre atom
counted twice, label B); v8: the post-change family is built by new_family() (jumps 0.02/0.01/0.005, newborn masses scaled to the jump; q508h:
the NPMLE 14 -> 15 newborn weighs 5.5e-03 at +0.01, so seeds of 0.05 and more never converged); v7: the continuation step halves (0.05 down to 0.002) when a polish fails (q508g: after the NPMLE 13 -> 14 split
the K = 14 family is stiff, smallest singular value 4e-06 to 7e-06 against 10, but regular); v6: post-change jump 0.02 (q508f: after the NPMLE 13 -> 14 centre split every split seed failed at +0.1 and converged
cleanly at +0.01 to +0.05, the new pair opening as 1.55 sqrt(a - a_c)); v5: post-change jump 0.1 and newborn seed masses 0.05, 0.02, 0.1, 0.2 (q508d: masses 0.1-0.3 at a jump of 0.3 did not
converge or fell back to the old family once K ~ 9, label B); v4: the earlier of two positive signals defines the change; v3: the main block is a function main(AMAX, tag) so that q509 can reuse it with the NPMLE functional; v2: checkpoints every accepted state (q508c_ckpt.npz, resume with START), prints diagnostics at each change, and accepts +2
changes (an off-centre pair split into two pairs, or an off-centre pair birth); v1 stopped at a = 8.874 after a change 8 -> 9 read
as a split, which a symmetric state can only realise as +2.
Usage: py q508c_immune_symmetric.py [AMAX]"""

import sys, time, json
import numpy as np
from scipy.optimize import root

_a = sys.argv[1:]
sys.argv = [sys.argv[0]]
import q508_immune_bridge as Q

sys.argv = [sys.argv[0]] + _a
say = Q.say
t0 = time.time()
AMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
import os

SIGTOL = float(os.environ.get("SIGTOL", "1e-10"))
RESTOL = float(
    os.environ.get("RESTOL", "1e-8")
)  # v17: residual floor for stiff steps and new families                # v13: signal threshold (1e-8 at the K ~ 25 noise floor)


def full(p, v, centre):
    """reduced (p, v, centre) -> atoms and weights"""
    if centre:
        v0 = 1 - v.sum()
        ys = np.concatenate([-p[::-1], [0.0], p])
        ws = np.concatenate([v[::-1] / 2, [v0], v / 2])
    else:
        ys = np.concatenate([-p[::-1], p])
        ws = np.concatenate([v[::-1] / 2, v / 2])
    return ys, ws


def spolish(p, v, centre, xg, qx):
    n = len(p)

    def unpack(u):
        pp = u[:n]
        if centre:
            vv = u[n:]
        else:
            vv = np.concatenate([u[n:], [1 - u[n:].sum()]])
        return pp, vv

    def F(u):
        pp, vv = unpack(u)
        ys, ws = full(pp, vv, centre)
        if np.any(ws <= 0) or np.any(pp <= 1e-6) or np.any(np.diff(pp) <= 0):
            return np.full(len(u), 1e3)
        if centre:  # v10: the centre condition g(0) = C is imposed directly and the
            g0, g1, _, C = Q.gfun(
                ys, ws, np.concatenate([[0.0], pp]), xg, qx
            )  # last pair's g equation is the implied one; with a
            eq = [g1[1:] / C, g0[:-1] / C - 1]  # newborn centre atom of mass ~0 the old form left g(0) = C
        else:  # implied through that tiny mass only (stiff, q508c v9 STOP)
            g0, g1, _, C = Q.gfun(ys, ws, pp, xg, qx)
            eq = [g1 / C, g0[:-1] / C - 1]
        return np.concatenate(eq)

    u0 = np.concatenate([p, v if centre else v[:-1]])
    sol = root(F, u0, method="hybr", tol=1e-14)
    u = sol.x
    fu = F(u)
    res = np.abs(fu).max()
    for _ in range(6):  # v14: hybr stops on a small relative step while a stiff system's
        if res < 1e-13 or res > 1e-4:
            break  # residual is still ~1e-9; plain Newton steps with a central-
        J = np.empty((len(u), len(u)))  # difference Jacobian and least squares push it down, accepted only
        for i in range(len(u)):  # when the residual falls
            du = np.zeros_like(u)
            du[i] = 1e-7 * max(1.0, abs(u[i]))
            J[:, i] = (F(u + du) - F(u - du)) / (2 * du[i])
        un = u + np.linalg.lstsq(J, -fu, rcond=None)[0]
        fn_ = F(un)
        rn = np.abs(fn_).max()
        if rn >= res:
            break
        u, fu, res = un, fn_, rn
    pp, vv = unpack(u)
    return pp, vv, res


def signals(a, p, v, centre, xg, qx):
    ys, ws = full(p, v, centre)
    g0, g1, g2, C = Q.gfun(ys, ws, ys, xg, qx)
    yy = np.linspace(-a - 3, a + 3, 6001)
    G0, _, _, _ = Q.gfun(ys, ws, yy, xg, qx)
    excl = 0.3 * np.min(np.diff(ys)) if len(ys) > 1 else 0.3
    far = np.min(np.abs(yy[:, None] - ys[None, :]), 1) > excl
    return (G0[far] / C - 1).max(), abs(yy[far][np.argmax(G0[far])]), (g2 / C).max(), ys[int(np.argmax(g2))]


def cost(p, v, centre, xg, qx):
    """the objective at a symmetric state: immune C = sum q/Pt; NPMLE -sum q log Pt (lower is better in both)"""
    ys, ws = full(p, v, centre)
    Pt = (ws[:, None] * np.exp(-((xg[None, :] - ys[:, None]) ** 2) / 2)).sum(0)
    return float((qx * np.log(Pt)).sum() * -1) if Q.gfun.__name__ == "gfun_npmle" else float((qx / Pt).sum())


def new_family(ac, kind, lo, pl, vl, centre, K0):
    """v8: the post-change family. Jumps 0.02, 0.01, 0.005 past the change; a newborn's mass grows about as 0.5 x (a - a_c) (q508h:
    5.5e-03 at +0.01, 1.0e-02 at +0.02), so birth seeds use masses 0.5 J, J, 0.25 J, 0.02, 0.05 at jump J; split seeds use pair
    half-widths 1.5 sqrt(J), 0.25, 0.4, 0.1 (the new pair opens as about 1.55 sqrt(a - a_c), q508f). Returns (p, v, centre, a) or None.
    """
    xl, ql = Q.setup(lo)
    for J in (0.02, 0.01, 0.005):
        au = ac + J
        xu, qu = Q.setup(au)
        pr, vr, rr = (pl, vl, 0.0) if len(pl) == 0 else spolish(pl, vl, centre, xu, qu)
        b_, bwh, s_, wat = signals(au, pr, vr, centre, xu, qu)
        trials = []
        if kind == "birth":
            bw = signals(
                lo, *((pl, vl) if len(pl) == 0 else spolish(pl, vl, centre, xl, ql)[:2]), centre, xl, ql
            )[1]
            for m0 in (0.5 * J, J, 0.25 * J, 0.02, 0.05):
                if bw < 1e-3 and not centre:
                    trials.append((pr, vr * (1 - m0), True))
                else:
                    pp = np.concatenate([pr, [bw]])
                    vv = np.concatenate([vr * (1 - m0), [m0]])
                    o = np.argsort(pp)
                    trials.append((pp[o], vv[o], centre))
        else:
            wsp = signals(
                lo, *((pl, vl) if len(pl) == 0 else spolish(pl, vl, centre, xl, ql)[:2]), centre, xl, ql
            )[3]
            for d in (1.5 * np.sqrt(J), 0.25, 0.4, 0.1):
                if centre and abs(wsp) < 1e-9:
                    trials.append((np.concatenate([[d], pr]), np.concatenate([[1 - vr.sum()], vr]), False))
                else:
                    j = int(np.argmin(np.abs(pr - abs(wsp))))
                    pp = np.concatenate([np.delete(pr, j), [pr[j] - d, pr[j] + d]])
                    vv = np.concatenate([np.delete(vr, j), [vr[j] / 2] * 2])
                    o = np.argsort(pp)
                    trials.append((pp[o], vv[o], centre))
        if (
            kind == "birth" and bw < 1e-3 and not centre
        ):  # v15: a centre birth may be degenerate (newborn g''(0) ~ 0); also
            for m0 in (0.5 * J, J, 0.02):  # try the K0 + 2 family in which the newborn has already split
                for d in (1.5 * np.sqrt(J), 0.1, 0.25):
                    pp = np.concatenate([[d], pr])
                    vv = np.concatenate([[m0], vr * (1 - m0)])
                    trials.append((pp, vv, False))
        good = []
        if (
            len(pr) and rr < RESTOL and b_ < SIGTOL and s_ < SIGTOL
        ):  # v15: the old family itself competes (a change flagged on noise loses)
            good.append((cost(pr, vr, centre, xu, qu), len(full(pr, vr, centre)[0]), pr, vr, centre))
        for pt, vt, ct in trials:
            if len(pt) == 0:
                continue
            p2, v2, r2 = spolish(pt, vt, ct, xu, qu)
            if r2 < RESTOL:  # v12/v17: residual floor for new families
                b2, _, s2, _ = signals(au, p2, v2, ct, xu, qu)
                ys2 = full(p2, v2, ct)[0]
                if (
                    b2 < SIGTOL
                    and s2 < SIGTOL
                    and len(ys2) in (K0 + 1, K0 + 2)
                    and np.min(np.diff(ys2)) > 1e-3
                ):  # v9: distinct atoms
                    good.append((cost(p2, v2, ct, xu, qu), len(ys2), p2, v2, ct))
        if good:  # v15: among clean candidate families the lowest cost is the optimum
            good.sort(key=lambda g_: g_[0])
            if len(set(g_[1] for g_ in good)) > 1:
                say(
                    f"   candidate families at a = {au:.4f}: "
                    + ", ".join(f"K = {g_[1]} cost {g_[0]:.15f}" for g_ in good[:4])
                )
            return good[0][2], good[0][3], good[0][4], au
    return None


def analyse(a, p, v, centre):
    ys, ws = full(p, v, centre)
    gaps = np.diff(ys)
    mid = 0.5 * (ys[1:] + ys[:-1])
    rho = 1 / gaps
    rp = np.gradient(rho, mid)
    bulk = (a - np.abs(mid) >= 3) & (np.abs(mid) > 1.5)
    s = 16 * np.pi**2 * rho**2 * rp * np.sign(-mid)
    say(f"   a = {a}: K = {len(ys)}; gaps {np.round(gaps, 3)}")
    med = float(np.median(s[bulk])) if bulk.sum() >= 2 else float("nan")
    if bulk.sum() >= 2:
        say(f"   bulk chirp s (inward) {np.round(s[bulk], 3)}; median {med:.3f} (law ~1, evenly spaced 0)")
    say(f"   leading term 0.4002 a^(4/3) = {0.4002442498*a**(4/3):.2f}")
    return med, len(ys)


def main(AMAX, tag="immune"):
    global Q
    say(f"q508c symmetric continuation to a = {AMAX}")
    a = 0.5
    p, v, centre = np.array([]), np.array([]), True  # one atom at the centre
    if os.environ.get("START"):  # v2: resume from a checkpoint
        z = np.load(os.environ["START"])
        p, v, centre, a = z["p"], z["v"], bool(z["centre"]), float(z["a"])
        say(f"resumed from {os.environ['START']}: a = {a}, K = {len(full(p, v, centre)[0])}")
    changes = []
    saved = {}
    targets = [12.0, 18.0, 24.0]
    while a < AMAX - 1e-9:
        best = None
        for step in (
            0.05,
            0.02,
            0.01,
            0.005,
            0.002,
        ):  # v7: halve the step when the polish fails (stiff families)
            an = round(a + step, 10)
            xg, qx = Q.setup(an)
            if len(p) == 0:
                pn, vn, res = p, v, 0.0
            else:
                pn, vn, res = spolish(p, v, centre, xg, qx)
                if len(pn) and np.min(np.diff(full(pn, vn, centre)[0])) <= 1e-3:
                    res = 1.0  # v9: coincident atoms = failure
            if res <= 1e-10:
                break
            if best is None or res < best[3]:
                best = (an, pn, vn, res, xg, qx)
        if (
            res > 1e-10 and best is not None and best[3] <= RESTOL
        ):  # v11: a stiff family whose residual floor is between 1e-10 and
            an, pn, vn, res, xg, qx = best  # 1e-8 is accepted and the residual reported
            say(f"   a = {an}: accepted with residual {res:.1e} (stiff family, v11)")
        if res > 1e-10 and not (res <= RESTOL):
            say(f"STOP at a = {an}: residual {res:.1e} (label B until diagnosed)")
            break
        b, where, sp, which = signals(an, pn, vn, centre, xg, qx)
        if b > SIGTOL or sp > SIGTOL:

            def cross(which_sig):
                lo_, hi_, pl_, vl_ = a, an, p, v
                for _ in range(30):
                    am = 0.5 * (lo_ + hi_)
                    xm, qm = Q.setup(am)
                    pm, vm, rm = (pl_, vl_, 0.0) if len(pl_) == 0 else spolish(pl_, vl_, centre, xm, qm)
                    bm, _, spm, _ = signals(am, pm, vm, centre, xm, qm)
                    if (spm if which_sig == "split" else bm) > 0:
                        hi_ = am
                    else:
                        lo_, pl_, vl_ = am, pm, vm
                return lo_, hi_, pl_, vl_

            # v4: when both signals are positive, the one that crossed zero FIRST defines the change (v2/v3 gave the split precedence
            # and mislabelled the 8 -> 9 centre birth of the immune functional, label B)
            cands = [k_ for k_, val in (("split", sp), ("birth", b)) if val > SIGTOL]
            if os.environ.get("COARSE"):  # v16: no bisection; the change point is known to within the step,
                K0 = len(full(p, v, centre)[0])
                kind = "split" if sp >= b else "birth"  # and the family is chosen by cost among all
                nf = new_family(
                    a + 1e-9, kind, a, p, v, centre, K0
                )  # clean candidates (the old one included)
                if nf is None:
                    kind = "birth" if kind == "split" else "split"
                    nf = new_family(a + 1e-9, kind, a, p, v, centre, K0)
                if nf is None:
                    say(f"STOP (coarse): no clean family between {a} and {an}")
                    break
                p, v, centre, a = nf
                K1 = len(full(p, v, centre)[0])
                if K1 != K0:
                    changes.append((a, K0, K1, kind))
                    say(
                        f"CHANGE {K0} -> {K1} (coarse, {kind} signal) within ({a - 0.02:.4f}, {a:.4f}]   [{time.time() - t0:.0f}s]"
                    )
                np.savez(f"q508c_ckpt_{tag}.npz", p=p, v=v, centre=centre, a=a)
                continue
            res_c = {k_: cross(k_) for k_ in cands}
            kind = min(res_c, key=lambda k_: res_c[k_][1])
            lo, hi, pl, vl = res_c[kind]
            if an - lo < 1e-6:  # v9: no crossing inside (a, an): the positive signal came from
                say(
                    f"   positive signal at a = {an} only at the step point (no crossing found by bisection); step state discarded"
                )
                p, v, a = pl, vl, lo
                continue  # the step's own polish (another branch), label B
            if len(cands) == 2:
                say(
                    f"   both signals positive at a = {an}: split crosses at {0.5*sum(res_c['split'][:2]):.6f}, birth at {0.5*sum(res_c['birth'][:2]):.6f}; taking {kind}"
                )
            ac = 0.5 * (lo + hi)
            K0 = len(full(p, v, centre)[0])
            bd, wd, sd, atd = (
                signals(hi, *spolish(pl, vl, centre, *Q.setup(hi))[:2], centre, *Q.setup(hi))
                if len(pl)
                else (0, 0, 0, 0)
            )
            say(
                f"   diagnostics just past the change (a = {hi:.6f}): birth signal {bd:+.2e} at |y| = {wd:.4f}; split signal {sd:+.2e} at atom y = {atd:+.4f}; "
                f"state p = {np.round(pl, 4)}, centre atom {centre}"
            )
            changes.append((ac, K0, K0 + 1, kind))
            say(f"CHANGE {K0} -> {K0 + 1} ({kind}) at a = {ac:.6f}   [{time.time() - t0:.0f}s]")
            nf = new_family(ac, kind, lo, pl, vl, centre, K0)
            ok = nf is not None
            if ok:
                p, v, centre, a = nf
                say(f"   new family accepted at a = {a:.4f}: K = {len(full(p, v, centre)[0])}")
            if not ok:
                say(f"STOP: no clean new family after the change at a = {ac:.6f} (label B until diagnosed)")
                break
            continue
        p, v, a = pn, vn, an
        np.savez(f"q508c_ckpt_{tag}.npz", p=p, v=v, centre=centre, a=a)
        for t in targets:
            if t not in saved and abs(a - t) < 0.026:
                saved[t] = (p.copy(), v.copy(), centre)
                say(f"saved state at a = {t}: K = {len(full(p, v, centre)[0])}   [{time.time() - t0:.0f}s]")
    say("\nchanges: " + ", ".join(f"{c[1]}->{c[2]} {c[3]} {c[0]:.4f}" for c in changes))
    res_out = {}
    for t, (pp, vv, cc) in sorted(saved.items()):
        med, K = analyse(t, pp, vv, cc)
        res_out[str(t)] = {
            "K": K,
            "median_s": med,
            "p": list(map(float, pp)),
            "v": list(map(float, vv)),
            "centre": cc,
        }
    if 12.0 in saved and 24.0 in saved:
        say(
            f"log-slope of K between 12 and 24: {np.log(res_out['24.0']['K']/res_out['12.0']['K'])/np.log(2):.3f}"
        )
    json.dump({"changes": changes, "states": res_out}, open(f"results_q508c_{tag}.json", "w"), indent=1)
    say(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(AMAX)
