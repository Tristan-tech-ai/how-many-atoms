"""Q484: the HIGHER-TARGET cheapest test (23 Sep). Least-favourable priors for the BOUNDED NORMAL MEAN under squared
error: X ~ N(theta, 1), theta in [-m, m]. How fast does the number K(m) of support points grow?

WHY. The audit (papers/TIER3_AUDIT.md, citations re-checked 23 Sep) finds no growth rate, bound or table of K(m)
in the literature: finiteness (Ghosh 1964), K = 2 up to m ~ 1.057 and K = 3 just beyond (Casella-Strawderman 1981;
Johnstone's monograph, Sec. 4.6: "numerical work shows" 1.057), a cos^2 continuum limit (Bickel 1981), and
Johnstone's remark that the support points "become gradually more spaced out" as m grows. Wang-Barletta-Dytso
(arXiv 2512.22691) name support bounds for this problem as future work. Abbott-Machta (2019) conjecture that their
4/3 law "should apply to all" minimax problems of this kind, untested for squared error.

THE PROBLEM. For a prior pi = sum w_k delta(theta_k), the Bayes rule is delta(x) = sum w_k theta_k phi(x - theta_k) /
sum w_k phi(x - theta_k), its risk R(theta) = E_theta (delta(X) - theta)^2, the Bayes risk r(pi) = sum w_k R(theta_k).
The least-favourable prior maximises r (concave in pi); KKT: R(theta) <= r on [-m, m], equality on the support.
Envelope theorem: dr/dw_k = R(theta_k), dr/dtheta_k = w_k R'(theta_k) with delta held fixed.
METHOD. Exchange algorithm, continuation in m: L-BFGS-B on -r over positions (bounded to [-m, m]) and softmax weights
with the analytic gradient; merge atoms closer than 1e-4, drop weights below 1e-10; insert an atom at the maximum of
R - r on a fine theta grid while it exceeds TOL; repeat. Quadrature: trapezoid in x on [-m-10, m+10], step 0.01
(exponentially accurate for these Gaussian integrands).

PRE-REGISTERED, before the run:
 (C) CONTROL: the 2 -> 3 transition (an atom appears at 0) must fall within 2e-3 of Johnstone's 1.057. FAIL stops
     the reading of everything else.
 (M) MEASUREMENT: K(m) on m = 0.5 .. MMAX by 0.05, each state with max_theta (R - r) <= TOL = 1e-9 printed. Three
     readings of the top octave are reported, none predicted in advance: the free exponent of K against L = 2m
     (Fisher length, unit noise), the fit K = c L^(4/3) + b (capacity law), and K = c L + b (Johnstone's
     noise-scale heuristic); plus the centre spacing against m.
v2 (02:20): merge atoms closer than 0.02; insert only where the violation is more than 0.05 from every atom; a
violation next to an atom gets more L-BFGS iterations (flag -2 if it persists). v1 counted split atoms beyond m = 6.9.
Usage: py q484_lfp_bnm.py [MMAX] [DM]"""

import sys, time, json
import numpy as np
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
MMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
DM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
TOL = 1e-9
MERGE = 0.02  # v2: atoms closer than this are one atom (v1 used 1e-4 and split atoms beyond m ~ 6.9)
NEAR = 0.05  # v2: a violation within NEAR of an atom is an unconverged position, not a missing atom
H = 0.01
S2P = 1 / np.sqrt(2 * np.pi)


class Prob:
    def __init__(self, m):
        self.m = m
        self.x = np.arange(-m - 10, m + 10 + H / 2, H)

    def phi(self, th):
        return S2P * np.exp(-0.5 * (self.x[None, :] - np.asarray(th)[:, None]) ** 2)

    def rule(self, th, w):
        P = self.phi(th)
        Z = w @ P
        N = (w * th) @ P
        return N / Z

    def risk(self, th_eval, delta):
        """R(theta) and R'(theta) with the rule delta held fixed"""
        P = self.phi(th_eval)
        e = delta[None, :] - np.asarray(th_eval)[:, None]
        R = H * (e * e * P).sum(1)
        dR = H * ((-2 * e + e * e * (self.x[None, :] - np.asarray(th_eval)[:, None])) * P).sum(1)
        return R, dR

    def solve(self, th, w, iters=4000):
        K = len(th)

        def f(z):
            t = z[:K]
            a = z[K:]
            a = a - a.max()
            ww = np.exp(a)
            ww /= ww.sum()
            d = self.rule(t, ww)
            R, dR = self.risk(t, d)
            r = ww @ R
            gt = ww * dR
            gw = R  # dr/dw_k
            ga = ww * (gw - r)  # softmax chain rule
            return -r, -np.concatenate([gt, ga])

        z0 = np.concatenate([th, np.log(np.maximum(w, 1e-300))])
        bounds = [(-self.m, self.m)] * K + [(None, None)] * K
        res = minimize(
            f,
            z0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=iters, maxcor=50, ftol=1e-16, gtol=1e-13),
        )
        t = res.x[:K]
        a = res.x[K:]
        a = a - a.max()
        ww = np.exp(a)
        ww /= ww.sum()
        return t, ww, -res.fun

    def kkt(self, th, w, n=4001):
        d = self.rule(th, w)
        grid = np.linspace(-self.m, self.m, n)
        R, _ = self.risk(grid, d)
        Ra, _ = self.risk(th, d)
        r = w @ Ra
        i = int(np.argmax(R - r))
        return float(R[i] - r), float(grid[i]), float(r)


def clean(th, w):
    o = np.argsort(th)
    th, w = th[o], w[o]
    keep_t, keep_w = [th[0]], [w[0]]
    for t, ww in zip(th[1:], w[1:]):
        if t - keep_t[-1] < MERGE:
            tot = keep_w[-1] + ww
            keep_t[-1] = (keep_t[-1] * keep_w[-1] + t * ww) / tot
            keep_w[-1] = tot
        else:
            keep_t.append(t)
            keep_w.append(ww)
    th, w = np.array(keep_t), np.array(keep_w)
    m = w > 1e-10
    return th[m], w[m] / w[m].sum()


def optimum(m, th, w):
    P = Prob(m)
    th = np.clip(th, -m, m)
    extra = 0
    for rnd in range(60):
        th, w, r = P.solve(th, w, iters=4000 + 4000 * extra)
        th, w = clean(th, w)
        viol, where, r = P.kkt(th, w)
        if viol <= TOL:
            return th, w, r, viol, rnd
        if np.min(np.abs(th - where)) < NEAR:
            extra += 1
            if extra > 4:
                return th, w, r, viol, -2  # flagged: residual violation next to an atom
            continue
        th = np.append(th, where)
        w = np.append(w * 0.99, 0.01)
        # symmetric insertion keeps the solution symmetric
        if abs(where) > 1e-3:
            th = np.append(th, -where)
            w = np.append(w * 0.99, 0.01)
            w /= w.sum()
    return th, w, r, viol, -1


if __name__ == "__main__":
    t0 = time.time()
    say(
        f"least-favourable priors, bounded normal mean, squared error; m = 0.5..{MMAX} by {DM}; TOL {TOL}; quadrature step {H}"
    )
    th = np.array([-0.5, 0.5])
    w = np.array([0.5, 0.5])
    rows = []
    Kprev = None
    trans = []
    m = 0.5
    while m <= MMAX + 1e-9:
        th = th * (m / max(np.abs(th).max(), 1e-9)) if len(th) else th
        th, w, r, viol, rnd = optimum(m, th, w)
        K = len(th)
        pos = np.sort(th)
        gaps = np.diff(pos)
        cgap = gaps[len(gaps) // 2] if len(gaps) else float("nan")
        rows.append(
            dict(
                m=round(m, 6),
                K=K,
                r=r,
                viol=viol,
                rounds=rnd,
                cgap=float(cgap),
                wmin=float(w.min()),
                theta=[float(t) for t in pos],
            )
        )
        if Kprev is not None and K != Kprev:
            trans.append((round(m, 4), Kprev, K))
            say(f"   TRANSITION {Kprev} -> {K} between m = {m - DM:.3f} and {m:.3f}")
        if abs(m * 20 - round(m * 20)) < 1e-9 and (round(m * 20) % 10 == 0 or K != Kprev):
            say(
                f"m = {m:6.3f}: K = {K:3d}, Bayes risk {r:.10f}, max(R - r) {viol:+.2e}, min w {w.min():.4f}, "
                f"centre gap {cgap:.4f}, atoms >= 0: {np.round(pos[pos >= -1e-9][:8], 4)}   [{time.time() - t0:.0f}s]"
            )
        Kprev = K
        m = round(m + DM, 10)
    json.dump(dict(rows=rows, transitions=trans), open("results_q484_lfp_bnm.json", "w"), indent=1)
    # control
    t23 = [t for t in trans if t[1] == 2 and t[2] == 3]
    say(f"\n(C) 2 -> 3 transition bracket: {t23[:1]}   (Johnstone: 1.057)")
    ms = np.array([r["m"] for r in rows])
    Ks = np.array([r["K"] for r in rows])
    say("(M) transitions: " + ", ".join(f"{a}->{b} at {mm}" for mm, a, b in trans))
    sel = ms >= ms.max() / 2
    L = 2 * ms[sel]
    if sel.sum() > 5 and len(set(Ks[sel])) > 2:
        e, c = np.polyfit(np.log(L), np.log(Ks[sel]), 1)
        say(f"    free exponent over the top octave: K ~ L^{e:.3f}")
        for name, f in (("c L^(4/3) + b", L ** (4 / 3)), ("c L + b", L), ("c L^(2/3) + b", L ** (2 / 3))):
            A = np.vstack([f, np.ones_like(f)]).T
            (cc, bb), res, *_ = np.linalg.lstsq(A, Ks[sel], rcond=None)
            say(
                f"    {name}: c = {cc:.4f}, b = {bb:+.3f}, rms {np.sqrt(np.mean((Ks[sel] - A @ [cc, bb])**2)):.3f}"
            )
    say(f"[{time.time() - t0:.0f}s]")
