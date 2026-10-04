"""Q505: the RATE-DISTORTION bridge of the highest target (memory project-highest-target-optimal-discreteness, lapor177).

CLAIM UNDER TEST. The population NPMLE of a Gaussian location mixture with kernel variance sigma^2 is the rate-distortion problem with
quadratic distortion at lambda = 1/sigma^2 (Yang et al., arXiv 2310.18908). If so, the reproduction alphabet of the RD-optimal test
channel for a UNIFORM source changes size exactly where the record's NPMLE for uniform data changes (q496, events a_star), and the
deterministic-annealing critical temperatures of a uniform source (Rose, Gurewitz, Fox 1990) are those same points.

THIS SOLVER IS INDEPENDENT of the record's NPMLE code: nothing from q49x is imported; the source, the kernel and the units are set up
afresh. Source X ~ U[-1, 1]; distortion (x - y)^2; slope s: the optimal reproduction law q minimises
    F_s(q) = - int p(x) log int q(dy) exp(-s (x - y)^2) dx,
with KKT function c(y) = int p(x) exp(-s (x - y)^2) / Z(x) dx <= 1, equality on supp q, Z(x) = int q(dy') exp(-s (x - y')^2).
Kernel variance 1/(2s), so the half-length in noise units is L = sqrt(2 s) and the DA temperature is T = 1/s = 2/L^2.
Method: Blahut-Arimoto on a 2001-point reproduction grid for the seed, atoms by clustering the mass, Newton polish of
(c(y_k) - 1, c'(y_k)) in positions and weights, continuation in s; a change is flagged when either the birth signal
max_{y away from atoms} c(y) - 1 or the split signal max_k c''(y_k) crosses zero, and it is located by bisection on the old family.
x-integrals by 3000-point Gauss-Legendre on [-1, 1].

PRE-REGISTERED, before the first run:
 C1 (control, known in the DA literature): the alphabet changes 1 -> 2 at L = sqrt 3 = 1.7320508 (T = 2/3 = 2 Var X), by a split.
 P2-P6 (predictions, parameter-free, from results_q496_npmle_uniform.json events a_star, read before this run):
    L_2 = 2.979950 (2 -> 3, birth at the centre), L_3 = 4.100006 (3 -> 4, split of the centre atom), L_4 = 5.143951 (4 -> 5, birth),
    L_5 = 6.132721 (5 -> 6, split), L_6 = 7.078094 (6 -> 7, birth); the run goes to LMAX = 7.2 by default.
 PASS if every located change is within 1e-3 relative of its prediction and of the same kind; FAIL otherwise, with the diagnosis
 labelled M (the bridge is false) or B (a bug or an instrument limit), decided by reading the code and the signals.
 Independent cross-check: at the middle of each plateau, Blahut-Arimoto from the uniform grid alone must show the same number of mass
 clusters as the continuation.
v5: BA clusters are the local maxima of the mass (v4 fused a freshly split pair, stopping at L = 4.25, label B); the new family is
seeded at L_c + 0.3 with a split fallback. v4 had located 2 -> 3 at 2.9799389 and 3 -> 4 at 4.0999814 before stopping.
v4: the direct seed failed (a pitchfork pair is too close to seed by +-0.05; the failure cascaded into false changes, label B);
new families now come from BA at L_c + 0.15 and every state must have residual < 1e-10 or the run stops.
v3: new families were seeded directly (split: the atom becomes a pair at +-0.05; birth: a 2% atom at the signal maximum), because
Blahut-Arimoto near a transition converges too slowly to serve as a seed; BA stays as the plateau cross-check (801 points, 4000 iterations).
v2 (before any comparison was read): the first launch classified the 1 -> 2 change as a birth because the birth signal was read
0.05 from the atom, where c already exceeds 1 once c''(atom) > 0; the split signal now takes precedence and the exclusion zone is
0.3 of the smallest gap (label B, the instrument). Its location, 1.7327007, was bisected on the wrong signal and is discarded.
Usage: py q505_rd_bridge.py [LMAX]"""

import os, sys, json, time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import root

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
LMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 7.2
t0 = time.time()

xg, wg = leggauss(3000)
px = wg * 0.5  # source U[-1, 1]: density 1/2, quadrature weights


def kern(s, y):
    return np.exp(-s * (xg[None, :] - np.asarray(y)[:, None]) ** 2)


def cfun(s, ys, ws, y):
    """c(y), c'(y), c''(y) for the reproduction law (ys, ws)"""
    Z = (ws[:, None] * kern(s, ys)).sum(0)
    E = kern(s, y)
    d = xg[None, :] - np.asarray(y)[:, None]
    c0 = (E * px / Z).sum(1)
    c1 = (E * 2 * s * d * px / Z).sum(1)
    c2 = (E * (4 * s * s * d * d - 2 * s) * px / Z).sum(1)
    return c0, c1, c2


def ba(s, n=801, iters=20000, tol=1e-9, nx=800):
    """Blahut-Arimoto seed on its own coarser quadrature (nx Gauss-Legendre nodes); the polish uses the 3000-node rule"""
    xb, wb = leggauss(nx)
    pb = wb * 0.5
    y = np.linspace(-1, 1, n)
    q = np.full(n, 1.0 / n)
    K = np.exp(-s * (xb[None, :] - y[:, None]) ** 2)
    for it in range(iters):
        Z = q @ K
        c = (K * pb / Z).sum(1)
        q = q * c
        q /= q.sum()
        if it % 100 == 0 and c.max() - 1 < tol:
            break
    return y, q


def clusters(y, q, thr=1e-6):
    """v5: atoms = local maxima of the grid mass above thr * max; every grid point's mass goes to its nearest peak
    (v4 merged contiguous mass, which fused a freshly split pair into one cluster)"""
    pk = [i for i in range(1, len(q) - 1) if q[i] >= q[i - 1] and q[i] > q[i + 1] and q[i] > thr * q.max()]
    if q[0] > q[1] and q[0] > thr * q.max():
        pk = [0] + pk
    if q[-1] > q[-2] and q[-1] > thr * q.max():
        pk = pk + [len(q) - 1]
    yp = y[pk]
    lab = np.argmin(np.abs(y[:, None] - yp[None, :]), 1)
    ys = np.array([(y[lab == k] * q[lab == k]).sum() / q[lab == k].sum() for k in range(len(pk))])
    ws = np.array([q[lab == k].sum() for k in range(len(pk))])
    return ys, ws


def polish(s, ys, ws):
    K = len(ys)

    def F(v):
        yy, ww = v[:K], v[K:]
        if np.any(ww <= 0) or np.any(np.abs(yy) > 1.5):
            return np.full(2 * K, 1e3)
        c0, c1, _ = cfun(s, yy, ww, yy)
        out = np.concatenate([c0 - 1, c1])
        return out if np.all(np.isfinite(out)) else np.full(2 * K, 1e3)

    sol = root(F, np.concatenate([ys, ws]), method="hybr", tol=1e-14)
    v = sol.x
    res = np.abs(F(v)).max()
    return v[:K], v[K:], res


def signals(s, ys, ws):
    yy = np.linspace(-1, 1, 4001)
    excl = (
        0.3 * np.min(np.diff(np.sort(ys))) if len(ys) > 1 else 0.25
    )  # v2: wider exclusion (v1 0.05 read a split as a birth)
    far = np.min(np.abs(yy[:, None] - ys[None, :]), 1) > excl
    c0, _, _ = cfun(s, ys, ws, yy)
    birth = (c0[far] - 1).max()
    where = yy[far][np.argmax(c0[far] - 1)]
    _, _, c2 = cfun(s, ys, ws, ys)
    split = c2.max()
    which = int(np.argmax(c2))
    return birth, where, split, which


def state_at(s, ys, ws):
    ys2, ws2, res = polish(s, ys, ws)
    o = np.argsort(ys2)
    return ys2[o], ws2[o], res


if __name__ == "__main__":
    ev = json.load(open(os.path.join(HERE, "results_q496_npmle_uniform.json")))["events"]
    pred = [
        (e["a_star"], e["K_before"], e["K_after"], "split" if "split" in e["type"] else "birth")
        for e in ev
        if e["a_star"] <= LMAX + 0.1
    ]
    say(
        f"q505 rate-distortion bridge, source U[-1,1], distortion (x-y)^2; L = sqrt(2 s), T = 2/L^2; LMAX = {LMAX}"
    )
    say("predictions read from results_q496_npmle_uniform.json (a_star, K_before -> K_after, kind):")
    for p in pred:
        say(f"   L = {p[0]:.6f}   {p[1]} -> {p[2]}   {p[3]}")
    L = 1.2
    s = L * L / 2
    ys, ws = np.array([0.0]), np.array([1.0])
    import os as _os0

    if _os0.environ.get("START"):  # v7: resume from a saved state
        z = np.load(_os0.environ["START"])
        ys, ws, L = z["ys"], z["ws"], float(z["L"])
        s = L * L / 2
        say(f"resumed from {_os0.environ['START']}: L = {L:.4f}, K = {len(ys)}")
    found = []
    step = 0.02
    while L < LMAX:
        Ln = L + step
        sn = Ln * Ln / 2
        ys_n, ws_n, res = state_at(sn, ys, ws)
        if res > 1e-10:
            say(f"   STOP at L = {Ln:.4f}: continuation residual {res:.1e} (label B until diagnosed)")
            break
        b, where, sp, which = signals(sn, ys_n, ws_n)
        if b > 1e-12 or sp > 1e-12:
            kind = "split" if sp > 1e-12 else "birth"  # v2: the split signal takes precedence
            lo, hi = L, Ln
            yl, wl = ys, ws
            for _ in range(40):  # bisection on the old family
                Lm = 0.5 * (lo + hi)
                sm = Lm * Lm / 2
                ym, wm, rm = state_at(sm, yl, wl)
                bm, _, spm, _ = signals(sm, ym, wm)
                sig = bm if kind == "birth" else spm
                if sig > 0:
                    hi = Lm
                else:
                    lo, yl, wl = Lm, ym, wm
            Lc = 0.5 * (lo + hi)
            K0 = len(ys)
            found.append((Lc, K0, K0 + 1, kind))
            say(
                f"   CHANGE {K0} -> {K0 + 1} ({kind}) at L = {Lc:.7f}  (T = {2/Lc**2:.7f})   old-family residual {res:.1e}   [{time.time() - t0:.0f}s]"
            )
            # new family (v4): Blahut-Arimoto at L_c + 0.15, away from the pitchfork where BA converges, then polish; strict acceptance
            Lu = Lc + 0.3
            su = Lu * Lu / 2
            yb, qb = ba(su)
            yc, wc = clusters(yb, qb, thr=1e-6)
            ys, ws, res = state_at(su, yc, wc)
            if (
                len(ys) == K0 and kind == "birth"
            ):  # v7 fallback: re-polish the old family, insert at the signal maximum
                yr, wr, _ = state_at(su, yl, wl)
                _, wh, _, _ = signals(su, yr, wr)
                for m0 in (0.1, 0.2, 0.05, 0.3):
                    yb2 = np.concatenate([yr, [wh]])
                    wb2 = np.concatenate([wr * (1 - m0), [m0]])
                    o = np.argsort(yb2)
                    ys, ws, res = state_at(su, yb2[o], wb2[o])
                    if res < 1e-10 and len(ys) == K0 + 1:
                        break
            if len(ys) == K0 and kind == "split":  # fallback: split the atom nearest the old split site
                j = int(np.argmin(np.abs(ys)))
                gap = np.min(np.abs(np.delete(ys, j) - ys[j])) if len(ys) > 1 else 0.5
                ysd = np.sort(np.concatenate([np.delete(ys, j), [ys[j] - 0.2 * gap, ys[j] + 0.2 * gap]]))
                wsd = np.concatenate([np.delete(ws, j), [ws[j] / 2, ws[j] / 2]])
                wsd = wsd[
                    np.argsort(np.concatenate([np.delete(ys, j), [ys[j] - 0.2 * gap, ys[j] + 0.2 * gap]]))
                ]
                ys, ws, res = state_at(su, ysd, wsd)
            bn, _, spn, _ = signals(su, ys, ws)
            say(
                f"      new family at L = {Lu:.4f}: BA gave {len(yc)} clusters; polished {len(ys)} atoms {np.round(ys, 4)}, residual {res:.1e}, "
                f"signals birth {bn:.1e} split {spn:.1e}   [{time.time() - t0:.0f}s]"
            )
            if res > 1e-10 or len(ys) != K0 + 1 or bn > 1e-12 or spn > 1e-12:
                say("   STOP: the new family is not a clean stationary state (label B until diagnosed)")
                break
            L = Lu
            continue
        ys, ws, L = ys_n, ws_n, Ln
    import os as _os

    if _os.environ.get("SAVE"):  # v6: save the final state (for q504); v7 adds a birth fallback and START
        np.savez(_os.environ["SAVE"], ys=ys, ws=ws, L=L)
        say(f"saved final state L = {L:.4f}, K = {len(ys)} to {_os.environ['SAVE']}")
    if _os.environ.get("NOCHECK") == "1":
        sys.exit()
    # independent plateau cross-check by BA alone
    say("\nplateau cross-check (Blahut-Arimoto from the uniform grid only):")
    edges = [1.2] + [f[0] for f in found] + [LMAX]
    for a, b in zip(edges[:-1], edges[1:]):
        Lm = 0.5 * (a + b)
        yb, qb = ba(Lm * Lm / 2)
        yc, wc = clusters(yb, qb)
        say(f"   L = {Lm:.3f}: {len(yc)} mass clusters at {np.round(yc, 4)}")
    say("\ncomparison with the NPMLE predictions:")
    ok = True
    for i, p in enumerate(pred):
        if i >= len(found):
            say(f"   L = {p[0]:.6f} ({p[1]} -> {p[2]}, {p[3]}): not reached")
            continue
        f = found[i]
        rel = abs(f[0] - p[0]) / p[0]
        same = (f[1], f[2], f[3]) == (p[1], p[2], p[3])
        ok = ok and rel < 1e-3 and same
        say(
            f"   predicted L = {p[0]:.6f} ({p[1]}->{p[2]} {p[3]})   found {f[0]:.6f} ({f[1]}->{f[2]} {f[3]})   rel {rel:.1e}   {'PASS' if rel < 1e-3 and same else 'FAIL'}"
        )
    say(
        f"\nVERDICT: {'PASS' if ok and len(found) >= len(pred) else 'FAIL or incomplete'}   [{time.time() - t0:.0f}s]"
    )
    json.dump(
        {"found": found, "pred": pred}, open(os.path.join(HERE, "results_q505_rd_bridge.json"), "w"), indent=1
    )
