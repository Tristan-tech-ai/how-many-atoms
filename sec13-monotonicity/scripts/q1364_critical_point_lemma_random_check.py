"""q1364 (8 Oct 2026, EXPLORATORY reading, no item, no verdict; run as py q1364_critical_point_lemma_random_check.py from temporal_within/centre_scripts/): numerical check of the two inequalities behind the critical-point lemma of the Tier A note, on random inputs (not optimal ones).
Lemma pieces. (L1) A mixture Q = phi * P of K Gaussians of equal variance (unit) with K distinct atoms has at most K modes, i.e. at most 2K - 1 sign changes of Q'. (L2) For D(x) = -(phi * log Q)(x), D' = -phi * (Q'/Q), so by the variation-diminishing property of the Gaussian kernel the number of sign changes of D' on the real line is at most the number of sign changes of Q', at most 2K - 1.
Test. For K = 2 to 7, 700 random inputs each (atoms uniform in [-8, 8], weights from a Dirichlet distribution, atoms at least 0.05 apart): sign changes of Q' and of D' on the grid x in [-16, 16], step 0.025, counted only where the absolute value exceeds 1e-13 times the largest absolute value of the function. Printed per K: the largest count of Q' changes, of D' changes, the bound 2K - 1, and how many inputs attain the bound (so that the bound is not vacuous).
Controls (first): c1 the counter on a synthetic sequence with three known sign changes; c2 negative control for L1: a mixture of K = 2 components of the positive but not totally positive shape psi(t) = phi_4(t)(1 + 0.9 cos 1.5 t) (phi_4 the Gaussian of standard deviation 4) has inputs with more than 2 modes (more than 3 sign changes of Q'), so the check can fail; c3 negative control for L2: the same kernel psi applied in place of phi to the Gaussian mixture of K = 3 gives at least one input with more than 2K - 1 sign changes of the smoothed derivative. (Mixtures of two or three Gaussians with unequal variances were also searched, 4000 inputs each, and did not exceed K modes, so they are not used as a control.)
Usage: py q1364_critical_point_lemma_random_check.py"""

import os, sys
import numpy as np


def die(msg):
    print(msg, flush=True)
    os._exit(2)


def changes(v, floor):
    s = v[np.abs(v) > floor * np.max(np.abs(v))]
    sg = np.sign(s)
    return int(np.sum(sg[1:] != sg[:-1])) if len(sg) > 1 else 0


c1 = changes(np.array([1, 2, 1, -1, -2, 1.0, 1, -1]), 1e-13) == 3
print(f"control c1 (counter on a synthetic sequence): {c1}", flush=True)
if not c1:
    die("READING q1364: VOID (control c1)")
rng = np.random.default_rng(20261008)
xg = np.arange(-16, 16, 0.025)
yg = np.arange(-30, 30, 0.01)
phi = lambda t, s=1.0: np.exp(-t * t / (2 * s * s)) / (s * np.sqrt(2 * np.pi))


def draw(K):
    while True:
        u = np.sort(rng.uniform(-8, 8, K))
        if K == 1 or np.min(np.diff(u)) > 0.05:
            return u, rng.dirichlet(np.ones(K))


def qprime_y(u, w, kern):  # Q and Q' on yg
    Q = sum(wk * kern(yg - uk) for uk, wk in zip(u, w))
    h = 1e-4
    Qp = sum(wk * (kern(yg + h - uk) - kern(yg - h - uk)) / (2 * h) for uk, wk in zip(u, w))
    return Q, Qp


def dprime_x(u, w, kern):
    Q, Qp = qprime_y(u, w, kern)
    f = Qp / Q
    K = np.array([kern(xx - yg) for xx in xg[::1]])
    return -(K @ f) * 0.01


def modes_psi(rng):
    best = 0
    for _ in range(300):
        u = np.sort(rng.uniform(-8, 8, 2))
        w = rng.dirichlet(np.ones(2))
        Q = sum(wk * psi(yg - uk) for uk, wk in zip(u, w))
        best = max(best, changes(np.gradient(Q, yg), 1e-9))
    return best


psi = lambda t: phi(t, 4.0) * (1 + 0.9 * np.cos(1.5 * t))
KPHI = np.array([phi(xx - yg) for xx in xg])
KPSI = np.array([psi(xx - yg) for xx in xg])
c2 = modes_psi(rng) > 3
print(
    f"control c2 (a mixture of two non-totally-positive components has more than 2 modes): {c2}", flush=True
)


def viol_kernel(rng):
    best = 0
    for _ in range(200):
        K = 3
        u, w = draw(K)
        Q, Qp = qprime_y(u, w, phi)
        f = Qp / Q
        best = max(best, changes(-(KPSI @ f) * 0.01, 1e-9) - (2 * K - 1))
    return best


c3 = viol_kernel(rng) > 0
print(
    f"control c3 (a positive kernel that is not totally positive breaks the bound 2K - 1): {c3}", flush=True
)
if not (c2 and c3):
    die("READING q1364: VOID (control)")
print("K | trials | max sign changes of Q' | max of D' | bound 2K - 1 | inputs attaining the bound in D'")
allok = True
for K in range(2, 8):
    mq = md = 0
    att = 0
    n = 700
    for _ in range(n):
        u, w = draw(K)
        Q, Qp = qprime_y(u, w, phi)
        cq = changes(Qp, 1e-13)
        f = Qp / Q
        dp = -(KPHI @ f) * 0.01
        cd = changes(dp, 1e-13)
        mq = max(mq, cq)
        md = max(md, cd)
        att += int(cd == 2 * K - 1)
    ok = mq <= 2 * K - 1 and md <= 2 * K - 1
    allok = allok and ok
    print(
        f"  {K} | {n} | {mq} | {md} | {2 * K - 1} | {att} -> {'within the bound' if ok else 'VIOLATES'}",
        flush=True,
    )
print(f"all within the bounds: {allok}")
print("READING q1364 (exploratory, no verdict)")
