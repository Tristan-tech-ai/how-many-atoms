"""Step checks for the pointwise capacity edge law (pred_cap_edge_pointwise.txt). Reads a stored capacity state (read only);
writes nothing. Double precision for the sup norms on grids; mpmath for c* and J'(A)."""

import json, sys, numpy as np, mpmath as mp

mp.mp.dps = 30
d = json.load(open(sys.argv[1]))
A = float(d["A"])
u = [float(s) for s in d["u"]]
w = [float(s) for s in d["w"]]
we = float(d["we"])
wc = None if d.get("wc") is None else float(d["wc"])
xs = np.array([-A] + [-x for x in u[::-1]] + ([0.0] if wc is not None else []) + u + [A])
ws = np.array([we] + w[::-1] + ([wc] if wc is not None else []) + w + [we])
ph = lambda t: np.exp(-t * t / 2) / np.sqrt(2 * np.pi)


def pders(y):
    t = y[:, None] - xs[None, :]
    e = ph(t) * ws[None, :]
    return e.sum(1), (-t * e).sum(1), ((t * t - 1) * e).sum(1)


# c* = exp(J(x_j)) at an interior atom, J(x) = int phi(y - x) log p(y) dy (mpmath)
XS = [mp.mpf(x) for x in xs]
WS = [mp.mpf(x) for x in ws]
pm = lambda y: mp.fsum(wi * mp.npdf(y - x) for x, wi in zip(XS, WS))
J = lambda x: mp.quad(lambda y: mp.npdf(y - x) * mp.log(pm(y)), mp.linspace(x - 12, x + 12, 49))
cst = float(mp.e ** J(XS[len(XS) // 2 + (0 if wc is None else 1)]))
p1m = lambda y: mp.fsum(-wi * (y - x) * mp.npdf(y - x) for x, wi in zip(XS, WS))
JpA = float(
    mp.quad(lambda y: mp.npdf(y - XS[-1]) * p1m(y) / pm(y), mp.linspace(XS[-1] - 12, XS[-1] + 12, 49))
)
lhs = we * JpA
# constants
P, ell, G2e = 7.7210, 0.31199, 9.354
ss = np.linspace(-12, 12, 240001)
fs = np.abs(ss * ss - 1) * ph(ss)
S2 = sum(fs[(ss >= k) & (ss <= k + 1)].max() for k in range(-12, 12))
Hbd = S2 * P / ell + (G2e / ell) ** 2
x0 = 0.0 if wc is None else u[0] / 2
R0 = A ** (1 / 3)
yy = np.linspace(-A, A, int(2 * A * 200) + 1)
p, p1, p2 = pders(yy)
g2 = p2 / p - (p1 / p) ** 2
Hact = np.abs(g2).max()
yi = np.linspace(x0 - R0 - 1, x0 + R0 + 1, 20001)
pi_, p1i, _ = pders(yi)
gI = np.log(pi_ / cst)
eps = np.abs(gI).max()
yo = np.linspace(x0 - R0, x0 + R0, 20001)
po, p1o, _ = pders(yo)
gpo = np.abs(p1o / po).max()
land = 2 * np.sqrt(eps * Hact)


# currents at x0, split inner [x0 - R0, x0] and tail (-inf, x0 - R0], right side; left side by mirror (computed directly)
def cur(lo, hi, right, n=40001):
    y = np.linspace(lo, hi, n)
    t = y[:, None] - xs[None, :]
    e = ph(t) * ws[None, :]
    sel = (xs > x0) if right else (xs < x0)
    pS = e[:, sel].sum(1)
    pp = e.sum(1)
    gp = (-t * e).sum(1) / pp
    fvals = pS * gp
    h = y[1] - y[0]
    return h * (fvals.sum() - (fvals[0] + fvals[-1]) / 2)


IRin = cur(x0 - R0, x0, True)
IRt = cur(x0 - A - 14, x0 - R0, True)
ILin = cur(x0, x0 + R0, False)
ILt = cur(x0 + R0, x0 + A + 14, False)
tt = np.linspace(R0, R0 + 40, 400001)
kk = np.arange(0, 60)
tailfun = (ph(tt[:, None] + kk[None, :]).sum(1)) * (G2e / ell + 2 * (tt + 1))
tail = 2 * P * (tt[1] - tt[0]) * (tailfun.sum() - (tailfun[0] + tailfun[-1]) / 2)
p0 = pders(np.array([x0]))[0][0]
rhs_ident = -p0 + (IRin + IRt) - (ILin + ILt)
bound = abs(p0 / cst - 1) + 4 * np.e * R0 * np.sqrt(eps * Hact) + tail
print(
    "A=%.4f K=%d x*=%.5f c*=%.12e p(x*)/c*-1=%.3e | w_A J'(A)=%.12e ident rhs=%.12e diff=%.1e"
    % (A, len(xs), x0, cst, p0 / cst - 1, lhs, rhs_ident, lhs - rhs_ident)
)
print("  S2=%.6f H_bd=%.2f H_act=%.3f %s" % (S2, Hbd, Hact, "OK" if Hact <= Hbd else "FAIL"))
print(
    "  R0=%.3f eps_g=%.3e sup|g'| on |y-x*|<=R0: %.3e  Landau 2 sqrt(eps H_act)=%.3e %s"
    % (R0, eps, gpo, land, "OK" if gpo <= land else "FAIL")
)
print(
    "  inner currents R, L: %.3e %.3e (per-side bound e R0 sup|g'| = %.3e) %s"
    % (
        IRin,
        ILin,
        np.e * R0 * gpo * cst,
        "OK" if max(abs(IRin), abs(ILin)) <= np.e * R0 * gpo * cst else "FAIL",
    )
)
print(
    "  tail currents R, L: %.3e %.3e (both-sides bound tail(R0) c* = %.3e) %s"
    % (IRt, ILt, tail * cst, "OK" if abs(IRt) + abs(ILt) <= tail * cst else "FAIL")
)
print(
    "  |w_A J'(A) + c*|/c* = %.3e  assembled bound = %.3e %s"
    % (abs(lhs + cst) / cst, bound, "OK" if abs(lhs + cst) / cst <= bound else "FAIL"),
    flush=True,
)
