"""D_core (02 Oct 2026): the least-favourable-prior KKT system of q786_lfp_wallchain.py, copied as text into importable
functions (no top-level computation, no file writes). Same quadrature (separable Gauss-Hermite, nodes by Newton refinement as in
d90_mpD.gh_nodes), same Bayes risk r(theta) = E|delta(theta + Z) - theta|^2 and score-form gradient E[Z h] - 2 E[delta - theta],
same unknowns v = [G angles (rad)] + [orbit masses X, Y, G...] + [C] and the same equations (r - C at one atom per orbit, tangential
derivative at each G atom, masses sum to 1). Changes: rm is an argument (for continuation in rm); the double loop over the
quadrature grid is rewritten with matrix-vector products (same sums, fewer Python-level operations)."""

import numpy as np
import mpmath as mp
from flint import arb, arb_mat, ctx


def gh_nodes(n, dps=40):  # copied from d90_mpD.gh_nodes
    mp.mp.dps = dps
    t0, _ = np.polynomial.hermite.hermgauss(n)
    xs, ws = [], []
    for x0 in t0:
        x = mp.mpf(x0)
        for _ in range(60):
            h = mp.hermite(n, x)
            dh = 2 * n * mp.hermite(n - 1, x)
            dx = h / dh
            x -= dx
            if abs(dx) < mp.mpf(10) ** (-dps + 5):
                break
        xs.append(x)
        ws.append(2 ** (n - 1) * mp.factorial(n) * mp.sqrt(mp.pi) / (n**2 * mp.hermite(n - 1, x) ** 2))
    return xs, ws


class LFP:
    def __init__(self, rp, X, Y, nG, NGH=200, prec=160, ghdps=45):
        ctx.prec = prec
        self.prec = prec
        self.rp = arb(rp)
        self.X = bool(X)
        self.Y = bool(Y)
        self.nG = nG
        self.N = NGH
        self.nO = int(self.X) + int(self.Y) + nG
        self.pi = arb.pi()
        t_mp, w_mp = gh_nodes(NGH, dps=ghdps)
        s2 = arb(2).sqrt()
        self.T = [arb(mp.nstr(x, 40)) * s2 for x in t_mp]
        self.W = [arb(mp.nstr(x, 40)) / self.pi.sqrt() for x in w_mp]
        self.Wv = arb_mat(NGH, 1, self.W)
        self.WTv = arb_mat(NGH, 1, [a * b for a, b in zip(self.W, self.T)])
        self.WW = sum(self.W, arb(0)) ** 2

    def pos(self, t, rm):
        return (self.rp * t.cos(), rm * t.sin())

    def tang(self, t, rm):
        return (-self.rp * t.sin(), rm * t.cos())

    def atoms(self, G, m):
        pi = self.pi
        th, w = [], []
        k = 0
        if self.X:
            th += [arb(0), pi]
            w += [m[k]] * 2
            k += 1
        if self.Y:
            th += [pi / 2, 3 * pi / 2]
            w += [m[k]] * 2
            k += 1
        for i, a in enumerate(G):
            th += [a, -a, pi - a, pi + a]
            w += [m[k + i]] * 4
        return th, w

    def D_grad(self, q, xs, w, grad=True):
        N = self.N
        T = self.T
        K = len(xs)
        qx, qy = q
        Ex = arb_mat(
            N, K, [(-((qx + T[i] - xs[j][0]) ** 2) / 2).exp() * w[j] for i in range(N) for j in range(K)]
        )
        Ey = arb_mat(K, N, [(-((qy + T[l] - xs[j][1]) ** 2) / 2).exp() for j in range(K) for l in range(N)])
        P = (Ex * Ey).entries()
        NX = (
            Ex * arb_mat(K, K, [xs[j][0] if j == k else 0 for j in range(K) for k in range(K)]) * Ey
        ).entries()
        NY = (
            Ex * (arb_mat(K, K, [xs[j][1] if j == k else 0 for j in range(K) for k in range(K)]) * Ey)
        ).entries()
        ex = [nx / p - qx for nx, p in zip(NX, P)]
        ey = [ny / p - qy for ny, p in zip(NY, P)]
        H = arb_mat(N, N, [a * a + b * b for a, b in zip(ex, ey)])
        HW = H * self.Wv
        r = (self.Wv.transpose() * HW)[0, 0]
        if not grad:
            return r
        EX = arb_mat(N, N, ex)
        EY = arb_mat(N, N, ey)
        gx = (self.WTv.transpose() * HW)[0, 0] - 2 * (self.Wv.transpose() * (EX * self.Wv))[0, 0]
        gy = (self.Wv.transpose() * (H * self.WTv))[0, 0] - 2 * (self.Wv.transpose() * (EY * self.Wv))[0, 0]
        return r, (gx, gy)

    def reps(self, G):
        return ([arb(0)] if self.X else []) + ([self.pi / 2] if self.Y else []) + list(G)

    def resid(self, v, rm):
        nG, nO = self.nG, self.nO
        G = v[:nG]
        m = v[nG : nG + nO]
        C = v[nG + nO]
        th, w = self.atoms(G, m)
        xs = [self.pos(t, rm) for t in th]
        out = []
        tg = []
        for k, t in enumerate(self.reps(G)):
            if k >= nO - nG:
                D, g = self.D_grad(self.pos(t, rm), xs, w)
                tv = self.tang(t, rm)
                tg.append(g[0] * tv[0] + g[1] * tv[1])
            else:
                D = self.D_grad(self.pos(t, rm), xs, w, grad=False)
            out.append(D - C)
        return out + tg + [sum(w, arb(0)) - 1]

    def kkt(self, v, rm, t):
        """D - C at ellipse parameter t (arb or float, radians)."""
        nG, nO = self.nG, self.nO
        G = v[:nG]
        m = v[nG : nG + nO]
        C = v[nG + nO]
        th, w = self.atoms(G, m)
        xs = [self.pos(a, rm) for a in th]
        return self.D_grad(self.pos(arb(t), rm), xs, w, grad=False) - C

    def scan(self, v, rm, nscan=90):
        """margins midway between neighbouring atoms in [0, pi/2] and the maximum of D - C on nscan + 1 points of [0, pi/2]."""
        nG, nO = self.nG, self.nO
        G = v[:nG]
        m = v[nG : nG + nO]
        th, w = self.atoms(G, m)
        pi = float(self.pi.mid())
        full = sorted(float(t.mid()) % (2 * pi) for t in th)
        full = full + [full[0] + 2 * pi]
        mids = [(a + b) / 2 for a, b in zip(full[:-1], full[1:]) if (a + b) / 2 <= pi / 2 + 1e-9]
        marg = [(c * 180 / pi, self.kkt(v, rm, c)) for c in mids]
        best = None
        for i in range(nscan + 1):
            t = (self.pi / 2) * i / nscan
            e = self.kkt(v, rm, t)
            if best is None or e > best[1]:
                best = (float((t * 180 / self.pi).mid()), e)
        return marg, best


def load_state(f):
    import json

    d = json.load(open(f))
    return d


def state_vec(d, prec=160):
    ctx.prec = prec
    return [arb(g) for g in d["G"]] + [arb(x) for x in d["m"]] + [arb(d["C"])]
