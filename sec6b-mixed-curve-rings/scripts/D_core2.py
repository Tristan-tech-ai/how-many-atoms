"""D_core2 (02 Oct 2026): D_core generalised to three KKT functions and the mixed curve. All KKT functions are copied as
text from the record's solvers (none is imported), rewritten with matrix-vector products over the separable Gauss-Hermite grid
(same sums as the originals):
- FUNC lfp: least favourable prior, r(theta) = E|delta(theta + Z) - theta|^2 (q786; identical to D_core.LFP).
- FUNC cap: capacity, D(q) = -E log p(q + Z) - log(2 pi e) with p the output density (q719 / q781 / q805):
  D = -sum W_i W_l log P_il + log(2 pi) - log(2 pi e), P = Ex Ey, Ex_ij = exp(-(qx + T_i - x_j)^2/2) w_j, Ey_jl = exp(-(qy + T_l - y_j)^2/2);
  gradient (-gx, -gy) with gx = sum W W Gx/P, Gx = (-ax o Ex) Ey, ax_ij = qx + T_i - x_j (and the same in y).
- FUNC rd: NPMLE / rate-distortion dual with source N(0, s^2 I), env SRC_S (q789 / q801): nodes YN = s T,
  D(q) = sum W_i W_l fx_i fy_l / P_il, fx_i = exp(-(YN_i - qx)^2/2), P = Ex Ey with Ex_ij = exp(-(YN_i - x_j)^2/2) w_j (P does not
  depend on q and is cached per atom set); gradient sum W W fx fy (YN - q)/P.
Geometry (q805): z = R e^(it) + eta (e^(-it) + LAM e^(3it)), e = rp - rm, R = rp - e/2, eta = e/(2 (1 + LAM)), i.e.
pos = ((R + eta) cos t + LAM eta cos 3t, (R - eta) sin t + LAM eta sin 3t); LAM = 0 is the ellipse (rp cos t, rm sin t), used verbatim.
Unknowns v = [G angles] + [orbit masses X, Y, G...] + [C]; equations as q786 (D - C at one atom per orbit, tangential derivative at
each G atom, masses sum to 1). No top-level computation; the only file written is the optional node cache (env GHCACHE).
"""

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


def gh_cached(n, dps):
    """gh_nodes, read from / written to env GHCACHE (a folder) when set: the large-n node sets are slow to refine."""
    import os, json

    d = os.environ.get("GHCACHE")
    if not d:
        return gh_nodes(n, dps=dps)
    f = os.path.join(d, f"gh_{n}_{dps}.json")
    if os.path.exists(f):
        js = json.load(open(f))
        mp.mp.dps = dps
        return [mp.mpf(x) for x in js["t"]], [mp.mpf(x) for x in js["w"]]
    t, w = gh_nodes(n, dps=dps)
    os.makedirs(d, exist_ok=True)
    tmp = f + f".{os.getpid()}.tmp"
    json.dump({"t": [mp.nstr(x, dps) for x in t], "w": [mp.nstr(x, dps) for x in w]}, open(tmp, "w"))
    os.replace(tmp, f)
    return t, w


def gl_nodes(n, dps):  # copied from D_quad2.gl_nodes (no import: D_quad2 runs on import)
    mp.mp.dps = dps
    x0, _ = np.polynomial.legendre.leggauss(n)
    xs, ws = [], []
    for g in x0:
        x = mp.mpf(g)
        for _ in range(100):
            p = mp.legendre(n, x)
            dp = n * (x * p - mp.legendre(n - 1, x)) / (x * x - 1)
            dx = p / dp
            x -= dx
            if abs(dx) < mp.mpf(10) ** (-dps + 3):
                break
        dp = n * (x * mp.legendre(n, x) - mp.legendre(n - 1, x)) / (x * x - 1)
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


class Ring:
    def __init__(self, func, rp, X, Y, nG, NGH=200, prec=160, lam="0", src_s="1", ghdps=None):
        ctx.prec = prec
        self.prec = prec
        self.func = func
        self.rp = arb(rp)
        self.X = bool(X)
        self.Y = bool(Y)
        self.nG = nG
        self.N = NGH
        self.nO = int(self.X) + int(self.Y) + nG
        self.pi = arb.pi()
        self.lam = arb(lam)
        self.lam0 = float(lam) == 0.0
        ghdps = ghdps or (45 if prec <= 200 else int(prec * 0.30103) + 10)
        nd = ghdps - 5  # the record's rule (45 / 40 digits) up to 200 bits
        t_mp, w_mp = gh_cached(NGH, ghdps)
        s2 = arb(2).sqrt()
        self.T = [arb(mp.nstr(x, nd)) * s2 for x in t_mp]
        self.W = [arb(mp.nstr(x, nd)) / self.pi.sqrt() for x in w_mp]
        self.Wv = arb_mat(NGH, 1, self.W)
        self.WTv = arb_mat(NGH, 1, [a * b for a, b in zip(self.W, self.T)])
        self.hY = (2 * self.pi * arb(1).exp()).log()
        self.L2P = (2 * self.pi).log()
        self.SS = arb(src_s)
        self.YN = [self.SS * t for t in self.T]
        self._cache = None
        if func == "capfix":  # 20:5x: fixed-grid capacity rule (see _D_capfix)
            import os

            n = int(os.environ.get("FIX_N", "40"))
            B = int(os.environ.get("FIX_B", "20"))
            H = int(os.environ.get("FIX_H", "1"))
            dps = int(prec * 0.30103) + 10
            xs, ws = gl_nodes(n, dps)
            mp.mp.dps = dps
            Yp, Vp = [], []
            for k in range(B // H):
                a = mp.mpf(k * H)
                b = a + H
                for x, wt in zip(xs, ws):
                    Yp.append((a + b) / 2 + (b - a) / 2 * x)
                    Vp.append((b - a) / 2 * wt)
            self.Yp = [arb(mp.nstr(y, dps - 3)) for y in Yp]
            self.Vp = [arb(mp.nstr(v, dps - 3)) for v in Vp]
            self.Np = len(Yp)
            self.c1 = 1 / (2 * self.pi).sqrt()

    # ---- geometry
    def pos(self, t, rm):
        if self.lam0:
            return (self.rp * t.cos(), rm * t.sin())
        e = self.rp - rm
        R = self.rp - e / 2
        et = e / (2 * (1 + self.lam))
        L = self.lam
        return ((R + et) * t.cos() + L * et * (3 * t).cos(), (R - et) * t.sin() + L * et * (3 * t).sin())

    def tang(self, t, rm):
        if self.lam0:
            return (-self.rp * t.sin(), rm * t.cos())
        e = self.rp - rm
        R = self.rp - e / 2
        et = e / (2 * (1 + self.lam))
        L = self.lam
        return (
            -(R + et) * t.sin() - 3 * L * et * (3 * t).sin(),
            (R - et) * t.cos() + 3 * L * et * (3 * t).cos(),
        )

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

    # ---- KKT functions
    def D_grad(self, q, xs, w, grad=True):
        return getattr(self, "_D_" + self.func)(q, xs, w, grad)

    def _D_lfp(self, q, xs, w, grad):  # q786 (as D_core.LFP.D_grad)
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

    def _D_cap(self, q, xs, w, grad):  # q719 / q781 / q805
        N = self.N
        T = self.T
        K = len(xs)
        qx, qy = q
        ax = [[qx + T[i] - xs[j][0] for j in range(K)] for i in range(N)]
        ay = [[qy + T[l] - xs[j][1] for l in range(N)] for j in range(K)]
        exl = [(-(ax[i][j] ** 2) / 2).exp() * w[j] for i in range(N) for j in range(K)]
        eyl = [(-(ay[j][l] ** 2) / 2).exp() for j in range(K) for l in range(N)]
        Ex = arb_mat(N, K, exl)
        Ey = arb_mat(K, N, eyl)
        P = (Ex * Ey).entries()
        Lm = arb_mat(N, N, [p.log() for p in P])
        D = -(self.Wv.transpose() * (Lm * self.Wv))[0, 0] + self.L2P - self.hY
        if not grad:
            return D
        Gx = (arb_mat(N, K, [-ax[i][j] * exl[i * K + j] for i in range(N) for j in range(K)]) * Ey).entries()
        Gy = (Ex * arb_mat(K, N, [-ay[j][l] * eyl[j * N + l] for j in range(K) for l in range(N)])).entries()
        RX = arb_mat(N, N, [g / p for g, p in zip(Gx, P)])
        RY = arb_mat(N, N, [g / p for g, p in zip(Gy, P)])
        gx = (self.Wv.transpose() * (RX * self.Wv))[0, 0]
        gy = (self.Wv.transpose() * (RY * self.Wv))[0, 0]
        return D, (-gx, -gy)

    def _D_capfix(self, q, xs, w, grad):
        """capacity D on a FIXED grid: E log P(q + Z) = int phi(y - q) log P(y) dy by composite Gauss-Legendre (panels of width FIX_H,
        FIX_N points, box [-FIX_B, FIX_B]^2), P(y) = sum_j w_j exp(-|y - x_j|^2/2) as in q719. log P on the grid is computed once per
        atom set and folded to the positive quadrant by the ring's D2 symmetry (atoms in orbits, so P(+-y1, +-y2) = P(y1, y2)); a query
        costs matrix-vector products. Gradient: int (y - q) phi(y - q) log P(y) dy (the record's score form after integration by parts).
        """
        Np = self.Np
        Y = self.Yp
        K = len(xs)
        qx, qy = q
        if self._cache is None or self._cache[0] is not xs or self._cache[1] is not w:
            Ex = arb_mat(
                Np, K, [(-((Y[i] - xs[j][0]) ** 2) / 2).exp() * w[j] for i in range(Np) for j in range(K)]
            )
            Ey = arb_mat(K, Np, [(-((Y[l] - xs[j][1]) ** 2) / 2).exp() for j in range(K) for l in range(Np)])
            Lm = arb_mat(Np, Np, [p.log() for p in (Ex * Ey).entries()])
            self._cache = (xs, w, Lm)
        Lm = self._cache[2]
        c = self.c1
        e = lambda y, s: (-((y - s) ** 2) / 2).exp()
        a = [V * c * (e(y, qx) + e(-y, qx)) for V, y in zip(self.Vp, Y)]
        b = [V * c * (e(y, qy) + e(-y, qy)) for V, y in zip(self.Vp, Y)]
        Lb = Lm * arb_mat(Np, 1, b)
        E = sum((ai * Lb[i, 0] for i, ai in enumerate(a)), arb(0))
        D = -E + self.L2P - self.hY
        if not grad:
            return D
        ag = [V * c * ((y - qx) * e(y, qx) + (-y - qx) * e(-y, qx)) for V, y in zip(self.Vp, Y)]
        bg = [V * c * ((y - qy) * e(y, qy) + (-y - qy) * e(-y, qy)) for V, y in zip(self.Vp, Y)]
        gx = sum((ai * Lb[i, 0] for i, ai in enumerate(ag)), arb(0))
        Lbg = Lm * arb_mat(Np, 1, bg)
        gy = sum((ai * Lbg[i, 0] for i, ai in enumerate(a)), arb(0))
        return D, (-gx, -gy)

    def _D_rd(self, q, xs, w, grad):  # q789 / q801
        N = self.N
        YN = self.YN
        K = len(xs)
        qx, qy = q
        if self._cache is None or self._cache[0] is not xs or self._cache[1] is not w:
            Ex = arb_mat(
                N, K, [(-((YN[i] - xs[j][0]) ** 2) / 2).exp() * w[j] for i in range(N) for j in range(K)]
            )
            Ey = arb_mat(K, N, [(-((YN[l] - xs[j][1]) ** 2) / 2).exp() for j in range(K) for l in range(N)])
            Q = arb_mat(N, N, [1 / p for p in (Ex * Ey).entries()])
            self._cache = (xs, w, Q)
        Q = self._cache[2]
        a = [W * (-((y - qx) ** 2) / 2).exp() for W, y in zip(self.W, YN)]
        b = [W * (-((y - qy) ** 2) / 2).exp() for W, y in zip(self.W, YN)]
        Qb = Q * arb_mat(N, 1, b)
        d = sum((ai * Qb[i, 0] for i, ai in enumerate(a)), arb(0))
        if not grad:
            return d
        gx = sum((ai * (YN[i] - qx) * Qb[i, 0] for i, ai in enumerate(a)), arb(0))
        Qby = Q * arb_mat(N, 1, [bl * (YN[l] - qy) for l, bl in enumerate(b)])
        gy = sum((ai * Qby[i, 0] for i, ai in enumerate(a)), arb(0))
        return d, (gx, gy)

    # ---- system
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

    def kkt_many(self, v, rm, ts):
        """D - C at the parameters ts (arb or float radians), one atom set."""
        nG, nO = self.nG, self.nO
        G = v[:nG]
        m = v[nG : nG + nO]
        C = v[nG + nO]
        th, w = self.atoms(G, m)
        xs = [self.pos(a, rm) for a in th]
        return [self.D_grad(self.pos(arb(t), rm), xs, w, grad=False) - C for t in ts]

    def kkt(self, v, rm, t):
        return self.kkt_many(v, rm, [t])[0]

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
        ts = [(self.pi / 2) * i / nscan for i in range(nscan + 1)]
        vals = self.kkt_many(v, rm, [arb(c) for c in mids] + ts)
        marg = [(c * 180 / pi, x) for c, x in zip(mids, vals[: len(mids)])]
        best = None
        for t, e in zip(ts, vals[len(mids) :]):
            if best is None or e > best[1]:
                best = (float((t * 180 / self.pi).mid()), e)
        return marg, best
