"""d90_mpD (BACKLOG 17, Amendment 1073): the KKT function D of d89 in mpmath, for checks below double precision. Gauss-Hermite nodes
by Newton refinement of numpy's nodes on H_n in mp; weights 2^(n-1) n! sqrt(pi) / (n^2 H_(n-1)(x_i)^2). D(x) = -E log p(x + Z) -
log(2 pi e), p(y) = sum_k w_k exp(-|y - x_k|^2/2)/(2 pi), tensor rule over Z ~ N(0, I_2). Definitions only."""

import numpy as np
import mpmath as mp


def gh_nodes(n, dps=40):
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


class MPD:
    def __init__(self, n=80, dps=40):
        self.dps = dps
        t, w = gh_nodes(n, dps)
        s2 = mp.sqrt(2)
        self.Z = [(s2 * ti, s2 * tj, wi * wj / mp.pi) for ti, wi in zip(t, w) for tj, wj in zip(t, w)]
        self.hY = mp.log(2 * mp.pi * mp.e)

    def D(self, xq, xs, w):
        """D at the query point xq = (x, y) for atoms xs (list of (x, y)) with masses w (mp numbers)."""
        mp.mp.dps = self.dps
        tot = mp.mpf(0)
        qx, qy = mp.mpf(xq[0]), mp.mpf(xq[1])
        for zx, zy, wz in self.Z:
            yx, yy = qx + zx, qy + zy
            p = mp.fsum(
                wk * mp.exp(-((yx - ax) ** 2 + (yy - ay) ** 2) / 2) for (ax, ay), wk in zip(xs, w)
            ) / (2 * mp.pi)
            tot += wz * mp.log(p)
        return -tot - self.hY
