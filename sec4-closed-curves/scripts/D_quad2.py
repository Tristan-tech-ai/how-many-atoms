"""D_quad2 (02 Oct): a second quadrature for the capacity KKT function, independent of tensor Gauss-Hermite.
E f(Z), Z ~ N(0, I_2), by a tensor rule whose 1D factor is composite Gauss-Legendre on panels of width H covering [-L + OFF, L + OFF]
(n points per panel, weights times the normal density e^(-t^2/2)/sqrt(2 pi)); the Gaussian tails beyond the box are dropped (mass
2 Phi(-L) ~ 4e-73 at L 18; |log p| there is ~ L^2/2, so the dropped part is ~1e-70). Legendre nodes by Newton refinement in mpmath.
The KKT function is D_core2's capacity D (text copy of q719) with this rule in place of the Gauss-Hermite nodes.
Modes:  py D_quad2.py ring R K1,K2   -> regular K-ring ripple D(mid) - D(atom) on the circle R (check against Gauss-Hermite)
        py D_quad2.py state STATE_JSON "t1,t2,..." (deg)   -> D - C at the given t and at the orbit atoms, with both rules
Env: PREC (320), GL_N (40), GL_H (1), GL_L (18), GL_OFF (0), NGH (640, the Gauss-Hermite comparison), GHCACHE, RP (1.0).
"""

import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mpmath as mp
from flint import arb, arb_mat, ctx
from D_core2 import Ring

PREC = int(os.environ.get("PREC", "320"))
GLN = int(os.environ.get("GL_N", "40"))
GLH = os.environ.get("GL_H", "1")
GLL = os.environ.get("GL_L", "18")
OFF = os.environ.get("GL_OFF", "0")
NGH = int(os.environ.get("NGH", "640"))
RP = os.environ.get("RP", "1.0")
T0 = time.time()
say = lambda *a: print(*a, f"[{time.time() - T0:.0f}s]", flush=True)
f2 = lambda x: float(x.mid())
DPS = int(PREC * 0.30103) + 10


def gl_nodes(n):
    mp.mp.dps = DPS
    x0, _ = np.polynomial.legendre.leggauss(n)
    xs, ws = [], []
    for g in x0:
        x = mp.mpf(g)
        for _ in range(100):
            p = mp.legendre(n, x)
            dp = n * (x * p - mp.legendre(n - 1, x)) / (x * x - 1)
            dx = p / dp
            x -= dx
            if abs(dx) < mp.mpf(10) ** (-DPS + 3):
                break
        dp = n * (x * mp.legendre(n, x) - mp.legendre(n - 1, x)) / (x * x - 1)
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


def gl_rule():
    xs, ws = gl_nodes(GLN)
    mp.mp.dps = DPS
    L = mp.mpf(GLL)
    H = mp.mpf(GLH)
    off = mp.mpf(OFF)
    npan = int(mp.nint(2 * L / H))
    T, W = [], []
    for k in range(npan):
        a = -L + off + k * H
        b = a + H
        for x, w in zip(xs, ws):
            t = (a + b) / 2 + (b - a) / 2 * x
            T.append(t)
            W.append((b - a) / 2 * w * mp.exp(-t * t / 2) / mp.sqrt(2 * mp.pi))
    return T, W


def install(P, T, W):
    """replace the Gauss-Hermite rule of a D_core2.Ring by (T, W) (mpmath lists); weights already include the density."""
    P.T = [arb(mp.nstr(t, DPS - 3)) for t in T]
    P.W = [arb(mp.nstr(w, DPS - 3)) for w in W]
    P.N = len(T)
    P.Wv = arb_mat(P.N, 1, P.W)
    P.WTv = arb_mat(P.N, 1, [a * b for a, b in zip(P.W, P.T)])


T2, W2 = gl_rule()
mp.mp.dps = DPS
say(
    f"Gauss-Legendre panel rule: {len(T2)} nodes per axis (n {GLN}, H {GLH}, L {GLL}, offset {OFF}); sum of 1D weights - 1 = "
    f"{mp.nstr(mp.fsum(W2) - 1, 5)}; Gauss-Hermite comparison: {NGH} nodes; {PREC} bits"
)
mode = sys.argv[1]
if mode == "ring":
    R = sys.argv[2]
    for K in [int(k) for k in sys.argv[3].split(",")]:
        out = []
        for rule in ("GL", "GH"):
            P = Ring("cap", R, 0, 0, 0, NGH=NGH, prec=PREC)
            if rule == "GL":
                install(P, T2, W2)
            pi = P.pi
            Ra = arb(R)
            th = [2 * pi * k / K for k in range(K)]
            xs = [(Ra * t.cos(), Ra * t.sin()) for t in th]
            w = [arb(1) / K] * K
            da = P.D_grad((Ra, arb(0)), xs, w, grad=False)
            a = pi / K
            dm = P.D_grad((Ra * a.cos(), Ra * a.sin()), xs, w, grad=False)
            out.append((rule, dm - da, da))
        say(
            f"K {K}, circle R {R}: ripple D(mid) - D(atom): GL {f2(out[0][1]):+.10e}, GH {f2(out[1][1]):+.10e}, difference {f2(out[0][1] - out[1][1]):+.3e}; "
            f"D(atom) GL - GH {f2(out[0][2] - out[1][2]):+.3e}"
        )
else:
    d = json.load(open(sys.argv[2]))
    ts = [x for x in sys.argv[3].split(",")]
    res = {}
    for rule in ("GL", "GH"):
        P = Ring("cap", RP, d["X"], d["Y"], len(d["G"]), NGH=NGH, prec=PREC)
        if rule == "GL":
            install(P, T2, W2)
        rm = arb(d["rm"])
        pi = P.pi
        G = [arb(g) for g in d["G"]]
        m = [arb(x) for x in d["m"]]
        C1 = arb(d["C"])
        th, w = P.atoms(G, m)
        xs = [P.pos(t, rm) for t in th]
        reps = P.reps(G)
        mult = ([2] if P.X else []) + ([2] if P.Y else []) + [4] * len(G)
        Dat = [P.D_grad(P.pos(t, rm), xs, w, grad=False) for t in reps]
        C = sum((mu * mm * x for mu, mm, x in zip(mult, m, Dat)), arb(0))
        pts = []
        for s in ts:
            if s.startswith("mid"):
                i, j = [int(x) for x in s[3:].split("-")]
                pts.append((s, (G[i] + G[j]) / 2))
            elif s.startswith("half"):
                i = int(s[4:])
                pts.append((s, G[i] / 2))
            else:
                pts.append((s, pi * arb(s) / 180))
        Dq = [P.D_grad(P.pos(t, rm), xs, w, grad=False) for _, t in pts]
        res[rule] = (C, Dat, Dq, pts)
        say(
            f"{rule}: C (mass-weighted D over the atoms) - C_state {f2(C - C1):+.4e}; atoms D - C: "
            + ", ".join(f"{f2(x - C):+.3e}" for x in Dat)
        )
        say(
            f"{rule}: "
            + "; ".join(
                f"t {f2(t*180/pi):.6f} deg ({s}): D - C {f2(x - C):+.10e}, D - C_state {f2(x - C1):+.10e}"
                for (s, t), x in zip(pts, Dq)
            )
        )
    (Cl, _, Ql, pts), (Ch, _, Qh, _) = res["GL"], res["GH"]
    for (s, t), a, b in zip(pts, Ql, Qh):
        say(
            f"t {s}: D - C  GL {f2(a - Cl):+.10e}  GH {f2(b - Ch):+.10e}  difference {f2((a - Cl) - (b - Ch)):+.3e}"
        )
