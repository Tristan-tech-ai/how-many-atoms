"""q719 (BACKLOG 20, Amendment 1082): the D2-orbit wall-chain solve of q708/q712 in ball arithmetic (python-flint arb), with the
separable Gauss-Hermite rule of q718: for a query point q, Ex[i, j] = exp(-(qx + T_i - x_j)^2/2), Ey[j, l] = exp(-(qy + T_l - y_j)^2/2),
p = Ex diag(w) Ey / (2 pi); the gradient uses the same products with the factors (x_j - qx - T_i) and (y_j - qy - T_l).
Orbits in the ellipse parameter t: X = {0, pi}, Y = {pi/2, 3 pi/2}, G = {a, -a, pi - a, pi + a}. Unknowns: G angles, orbit masses, C.
Equations: D = C at one atom per orbit, zero tangential derivative at each G atom, masses sum to 1. Newton with a finite-difference
Jacobian (step 1e-14, 110-bit balls). Then D - C midway (in t) between neighbouring atoms in the first quadrant, and the maximum of
D - C on NSCAN points of [0, pi/2].
Usage: py q719_wallchain_arb.py RM X Y "angles_deg" "masses" [NGH] [NSCAN]   (rp = 1.3; masses in the order X, Y, G...)
"""

import sys, time, json
import mpmath as mp
from flint import arb, arb_mat, ctx
import d90_mpD as M90
import os

ctx.prec = int(os.environ.get("PREC", "110"))  # bits; FDH sets the finite-difference step
rp = arb("1.3")
rm = arb(sys.argv[1])
X = sys.argv[2] == "1"
Y = sys.argv[3] == "1"
pi = arb.pi()
G0 = [arb(t) * pi / 180 for t in sys.argv[4].split(",")] if sys.argv[4] else []
m0 = [arb(t) for t in sys.argv[5].split(",")]
NGH = int(sys.argv[6]) if len(sys.argv) > 6 else 200
NSCAN = int(sys.argv[7]) if len(sys.argv) > 7 else 360
T0 = time.time()
say = lambda *a: print(*a, flush=True)
t_mp, w_mp = M90.gh_nodes(NGH, dps=45)
s2 = arb(2).sqrt()
T = [arb(mp.nstr(x, 40)) * s2 for x in t_mp]
W = [arb(mp.nstr(x, 40)) / pi.sqrt() for x in w_mp]
hY = (2 * pi * arb(1).exp()).log()
L2P = (2 * pi).log()
pos = lambda t: (rp * t.cos(), rm * t.sin())
tang = lambda t: (-rp * t.sin(), rm * t.cos())
nG = len(G0)
nO = int(X) + int(Y) + nG


def atoms(G, m):
    th, w = [], []
    k = 0
    if X:
        th += [arb(0), pi]
        w += [m[k]] * 2
        k += 1
    if Y:
        th += [pi / 2, 3 * pi / 2]
        w += [m[k]] * 2
        k += 1
    for i, a in enumerate(G):
        th += [a, -a, pi - a, pi + a]
        w += [m[k + i]] * 4
    return th, w


def D_grad(q, xs, w, grad=True):
    K = len(xs)
    qx, qy = q
    ax = [[qx + T[i] - xs[j][0] for j in range(K)] for i in range(NGH)]
    ay = [[qy + T[l] - xs[j][1] for l in range(NGH)] for j in range(K)]
    Ex = arb_mat(NGH, K, [(-(ax[i][j] ** 2) / 2).exp() * w[j] for i in range(NGH) for j in range(K)])
    Ey = arb_mat(K, NGH, [(-(ay[j][l] ** 2) / 2).exp() for j in range(K) for l in range(NGH)])
    P = Ex * Ey
    if grad:
        Gx = arb_mat(NGH, K, [-ax[i][j] * Ex[i, j] for i in range(NGH) for j in range(K)]) * Ey
        Gy = Ex * arb_mat(K, NGH, [-ay[j][l] * Ey[j, l] for j in range(K) for l in range(NGH)])
    d = arb(0)
    gx = arb(0)
    gy = arb(0)
    for i in range(NGH):
        for l in range(NGH):
            p = P[i, l]
            ww = W[i] * W[l]
            d += ww * p.log()
            if grad:
                ip = ww / p
                gx += ip * Gx[i, l]
                gy += ip * Gy[i, l]
    D = -d + L2P - hY
    return (D, (-gx, -gy)) if grad else D


def resid(v):
    G = v[:nG]
    m = v[nG : nG + nO]
    C = v[nG + nO]
    th, w = atoms(G, m)
    xs = [pos(t) for t in th]
    reps = ([arb(0)] if X else []) + ([pi / 2] if Y else []) + list(G)
    out = []
    tg = []
    for k, t in enumerate(reps):
        if k >= nO - nG:
            D, g = D_grad(pos(t), xs, w)
            tv = tang(t)
            tg.append(g[0] * tv[0] + g[1] * tv[1])
        else:
            D = D_grad(pos(t), xs, w, grad=False)
        out.append(D - C)
    return out + tg + [sum(w, arb(0)) - 1]


s = sum(atoms(G0, m0)[1], arb(0))
m0 = [x / s for x in m0]
th, w = atoms(G0, m0)
xs = [pos(t) for t in th]
C0 = sum((wk * D_grad(pos(t), xs, w, grad=False) for t, wk in zip(th, w)), arb(0))
v = list(G0) + list(m0) + [C0]
h = arb(os.environ.get("FDH", "1e-10"))
for it in range(60):
    r = resid(v)
    nr = max(abs(float(x.mid())) for x in r)
    n2 = sum(float(x.mid()) ** 2 for x in r)
    say(f"iteration {it}: residual {nr:.3e}  [{time.time() - T0:.0f}s]")
    if nr < 1e-27:
        break
    J = arb_mat(len(r), len(v))
    for j in range(len(v)):
        vp = list(v)
        vp[j] = vp[j] + h
        rp_ = resid(vp)
        vm = list(v)
        vm[j] = vm[j] - h
        rm_ = resid(vm)  # central differences (1086):
        for i in range(len(r)):
            J[i, j] = (
                (rp_[i] - rm_[i]) / (2 * h)
            ).mid()  # forward ones left errors h*D'' above the smallest singular values
    dv = J.solve(arb_mat(len(r), 1, [-(x.mid()) for x in r]))
    lam = arb(1)  # damped: keep a step only if the residual falls
    for _ in range(14):
        vn = [(v[j] + lam * dv[j, 0]).mid() for j in range(len(v))]
        nn = sum(float(x.mid()) ** 2 for x in resid(vn))  # 1102: sum of squares
        if nn < n2:
            break  # (a Newton step always lowers it; the max norm need not)
        lam = lam / 2
    if nn >= n2:
        say("damped step found no decrease; stopping Newton")
        break
    v = vn
rf = resid(v)
say(
    f"final residual blocks: D - C {[f'{float(x.mid()):+.2e}' for x in rf[:nO]]}; tangential {[f'{float(x.mid()):+.2e}' for x in rf[nO:nO + nG]]}; mass {float(rf[-1].mid()):+.2e}"
)
G = v[:nG]
m = v[nG : nG + nO]
C = v[nG + nO]
th, w = atoms(G, m)
xs = [pos(t) for t in th]
ths = sorted([float(t.mid()) % (2 * float(pi.mid())) for t in th])
say(
    f"rm {sys.argv[1]}: K {len(th)}; C {C.str(25, radius=False)}; G deg {[float((a*180/pi).mid()) for a in G]}; masses {[float(x.mid()) for x in m]}"
)
mids = []
full = sorted(ths)
full = full + [full[0] + 2 * float(pi.mid())]
for a, b in zip(full[:-1], full[1:]):
    c = (a + b) / 2
    if c <= float((pi / 2).mid()) + 1e-9:
        mids.append(c)
marg = [(c * 180 / float(pi.mid()), D_grad(pos(arb(c)), xs, w, grad=False) - C) for c in mids]
say("margins midway (deg, D - C): " + "; ".join(f"{d:.4f} {float(x.mid()):+.6e}" for d, x in marg))
best = None
for i in range(NSCAN + 1):
    t = (pi / 2) * i / NSCAN
    e = D_grad(pos(t), xs, w, grad=False) - C
    if best is None or e > best[1]:
        best = (float((t * 180 / pi).mid()), e)
say(
    f"max(D - C) on {NSCAN + 1} points of [0, 90] deg: {float(best[1].mid()):+.6e} at {best[0]:.3f} deg  [{time.time() - T0:.0f}s]"
)
json.dump(
    {
        "rm": sys.argv[1],
        "X": int(X),
        "Y": int(Y),
        "G": [g.str(30, radius=False) for g in G],
        "m": [x.str(30, radius=False) for x in m],
        "C": C.str(30, radius=False),
        "margins": [[d, float(x.mid())] for d, x in marg],
        "scanmax": [best[0], float(best[1].mid())],
    },
    open(f"q719_rm{sys.argv[1]}_X{int(X)}Y{int(Y)}G{nG}.json", "w"),
    indent=1,
)
