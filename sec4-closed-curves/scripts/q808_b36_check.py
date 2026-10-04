"""q808 (Amendment 1329): my check of the flank 32-ring (BACKLOG 36) with the record's capacity function (q719's D_grad, copied
as text from q802), not D's code. Reads a D-format state (G in radians, orbit masses, C), evaluates D - C at the major vertex, at one
atom per orbit and at the midpoints between neighbouring atoms in [0, pi/2], and at extra angles given on the command line.
Usage: py q808_b36_check.py STATE.json NGH PREC [extra angles deg, comma-separated]"""

import sys, json, time
import mpmath as mp
from flint import arb, arb_mat, ctx

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import d90_mpD as M90

d = json.load(open(sys.argv[1]))
NGH = int(sys.argv[2])
ctx.prec = int(sys.argv[3])
dps = int(int(sys.argv[3]) * 0.30103) + 10
pi = arb.pi()
rp = arb(d.get("rp", "1.0"))
rm = arb(d["rm"])
T0 = time.time()
t_mp, w_mp = M90.gh_nodes(NGH, dps=dps)
s2 = arb(2).sqrt()
T = [arb(mp.nstr(x, dps - 5)) * s2 for x in t_mp]
W = [arb(mp.nstr(x, dps - 5)) / pi.sqrt() for x in w_mp]
hY = (2 * pi * arb(1).exp()).log()
L2P = (2 * pi).log()
pos = lambda t: (rp * t.cos(), rm * t.sin())


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


X = str(d["X"]) in ("1", "True")
Y = str(d["Y"]) in ("1", "True")
G = [arb(g) for g in d["G"]]
m = [arb(x) for x in d["m"]]
C = arb(d["C"])
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
xs = [pos(t) for t in th]
print(
    f"q808: rp {rp.str(6, radius=False)}, rm {rm.str(12, radius=False)}, K {len(th)}, NGH {NGH}, prec {ctx.prec}, sum w {float(sum(w, arb(0)).mid()):.15f}",
    flush=True,
)
reps = sorted([float(a.mid()) for a in G] + ([0.0] if X else []) + ([float((pi / 2).mid())] if Y else []))
full = [0.0] + reps + [float((pi / 2).mid())]
mids = sorted(set([(a + b) / 2 for a, b in zip(full[:-1], full[1:]) if b - a > 1e-9]))
pts = [("vertex", arb(0))] + [
    (f"atom {r*180/float(pi.mid()):.4f}", arb(repr(r)) if False else None) for r in []
]
for r in reps:
    pts.append((f"atom {r*180/float(pi.mid()):.4f}", None))
for c in mids:
    pts.append((f"mid {c*180/float(pi.mid()):.4f}", arb(repr(c))))
for e in (sys.argv[4].split(",") if len(sys.argv) > 4 else []):
    pts.append((f"extra {e}", arb(e) * pi / 180))
for name, t in pts:
    if t is None:  # atom: use the exact arb angle from the state
        deg = float(name.split()[1])
        t = min([arb(0)] + G + [pi / 2], key=lambda a: abs(float((a * 180 / pi).mid()) - deg))
    v = D_grad(pos(t), xs, w, grad=False) - C
    print(f"  {name:18s} D - C {float(v.mid()):+.6e}  [{time.time() - T0:.0f}s]", flush=True)
