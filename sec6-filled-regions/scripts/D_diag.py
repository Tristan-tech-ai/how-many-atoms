"""D_diag: step 1, diagnosis at the converged eps 0.1723 twelve-ring (lfp10/t6, rm 0.827708514103).
Jacobian of the q786 system by central differences at two steps (Richardson check), its SVD in mpmath, the soft direction read in
ring harmonics b_j = 2 sum_k m_k cos(j t_k), the branch tangent dv/drm = -J^-1 F_rm against the secant from rm 0.826583514103.
Writes D_work/diag_jac.json (J, F_rm, singular values and vectors as strings). Usage: py D_diag.py [PREC] [NGH]
"""

import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp
from flint import arb, arb_mat, ctx
from D_core import LFP, state_vec

PREC = int(sys.argv[1]) if len(sys.argv) > 1 else 160
NGH = int(sys.argv[2]) if len(sys.argv) > 2 else 200
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "centre_scripts", "lfp10", "t6")
say = lambda *a: print(*a, flush=True)
T0 = time.time()
P = LFP("1.0", 1, 1, 2, NGH=NGH, prec=PREC)
dA = json.load(open(os.path.join(SRC, "q719_rm0.827708514103_X1Y1G2.json")))
dB = json.load(open(os.path.join(SRC, "q719_rm0.826583514103_X1Y1G2.json")))
vA = state_vec(dA, PREC)
vB = state_vec(dB, PREC)
rmA = arb(dA["rm"])
rmB = arb(dB["rm"])
n = len(vA)
names = ["G1", "G2", "mX", "mY", "m1", "m2", "C"]


def jac(v, rm, h):
    J = [[None] * n for _ in range(n)]
    for j in range(n):
        vp = list(v)
        vp[j] = vp[j] + h
        vm = list(v)
        vm[j] = vm[j] - h
        a = P.resid(vp, rm)
        b = P.resid(vm, rm)
        for i in range(n):
            J[i][j] = ((a[i] - b[i]) / (2 * h)).mid()
    a = P.resid(v, rm + h)
    b = P.resid(v, rm - h)
    Frm = [((x - y) / (2 * h)).mid() for x, y in zip(a, b)]
    return J, Frm


F = P.resid(vA, rmA)
say("residual at the stored state:", [f"{float(x.mid()):+.2e}" for x in F])
J1, Frm1 = jac(vA, rmA, arb("1e-10"))
say(f"J at h 1e-10 [{time.time() - T0:.0f}s]")
J2, Frm2 = jac(vA, rmA, arb("2e-10"))
say(f"J at h 2e-10 [{time.time() - T0:.0f}s]")
J3, Frm3 = jac(vA, rmA, arb("1e-7"))
say(f"J at h 1e-7 [{time.time() - T0:.0f}s]")
mp.mp.dps = int(PREC * 0.30103) - 2
M1 = mp.matrix([[mp.mpf(J1[i][j].str(60, radius=False)) for j in range(n)] for i in range(n)])
M2 = mp.matrix([[mp.mpf(J2[i][j].str(60, radius=False)) for j in range(n)] for i in range(n)])
M3 = mp.matrix([[mp.mpf(J3[i][j].str(60, radius=False)) for j in range(n)] for i in range(n)])
R = (M1 * 4 - M2) / 3  # Richardson: removes the h^2 term
say(
    "max |J(1e-10) - Richardson| =",
    mp.nstr(max(abs(x) for x in (M1 - R)), 5),
    " max |J(1e-7) - Richardson| =",
    mp.nstr(max(abs(x) for x in (M3 - R)), 5),
)
U, S, V = mp.svd_r(R)
say("singular values:", [mp.nstr(s, 6) for s in S])
for k in range(n):
    say(
        f"  sigma_{k} {mp.nstr(S[k], 5)}: right vector "
        + ", ".join(f"{nm} {mp.nstr(V[k, j], 4)}" for j, nm in enumerate(names))
    )
    say(f"        left vector " + ", ".join(f"{mp.nstr(U[i, k], 4)}" for i in range(n)))
# perturbation of each singular value by the h^2 error of J(1e-10)
E = M1 - R
say(
    "||J(1e-10) - R|| in the soft pair: u_min^T E v_min =", mp.nstr((U[:, n - 1].T * E * V[n - 1, :].T)[0], 5)
)
# soft direction in ring harmonics
G = vA[:2]
m = vA[2:6]


def harm(G, m, js):
    th, w = P.atoms(G, m)
    return [2 * sum((wk * (j * t).cos() for t, wk in zip(th, w)), arb(0)) for j in js]


js = list(range(2, 26, 2))
b0 = harm(G, m, js)
say("ring harmonics b_j (j 2..24):", [f"{float(x.mid()):+.5f}" for x in b0])
for k in (n - 1, n - 2, n - 3):
    d = [arb(mp.nstr(V[k, j], 40)) for j in range(n)]
    e = arb("1e-12")
    bp = harm([G[i] + e * d[i] for i in range(2)], [m[i] + e * d[2 + i] for i in range(4)], js)
    bm = harm([G[i] - e * d[i] for i in range(2)], [m[i] - e * d[2 + i] for i in range(4)], js)
    say(
        f"  d b_j along v_{k} (sigma {mp.nstr(S[k], 4)}):",
        [f"{float(((x - y)/(2*e)).mid()):+.4f}" for x, y in zip(bp, bm)],
    )
# branch tangent
Fr = mp.matrix([mp.mpf(x.str(60, radius=False)) for x in Frm1])
tan = -mp.lu_solve(R, Fr)
sec = [
    mp.mpf((vA[j] - vB[j]).str(60, radius=False)) / mp.mpf((rmA - rmB).str(60, radius=False))
    for j in range(n)
]
say("F_rm:", [mp.nstr(x, 5) for x in Fr])
say("u_k^T F_rm:", [mp.nstr((U[:, k].T * Fr)[0], 5) for k in range(n)])
say("tangent dv/drm:", [mp.nstr(x, 6) for x in tan])
say("secant  dv/drm:", [mp.nstr(x, 6) for x in sec])
say("tangent in right singular coordinates:", [mp.nstr((V[k, :] * tan)[0], 5) for k in range(n)])
json.dump(
    {
        "rm": dA["rm"],
        "J": [[mp.nstr(R[i, j], 45) for j in range(n)] for i in range(n)],
        "Frm": [mp.nstr(x, 45) for x in Fr],
        "S": [mp.nstr(s, 30) for s in S],
        "U": [[mp.nstr(U[i, j], 40) for j in range(n)] for i in range(n)],
        "V": [[mp.nstr(V[i, j], 40) for j in range(n)] for i in range(n)],
        "tangent": [mp.nstr(x, 40) for x in tan],
    },
    open(os.path.join(HERE, "D_work", "diag_jac.json"), "w"),
    indent=1,
)
say(f"done [{time.time() - T0:.0f}s]")
