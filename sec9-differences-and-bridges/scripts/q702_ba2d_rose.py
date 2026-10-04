"""q702 (copy of q700 as text): env SHAPE=rose gives the analytic domain r <= A (1 - EPS cos 4 theta) (bulges toward the
diagonals, like a superellipse with p > 2); otherwise q700. Amendment 1045. q700 (copy of q698 as text with a second semi-axis B along y: the domain is (|x|/A)^p + (|y|/B)^p <= 1; B = A gives q698;
Amendment 1041). q698 (copy of q696 as text; Amendment 1035): warm start from a saved q696/q698 state (env INIT) and an over-relaxed update
w <- w exp(BETA (D - max D)), BETA adaptive: a step is kept only if the lower bound rises, else BETA halves (floor 1); after a kept step BETA grows by 1.2 up to env BETA (default 4); stops when the bracket falls below env GAPSTOP (default 2e-5). q696 (exploratory): q694 with the wall candidates placed exactly on the boundary curve. Input candidates: M points on the curve
|x|^p + |y|^p = A^p (equal arc-length spacing) plus the grid points with gauge radius <= A - GAP. The Gaussian kernel is split exactly,
phi = phi_a * phi_b (variances SA^2 and 1 - SA^2): off-grid masses are spread onto the grid with phi_a (sparse matrix S, spectrally
accurate for h << SA), grid masses are convolved with phi_a by FFT, and everything with phi_b by FFT. D at an off-grid point is
-(S^T (phi_b * log p)) - h(Y|X). Env P, A, H, PAD, M, GAP, SA, ITERS, OUT. Usage: py q696_ba2d_exactwall.py"""

import os, time
import numpy as np
from scipy.fft import rfft2, irfft2
from scipy import sparse

P = float(os.environ.get("P", "2"))
A = float(os.environ.get("A", "3"))
B = float(os.environ.get("B", os.environ.get("A", "3")))
h = float(os.environ.get("H", "0.05"))
PAD = float(os.environ.get("PAD", "8"))
M = int(os.environ.get("M", "2000"))
GAP = float(os.environ.get("GAP", "0.05"))
SA = float(os.environ.get("SA", "0.3"))
ITERS = int(os.environ.get("ITERS", "8000"))
OUT = os.environ.get("OUT", "q696_out.npz")
n = int(np.ceil(2 * (A + PAD) / h))
n += n % 2
x = (np.arange(n) - n // 2) * h
X, Y = np.meshgrid(x, x, indexing="ij")
gauge = (
    (lambda U, V: A * np.maximum(np.abs(U) / A, np.abs(V) / B))
    if np.isinf(P)
    else (lambda U, V: A * ((np.abs(U) / A) ** P + (np.abs(V) / B) ** P) ** (1 / P))
)
SHAPE = os.environ.get("SHAPE", "")
EPS = float(os.environ.get("EPS", "0"))
if SHAPE == "rose":  # gauge of the rose: |x| / (1 - EPS cos 4 theta), boundary at A
    gauge = lambda U, V: np.hypot(U, V) / (1 - EPS * np.cos(4 * np.arctan2(V, U)))
mask = gauge(X, Y) <= A - GAP + 1e-12
# boundary curve, equal arc length: dense polar sampling, then resample
t = np.linspace(0, 2 * np.pi, 200001)[:-1]
c, s_ = np.cos(t), np.sin(t)
rad = A / gauge(c, s_)
bx, by = rad * c, rad * s_
seg = np.hypot(np.diff(np.append(bx, bx[0])), np.diff(np.append(by, by[0])))
L = np.concatenate([[0], np.cumsum(seg)])
u = np.arange(M) * L[-1] / M
bxM = np.interp(u, L, np.append(bx, bx[0]))
byM = np.interp(u, L, np.append(by, by[0]))
# sparse spreading matrix S (grid x M): phi_a(z - x_m) h^2 for |z - x_m| <= 7 SA
rows, cols, vals = [], [], []
R = int(np.ceil(7 * SA / h))
for m in range(M):
    i0 = int(round(bxM[m] / h)) + n // 2
    j0 = int(round(byM[m] / h)) + n // 2
    ii, jj = np.meshgrid(np.arange(i0 - R, i0 + R + 1), np.arange(j0 - R, j0 + R + 1), indexing="ij")
    vv = np.exp(-0.5 * ((x[ii] - bxM[m]) ** 2 + (x[jj] - byM[m]) ** 2) / SA**2) / (2 * np.pi * SA**2) * h * h
    rows.append((ii * n + jj).ravel())
    cols.append(np.full(ii.size, m))
    vals.append(vv.ravel())
S = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, M))
d = np.minimum(np.arange(n), n - np.arange(n)) * h
ker = lambda var: rfft2(np.exp(-0.5 * (d[:, None] ** 2 + d[None, :] ** 2) / var) / (2 * np.pi * var) * h * h)
Ff, Fb = ker(1.0), ker(1.0 - SA**2)
cf = lambda f: irfft2(rfft2(f) * Ff, s=(n, n))
cb = lambda f: irfft2(rfft2(f) * Fb, s=(n, n))
hY = np.log(2 * np.pi * np.e)
wg = mask / (mask.sum() + M) * 1.0
wb = np.full(M, 1.0 / (mask.sum() + M))
BETA = float(os.environ.get("BETA", "4"))
GAPSTOP = float(os.environ.get("GAPSTOP", "2e-5"))
if os.environ.get("INIT"):
    z0 = np.load(os.environ["INIT"])
    assert z0["wg"].shape == wg.shape and len(z0["wb"]) == M
    wg = np.where(mask, z0["wg"], 0.0) + 1e-300 * mask
    wb = z0["wb"] + 1e-300
    Z = wg.sum() + wb.sum()
    wg /= Z
    wb /= Z
T0 = time.time()
bcur = 1.0
prev = None


def evaluate(wg, wb):
    spread = (S @ wb).reshape(n, n) / (h * h)
    p = np.maximum((cf(wg) + h * h * cb(spread)) / (h * h), 1e-300)
    lp = np.log(p)
    Dg = -cf(lp) - hY
    Db = -(S.T @ cb(lp).ravel()) - hY
    return Dg, Db, float((wg * Dg).sum() + (wb * Db).sum()), float(max(Dg[mask].max(), Db.max()))


Dg, Db, lo, hi = evaluate(wg, wb)
for it in range(ITERS + 1):
    if it % 1000 == 0 or it == ITERS:
        print(
            f"it {it}: C in [{lo:.8f}, {hi:.8f}] (gap {hi - lo:.2e}); wall mass {wb.sum():.4f}; beta {bcur:.2f}  [{time.time() - T0:.0f}s]",
            flush=True,
        )
    if hi - lo < GAPSTOP:
        break
    while True:
        wg2 = wg * np.exp(np.where(mask, bcur * (Dg - hi), -np.inf))
        wb2 = wb * np.exp(bcur * (Db - hi))
        Z = wg2.sum() + wb2.sum()
        wg2 /= Z
        wb2 /= Z
        Dg2, Db2, lo2, hi2 = evaluate(wg2, wb2)
        if lo2 >= lo or bcur <= 1.0:
            break
        bcur = max(1.0, bcur / 2)
    wg, wb, Dg, Db, lo, hi = wg2, wb2, Dg2, Db2, lo2, hi2
    bcur = min(BETA, bcur * 1.2)
np.savez_compressed(
    OUT,
    SHAPE=SHAPE,
    EPS=EPS,
    B=B,
    wg=wg,
    wb=wb,
    bx=bxM,
    by=byM,
    x=x,
    mask=mask,
    lo=lo,
    hi=hi,
    P=P,
    A=A,
    h=h,
    Db=Db,
)
print(f"saved {OUT}; grid {n} x {n}, {int(mask.sum())} interior points, {M} wall points")
