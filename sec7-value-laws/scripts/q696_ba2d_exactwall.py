"""q696 (exploratory): q694 with the wall candidates placed exactly on the boundary curve. Input candidates: M points on the curve
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
    (lambda U, V: np.maximum(np.abs(U), np.abs(V)))
    if np.isinf(P)
    else (lambda U, V: (np.abs(U) ** P + np.abs(V) ** P) ** (1 / P))
)
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
T0 = time.time()
for it in range(ITERS + 1):
    spread = (S @ wb).reshape(n, n) / (h * h)  # phi_a * (boundary masses), as a density on the grid
    p = np.maximum((cf(wg) + h * h * cb(spread)) / (h * h), 1e-300)
    lb = cb(np.log(p))
    Dg = -cf(np.log(p)) - hY
    Db = -(S.T @ lb.ravel()) - hY
    lo = float((wg * Dg).sum() + (wb * Db).sum())
    hi = float(max(Dg[mask].max(), Db.max()))
    if it % 2000 == 0 or it == ITERS:
        print(
            f"it {it}: C in [{lo:.8f}, {hi:.8f}] (gap {hi - lo:.2e}); wall mass {wb.sum():.4f}  [{time.time() - T0:.0f}s]",
            flush=True,
        )
    if hi - lo < 1e-9:
        break
    wg = wg * np.exp(np.where(mask, Dg - hi, -np.inf))
    wb = wb * np.exp(Db - hi)
    Z = wg.sum() + wb.sum()
    wg /= Z
    wb /= Z
np.savez_compressed(OUT, wg=wg, wb=wb, bx=bxM, by=byM, x=x, mask=mask, lo=lo, hi=hi, P=P, A=A, h=h, Db=Db)
print(f"saved {OUT}; grid {n} x {n}, {int(mask.sum())} interior points, {M} wall points")
