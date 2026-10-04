"""q859 (04 Oct 2026, BACKLOG 81): TEXT COPY of q696 for an ANNULUS with two exact walls: the outer circle |y| = R (centre 0) and the
inner circle |y - (E, 0)| = RI (eccentric if E != 0). Wall candidates: MO points on the outer circle and MI points on the inner circle,
equal angle spacing; interior candidates: the grid points with |y| <= R - GAP and |y - (E, 0)| >= RI + GAP. Everything else as q696
(exact kernel split phi = phi_a * phi_b, sparse spreading of wall masses, FFT; Blahut-Arimoto with the bracket [lo, hi] on C).
Env R, RI, E, H, PAD, MO, MI, GAP, SA, ITERS, OUT. Usage: py q859_ba2d_annulus.py. q696's docstring follows.
q696 (exploratory): q694 with the wall candidates placed exactly on the boundary curve. Input candidates: M points on the curve
|x|^p + |y|^p = A^p (equal arc-length spacing) plus the grid points with gauge radius <= A - GAP. The Gaussian kernel is split exactly,
phi = phi_a * phi_b (variances SA^2 and 1 - SA^2): off-grid masses are spread onto the grid with phi_a (sparse matrix S, spectrally
accurate for h << SA), grid masses are convolved with phi_a by FFT, and everything with phi_b by FFT. D at an off-grid point is
-(S^T (phi_b * log p)) - h(Y|X). Env P, A, H, PAD, M, GAP, SA, ITERS, OUT. Usage: py q696_ba2d_exactwall.py"""

import os, time
import numpy as np
from scipy.fft import rfft2, irfft2
from scipy import sparse

RO = float(os.environ.get("R", "15"))
RI = float(os.environ.get("RI", "10"))
E = float(os.environ.get("E", "0"))
h = float(os.environ.get("H", "0.05"))
PAD = float(os.environ.get("PAD", "8"))
MO = int(os.environ.get("MO", "3800"))
MI = int(os.environ.get("MI", "2500"))
GAP = float(os.environ.get("GAP", "0.05"))
SA = float(os.environ.get("SA", "0.3"))
ITERS = int(os.environ.get("ITERS", "8000"))
OUT = os.environ.get("OUT", "q859_out.npz")
assert RI + abs(E) < RO
n = int(np.ceil(2 * (RO + PAD) / h))
n += n % 2
x = (np.arange(n) - n // 2) * h
X, Y = np.meshgrid(x, x, indexing="ij")
mask = (X**2 + Y**2 <= (RO - GAP) ** 2 + 1e-12) & ((X - E) ** 2 + Y**2 >= (RI + GAP) ** 2 - 1e-12)
to = np.arange(MO) * 2 * np.pi / MO
ti = np.arange(MI) * 2 * np.pi / MI
bxM = np.concatenate([RO * np.cos(to), E + RI * np.cos(ti)])
byM = np.concatenate([RO * np.sin(to), RI * np.sin(ti)])
M = MO + MI
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
            f"it {it}: C in [{lo:.8f}, {hi:.8f}] (gap {hi - lo:.2e}); wall mass outer {wb[:MO].sum():.4f} inner {wb[MO:].sum():.4f}  [{time.time() - T0:.0f}s]",
            flush=True,
        )
    if hi - lo < 1e-9:
        break
    wg = wg * np.exp(np.where(mask, Dg - hi, -np.inf))
    wb = wb * np.exp(Db - hi)
    Z = wg.sum() + wb.sum()
    wg /= Z
    wb /= Z
np.savez_compressed(
    OUT, wg=wg, wb=wb, bx=bxM, by=byM, x=x, mask=mask, lo=lo, hi=hi, R=RO, RI=RI, E=E, h=h, Db=Db, MO=MO
)
print(
    f"saved {OUT}; grid {n} x {n}, {int(mask.sum())} interior points, {MO} + {MI} wall points; final C in [{lo:.10f}, {hi:.10f}]"
)
