"""q864 (04 Oct 2026, BACKLOG 82): principal Dirichlet eigenvalue (-Laplace u = lambda u, u = 0 outside) of the filled ellipse
x^2/A^2 + y^2/B^2 <= 1 widened by distance DL (the set of points within DL of the ellipse), by the 5-point Laplacian on a grid of step H
(grid points inside the region are unknowns) and shift-invert Lanczos. The disc check (A = B = R, DL = 0) gives j01^2/R^2.
Usage: py q864_dirichlet_offset.py A B DL H"""

import sys, math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

A, Bax, DL, H = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
L = max(A, Bax) + DL + 2 * H
n = int(np.ceil(2 * L / H)) + 1
x = -L + H * np.arange(n)
X, Y = np.meshgrid(x, x, indexing="ij")
inside = (X / A) ** 2 + (Y / Bax) ** 2 <= 1
if DL > 0:
    t = np.linspace(0, 2 * np.pi, 4001)[:-1]
    bx, by = A * np.cos(t), Bax * np.sin(t)
    cand = (~inside) & ((X / (A + DL)) ** 2 + (Y / (Bax + DL)) ** 2 <= 1.0 + 1e-9) | (
        (~inside) & (np.abs(X) <= A + DL) & (np.abs(Y) <= Bax + DL)
    )
    idx = np.where(cand.ravel())[0]
    px, py = X.ravel()[idx], Y.ravel()[idx]
    dmin = np.full(len(idx), np.inf)
    for c0 in range(0, 4000, 200):
        dmin = np.minimum(
            dmin,
            np.sqrt(
                (px[:, None] - bx[None, c0 : c0 + 200]) ** 2 + (py[:, None] - by[None, c0 : c0 + 200]) ** 2
            ).min(1),
        )
    region = inside.ravel().copy()
    region[idx[dmin <= DL]] = True
    region = region.reshape(n, n)
else:
    region = inside
num = -np.ones((n, n), int)
ids = np.where(region.ravel())[0]
num.ravel()[ids] = np.arange(len(ids))
rows, cols, vals = [], [], []
for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    sh = np.full((n, n), -1)
    src = num
    if di == 1:
        sh[:-1, :] = num[1:, :]
    if di == -1:
        sh[1:, :] = num[:-1, :]
    if dj == 1:
        sh[:, :-1] = num[:, 1:]
    if dj == -1:
        sh[:, 1:] = num[:, :-1]
    k = (src >= 0) & (sh >= 0)
    rows.append(src[k])
    cols.append(sh[k])
    vals.append(np.full(k.sum(), -1.0))
N = len(ids)
rows.append(np.arange(N))
cols.append(np.arange(N))
vals.append(np.full(N, 4.0))
Lap = sparse.csr_matrix(
    (np.concatenate(vals) / H**2, (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
)
lam = eigsh(Lap, k=1, sigma=0, which="LM", return_eigenvectors=False)[0]
print(
    f"A {A} B {Bax} DL {DL} H {H}: unknowns {N}; lambda_1 = {lam:.8f}; 4 lambda_1 = {4*lam:.8f}", flush=True
)
