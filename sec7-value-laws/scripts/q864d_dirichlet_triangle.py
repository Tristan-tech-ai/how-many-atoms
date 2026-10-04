"""q864d (04 Oct 2026): principal Dirichlet eigenvalue of an equilateral triangle (inradius RHO, centroid at the origin, as in q863c)
widened by distance DL in one of two ways: MODE "sharp" = every edge moved outward by DL (a triangle of inradius RHO + DL, exact value
4 pi^2/(9 (RHO + DL)^2)); MODE "round" = all points within distance DL of the triangle (rounded corners). 5-point Laplacian on a grid of
step H, shift-invert; the 5-point code is that of q864 (copied as text). Usage: py q864d_dirichlet_triangle.py MODE RHO DL H
"""

import sys, math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

MODE, RHO, DL, H = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
nrm = [(0.0, -1.0), (math.sqrt(3) / 2, 0.5), (-math.sqrt(3) / 2, 0.5)]
L = 2 * (RHO + DL) + 2 * H
n = int(np.ceil(2 * L / H)) + 1
x = -L + H * np.arange(n)
X, Y = np.meshgrid(x, x, indexing="ij")
if MODE == "sharp":
    region = np.ones_like(X, bool)
    for nx, ny in nrm:
        region &= X * nx + Y * ny <= RHO + DL
else:
    # vertices of the triangle (intersections of the edge lines), then distance to the triangle = 0 inside, else distance to the boundary
    def inter(a, b):
        A = np.array([a, b])
        return np.linalg.solve(A, np.array([RHO, RHO]))

    V = [inter(nrm[i], nrm[(i + 1) % 3]) for i in range(3)]
    inside = np.ones_like(X, bool)
    for nx, ny in nrm:
        inside &= X * nx + Y * ny <= RHO
    dist = np.full(X.shape, np.inf)
    for i in range(3):
        P, Q = V[i], V[(i + 1) % 3]
        dx, dy = Q - P
        t = np.clip(((X - P[0]) * dx + (Y - P[1]) * dy) / (dx * dx + dy * dy), 0, 1)
        dist = np.minimum(dist, np.hypot(X - P[0] - t * dx, Y - P[1] - t * dy))
    region = inside | (dist <= DL)
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
exact = 4 * math.pi**2 / (9 * (RHO + DL) ** 2) if MODE == "sharp" else float("nan")
print(
    f"{MODE} RHO {RHO} DL {DL} H {H}: unknowns {N}; lambda_1 = {lam:.8f}; 4 lambda_1 = {4*lam:.8f}; exact (sharp) {exact:.8f}",
    flush=True,
)
