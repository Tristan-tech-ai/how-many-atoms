"""q863 (04 Oct 2026, BACKLOG 82): the 2D least favourable prior on a general region (here the filled ellipse x^2/A^2 + y^2/B^2 <= 1),
unit noise, squared loss summed over both coordinates, by exponentiated-gradient ascent of the Bayes risk on a grid (step H).
FFT convolutions with the Gaussian: p = phi * pi, N_j = phi * (theta_j pi), delta = N/p, and the risk of pi's Bayes rule at theta,
r(theta) = (phi * |delta|^2)(theta) - 2 theta . (phi * delta)(theta) + |theta|^2 (phi symmetric). The Bayes risk is concave in pi with
gradient r: w <- w exp(eta (r - max r)) on the grid points of the region; B = sum w r. Reports B, 2 - B and max r - B (the KKT gap).
Usage: py q863_lfp_grid2d.py A B [H ITERS ETA PAD]
q863b (04 Oct 2026): TEXT COPY of q863 for the annulus RI <= |theta| <= RO (A = B = RO). Usage: py q863b_lfp_grid2d_annulus.py RI RO [H ITERS ETA PAD]
"""

import sys, time
import numpy as np
from scipy.fft import rfft2, irfft2

RI, RO = float(sys.argv[1]), float(sys.argv[2])
A = Bax = RO
H = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
ETA = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
PAD = float(sys.argv[6]) if len(sys.argv) > 6 else 7.0
L = max(A, Bax) + PAD
n = int(np.ceil(2 * L / H))
n += n % 2
x = (np.arange(n) - n // 2) * H
X, Y = np.meshgrid(x, x, indexing="ij")
mask = (X**2 + Y**2 <= RO**2 + 1e-12) & (X**2 + Y**2 >= RI**2 - 1e-12)
d = np.minimum(np.arange(n), n - np.arange(n)) * H
K = rfft2(np.exp(-0.5 * (d[:, None] ** 2 + d[None, :] ** 2)) / (2 * np.pi) * H * H)
conv = lambda f: irfft2(rfft2(f) * K, s=(n, n))
w = mask / mask.sum()
T0 = time.time()
for it in range(ITERS + 1):
    p = np.maximum(conv(w), 1e-300)
    dx = np.clip(conv(w * X) / p, -A, A)
    dy = np.clip(conv(w * Y) / p, -Bax, Bax)  # the posterior mean lies in the hull of the prior
    r = conv(dx * dx + dy * dy) - 2 * (X * conv(dx) + Y * conv(dy)) + X * X + Y * Y
    B = float((w * r).sum())
    rmax = float(r[mask].max())
    if it % 1000 == 0 or it == ITERS:
        print(
            f"it {it}: B {B:.10f}  2 - B {2 - B:.8f}  max r - B {rmax - B:.2e}  [{time.time() - T0:.0f}s]",
            flush=True,
        )
    if rmax - B < 1e-10:
        break
    w = np.where(mask, w * np.exp(ETA * (r - rmax)), 0.0)
    w /= w.sum()
print(
    f"annulus RI {RI} RO {RO} H {H}: Bayes risk {B:.10f}; 2 - B {2 - B:.8f}; KKT gap {rmax - B:.2e}; grid {n} x {n}, {int(mask.sum())} points",
    flush=True,
)
