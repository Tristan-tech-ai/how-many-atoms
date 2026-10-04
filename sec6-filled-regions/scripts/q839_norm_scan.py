"""q839 (03 Oct 2026, BACKLOG 60 ii, exploratory): which normalisation of the inner pair's copying condition is exactly met at the
observed losses? For each case (q822/q833 MI 1 log of the inner formal density, observed pair loss), interpolate (a_0, a_1, d_1) at the
loss and evaluate candidate statistics that equal 1 for a pair on the major axis:
  E  = eccentric second harmonic / 2 (the rule used so far; parameter s with x = A cos s, y = B sin s, A = rho(0), B = rho(pi/2));
  QR = <rho^2 cos 2th g> / <rho^2 g> (quadrupole over second moment, q824 exact);
  QA = <rho^2 cos 2th g> / A^2 (quadrupole over the squared major radius);
  XX = <x^2 - y^2> / <x^2 + y^2> computed with the weight g in the polar angle (same as QR) -- listed for the check;
  M2 = Re <(x + i y)^2 g> / <(x^2 + y^2)> with a unit weight in the eccentric parameter s (moment of the curve-uniform law);
  ES = second harmonic in the arc-length parameter / 2.
Prints each statistic minus 1 at each loss; a normalisation that is the law would give 0 across all cases.
Usage: py q839_norm_scan.py"""

import re
import numpy as np

cases = [
    ("cap ell 3.5", "q822_rp35_fine_MI1.log", 3.40971),
    ("cap ell 3.6", "q822_rp36_MI1.log", 3.39656),
    ("cap ell 3.7", "q822_rp37_MI1_ext.log", 3.383455),
    ("cap ell 3.85", "q822_rp385_MI1_ext.log", 3.36344),
    ("cap ell 4.0", "q822_rp40_MI1_ext.log", 3.34226),
    ("cap ell 4.15", "q822_rp415_MI1.log", 3.31902),
    ("cap ell 4.3", "q822_rp43_MI1.log", 3.29254),
    ("cap mix 3.7", "q832_mx37_inner_fine.log", 3.381956),
    ("lfp ell 2.6", "q828_lfp26_inner_fine.log", 2.272753),
    ("lfp mix 2.6", "q833_lfpmx26_inner.log", 2.271647),
    ("rd ell 2.9", "q830_rd29_inner_fine.log", 2.534208),
    ("rd mix.2 2.9", "q833_rdmx29_inner.log", 2.531636),
    ("rd mix.1 2.9", "q833_rdmx01_29_inner.log", 2.532921),
]
pat = re.compile(r"rm ([0-9.]+): .*inner radius a_0..: (\[[^\]]*\]); inner d_1..: (\[[^\]]*\])")
th = np.linspace(0, 2 * np.pi, 100001)[:-1]


def load(f):
    out = []
    for l in open(f):
        m = pat.search(l)
        if m:
            out.append((float(m.group(1)), eval(m.group(2)), eval(m.group(3))))
    return out


def at(rows, rm):
    r = sorted(rows, key=lambda q: abs(q[0] - rm))[:2]
    (x0, a0, d0), (x1, a1, d1) = r
    w = (rm - x0) / (x1 - x0)
    return [p + w * (q - p) for p, q in zip(a0, a1)], [p + w * (q - p) for p, q in zip(d0, d1)]


print(f"{'case':14s} {'eps':>7s} {'E-1':>10s} {'QR-1':>10s} {'QA-1':>10s} {'M2-1':>10s} {'ES-1':>10s}")
for name, f, rm in cases:
    a, d = at(load(f), rm)
    rho = sum(ak * np.cos(2 * k * th) for k, ak in enumerate(a))
    g = 1 + sum(dk * np.cos(2 * (k + 1) * th) for k, dk in enumerate(d))
    x, y = rho * np.cos(th), rho * np.sin(th)
    A, B = rho[0], rho[len(th) // 4]
    s = np.unwrap(np.arctan2(y / B, x / A))
    E = np.mean(np.cos(2 * s) * g)  # harmonic/2
    QR = np.mean(rho**2 * np.cos(2 * th) * g) / np.mean(rho**2 * g)
    QA = np.mean(rho**2 * np.cos(2 * th) * g) / A**2
    dl = np.hypot(np.gradient(x, th), np.gradient(y, th))
    sig = np.concatenate([[0], np.cumsum((dl[1:] + dl[:-1]) / 2) * (th[1] - th[0])])
    sig = 2 * np.pi * sig / (sig[-1] + (dl[-1] + dl[0]) / 2 * (th[1] - th[0]))
    ES = np.mean(np.cos(2 * sig) * g)
    # law g ds/2pi pushed to s: weight per s is g * dth/ds; complex moment over the curve, normalised by the second moment
    dth_ds = np.gradient(th, s)
    M2 = np.mean((x**2 - y**2) * g) / np.mean((x**2 + y**2) * g)
    print(
        f"{name:14s} {(A - B)/(A + B):7.4f} {E - 1:+10.2e} {QR - 1:+10.2e} {QA - 1:+10.2e} {M2 - 1:+10.2e} {ES - 1:+10.2e}"
    )
