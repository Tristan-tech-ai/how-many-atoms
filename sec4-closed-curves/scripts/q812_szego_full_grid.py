"""q812 (BACKLOG 42, 03 Oct 2026): q811's event finder on the FULL formal density (q790 logs on an eps grid, harmonics to 2M).
Reads every q790 log in a folder (name ..._e<eps>.log; lines 'K nn: c_K value'), Verblunsky alpha_0 .. alpha_(2M-1) by Levinson,
and for each even K <= 2M the crossings of alpha_(K-1) through tau = +-1 between neighbouring grid points (log-linear), flagged
in-regime when |alpha_j| < 1 for j <= K - 2 at the larger eps. Usage: py q812_szego_full_grid.py FOLDER   (env DPS)
"""

import sys, os, re, glob
from mpmath import mp, mpf, log, exp

mp.dps = int(os.environ.get("DPS", "60"))
runs = []
for f in glob.glob(os.path.join(sys.argv[1], "*_e*.log")):
    c = {int(a): mpf(b) for a, b in re.findall(r"K +(\d+): c_K +([-+0-9.e]+)", open(f).read())}
    if c:
        runs.append((mpf(re.search(r"_e([0-9.]+)\.log", f).group(1)), c))
runs.sort(key=lambda r: r[0])
nmax = min(max(c) for _, c in runs)


def verblunsky(c, n):
    mu = [mpf(1)] + [(c.get(m, mpf(0)) / 2 if m % 2 == 0 else mpf(0)) for m in range(1, n + 1)]
    a = [mpf(1)]
    E = mu[0]
    al = []
    for q in range(n):
        k = -sum(a[i] * mu[q + 1 - i] for i in range(q + 1)) / E
        an = a + [mpf(0)]
        a = [an[i] + k * an[q + 1 - i] for i in range(q + 2)]
        E = E * (1 - k * k)
        al.append(-k)
    return al


AL = [(e, verblunsky(c, nmax)) for e, c in runs]
print(f"q812: {len(runs)} densities, eps {[mp.nstr(e, 6) for e, _ in runs]}, harmonics to {nmax}")
for K in range(4, nmax + 1, 2):
    j = K - 1
    out = []
    for (e0, a0), (e1, a1) in zip(AL[:-1], AL[1:]):
        for tau in (1, -1):
            f0, f1 = a0[j] - tau, a1[j] - tau
            if f0 * f1 < 0:
                e = exp(log(e1) + (log(e0) - log(e1)) * f1 / (f1 - f0))
                inside = all(abs(a1[q]) < 1 for q in range(K - 1))
                out.append(f"{mp.nstr(e, 7)} (tau {tau:+d}{'' if inside else ', OUT of regime'})")
    print(f"K {K:3d}: " + ("; ".join(out) if out else "no crossing in the grid"))
