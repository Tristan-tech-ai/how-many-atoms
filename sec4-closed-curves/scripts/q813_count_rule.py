"""q813 (03 Oct 2026, Amendment 1364's count rule as a tool): for every finished q790 log in a folder, the count
K = 1 + the first odd j with |alpha_j| >= 1 (alpha_j the Verblunsky coefficients of the full formal density, Levinson-Durbin), with
K eps and K eps^(1/2). Only j <= 2M - 5 is used (harmonics within four of the truncation edge are not trusted).
Usage: py q813_count_rule.py FOLDER   (env DPS)"""

import sys, os, re, glob
from mpmath import mp, mpf, sqrt

mp.dps = int(os.environ.get("DPS", "80"))


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


rows = []
for f in glob.glob(os.path.join(sys.argv[1], "*_e*.log")):
    txt = open(f).read()
    c = {int(a): mpf(b) for a, b in re.findall(r"K +(\d+): c_K +([-+0-9.e]+)", txt)}
    if not c:
        continue
    M2 = max(c)
    e = mpf(re.search(r"_e([0-9.]+)\.log", f).group(1))
    al = verblunsky(c, M2)
    js = [j for j in range(1, M2 - 4, 2) if abs(al[j]) >= 1]
    rows.append((e, js[0] + 1 if js else None, al[js[0]] if js else None, M2))
for e, K, a, M2 in sorted(rows):
    if K:
        print(
            f"eps {float(e):.6f}: K {K:3d} (alpha_{K-1} {float(a):+.3f}); K eps {float(K*e):.4f}; K eps^(1/2) {float(K*sqrt(e)):.4f}  [harmonics to {M2}]"
        )
    else:
        print(f"eps {float(e):.6f}: no coefficient outside the disc below j = {M2 - 5}")
