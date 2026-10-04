"""q872b (1586): NPMLE d = 8 quadratic-law residuals against the capacity d = 8 residuals at the same K_eff.
NPMLE shifts as in q872 (1D q644b, 2D q668, 3D q679_d3, 8D q679_npmle_d8.json); capacity residuals as in q870.
Usage: py q872b_npmle_d8.py
"""

import json, glob


def load(path, split):
    return open(path, encoding="utf-8").read().split(split)[0]


ns = {}
exec(load("q872_npmle_shift_quadratic.py", "S2 = shifts("), ns)  # T, AG, shifts (NPMLE), copied as text
S2n = ns["shifts"](["q668_npmle2d.json", "q668_npmle2d_from23.json"])
S3n = ns["shifts"](["q679_npmle_d3.json"])
S8n = ns["shifts"](["q679_npmle_d8.json"])
cs = {}
exec(load("q870_cap_shift_quadratic.py", "S = {"), cs)  # AG (capacity), shifts, copied as text
S2c = cs["shifts"](["q659_rings2d_evstates.json"])
S3c = cs["shifts"](["q659_shells3d_evstates.json"])
S8c = cs["shifts"](["q662_ball_d8.json", "q666_ball_d8.json"])


def resid(S2, S3, S8, K, d=8):
    kind, s8, A = S8[K]
    s2 = S2[K][1]
    s3 = S3[K][1]
    p = (d - 1) * ((3 * s2 - s3) + (s3 / 2 - s2) * d)
    return kind, A, s8 / p - 1


rows = []
for K in sorted(S8n):
    if K not in S2n or K not in S3n:
        continue
    kind, A, rn = resid(S2n, S3n, S8n, K)
    rc = resid(S2c, S3c, S8c, K)[2] if (K in S8c and K in S2c and K in S3c) else None
    rows.append((K, kind, A, rn, rc))
    print(
        f"K_eff {K:2d} {kind:8s} A {A:8.4f}  NPMLE residual {rn:+.4f}  capacity {('%+.4f' % rc) if rc is not None else '   n/a '}"
        f"  difference {('%+.4f' % (rn - rc)) if rc is not None else 'n/a'}"
    )
sc = [r for r in rows if 7 <= r[0] <= 19 and r[4] is not None]
nin = sum(abs(r[3] - r[4]) <= 0.005 for r in sc)
print(
    f"scored K_eff 7-19: {len(sc)}; within 0.5 %: {nin}; verdict {'PASS' if sc and nin >= 0.8*len(sc) else 'FAIL'}"
)
