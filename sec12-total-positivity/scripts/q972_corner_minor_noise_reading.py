"""q972 (7 Oct 2026): READING (no verdict) on whether the five negative corner minors of X = [T | v] found by q971 (K 65 state A 43.5803, rows 0..k-1 and the last k columns, k = 8, 10, 12, 14, 16) are robust against the accuracy of the state.
The minor is linear in v: minor = sum_p v_p C_p with C_p the signed cofactors. It prints for each of the five minors the value, the cancellation ratio sum |v_p C_p| / |sum v_p C_p|, and the sign and size of the minor when v is multiplied entrywise by 1 + delta r_p
with r_p uniform in [-1, 1] (three seeds) for delta = 1e-60, 1e-45, 1e-39 (the polish residual of the stored states is 1e-39 to 1e-40).
Usage: py q972_corner_minor_noise_reading.py STATE.json (from temporal_within/)"""

import sys, os, random
from flint import arb, arb_mat, ctx
import q969_extended_tp_test as Q
import q970_extended_tp_diagnosis as D

path = sys.argv[1]
A_str, m, odd, Tm, vv = D.build(path)
ctx.prec = 800
n = m + 1
lg = lambda d: float((abs(d).log() / arb(10).log()).mid().str(6, radius=False)) if d != 0 else float("-inf")
print(f"state {os.path.basename(path)}: A {A_str}, m {m}", flush=True)
for k in (8, 10, 12, 14, 16):
    cols = list(range(n - k, n))
    rows = list(range(k))

    def minor_of(v):
        X = [[arb(Tm[p][c]) if c < m else arb(v[p]) for c in cols] for p in rows]
        return arb_mat(k, k, [x for r in X for x in r]).det()

    base = minor_of(vv)
    cof = []
    for p in rows:
        sub = arb_mat(k - 1, k - 1, [arb(Tm[r][c]) for r in rows if r != p for c in cols[:-1]]).det()
        cof.append(((-1) ** (p + k - 1)) * sub)
    terms = [arb(vv[p]) * cof[p] for p in rows]
    canc = float((sum((abs(t) for t in terms), arb(0)) / abs(sum(terms, arb(0)))).mid().str(6, radius=False))
    print(
        f"order {k}: minor sign {'negative' if base < 0 else 'positive'}, log10|minor| {lg(base):.2f}, cancellation ratio sum|v_p C_p|/|minor| {canc:.3e}",
        flush=True,
    )
    for delta in (arb("1e-60"), arb("1e-45"), arb("1e-39")):
        res = []
        for seed in (1, 2, 3):
            rng = random.Random(seed * 1000 + k)
            v2 = [vv[p] * (1 + delta * arb(rng.uniform(-1, 1))) for p in range(m)]
            d = minor_of(v2)
            res.append(("-" if d < 0 else ("+" if d > 0 else "?")) + f"{lg(d):.1f}")
        print(
            f"   delta {delta.str(1, radius=False)}: signs and log10 sizes over three seeds: {' '.join(res)}",
            flush=True,
        )
print("END q972 READING (no verdict)")
