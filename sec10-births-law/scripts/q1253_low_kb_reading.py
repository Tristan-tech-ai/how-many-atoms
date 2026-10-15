"""q1253 (8 Oct 2026, a reading, no item): capacity K_b 10 to 14, bar L1 and |dev(12)| against |dev(16)| with the cubic (k = 3) and the quintic (k = 5) spline of the counting function; statements of Section 6.5 (bar 1.2e-4 at K_b 10; |dev16| > |dev12| at K_b 10 to 14).
Control: the cubic variant gives the dev16 of q1220_out.txt at K_b 10 (-31114.3189e-9... see the output of q1220: dev_med; here dev16 +49823.7895e-9) within 0.01e-9.
Usage: py q1253_low_kb_reading.py  (from temporal_within/centre_scripts/)"""

import json, sys
import q1252_order16_quintic as R

Q, C, m = R.Q, R.C, R.m
gv = lambda f: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))
capN = lambda a: C.Q.N_of(a, *C.CAPL)
out = {}
for k in (3, 5):
    Nf = Q.make(capN, C.X, k)
    out[k] = {}
    for K in range(10, 15):
        A = gv(m.CAP_FILES[K])
        a16, a14, a12 = (R.root_n(Nf, K, n) for n in (16, 14, 12))
        out[k][K] = ((A - a16) * 1e9, (A - a12) * 1e9, (0.3e-9 + 3 * abs(a16 - a14)))
ok = abs(out[3][10][0] - 49823.7895) < 0.01
print(f"control: cubic dev16 at K_b 10 {out[3][10][0]:+.4f}e-9 (q1220 output +49823.7895): {ok}")
if not ok:
    print("READING q1253: VOID (control)")
    sys.exit(2)
for k in (3, 5):
    print(
        f"k = {k}: "
        + "; ".join(
            f"K_b {K}: dev16 {d16:+.1f}e-9, dev12 {d12:+.1f}e-9, bar {b:.3e}, |dev16| > |dev12|: {abs(d16) > abs(d12)}"
            for K, (d16, d12, b) in out[k].items()
        )
    )
print("READING q1253 (no verdict)")
