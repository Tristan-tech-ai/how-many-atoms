"""q1274 (8 Oct 2026, a reading after the verdict of q1273, no item): the amplitude c(k) = R_k / R_NPMLE of the median-sum residual at equal v^2 (the v^2 of each k = 4 event K_b 29 to 36), for k = 0 (capacity, interpolated where its residual is above 0.09e-9), 1 (NPMLE), 2 (immune), 3 (q1271) and 4 (q1273); the increments c(k+1) - c(k) and the second differences.
Control (first): R_4 / R_3 at K_b 33 equals 1.2868 of q1273_out.txt within 1e-4 and the interpolated R_3 at the v^2 of the k = 3 event K_b 33 is its own measured value within 1e-9 relative.
Usage: py q1274_ck_table_reading.py  (from temporal_within/centre_scripts/)"""

import json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np
import q1250_quintic_residual as Q
import q1271_linear_in_k_blind as Z
import q1273_k4_curvature_blind as W

C = Z.C
m = Z.m
law4 = tuple(json.load(open("q1273_predictions.json"))["law"])
N4 = W.nhalf("c33_k4c.json", 92.0, 5, law4)
d4 = {}
for K in W.TARGET:
    a = W.loc(f"q607_u1273_K{K}")
    d4[K] = ((a - Q.root(N4, K)) * 1e9, W.v2f(N4, a))
d3 = W.ref3()
R3 = W.curve(d3)
di, dn = Z.refs()
Ri, Rn = Z.curve(di), Z.curve(dn)
capN = Q.make(lambda a: C.Q.N_of(a, *C.CAPL), C.X, 5)
gv = lambda f: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))
cap = {
    K: ((gv(f) - Q.root(capN, K)) * 1e9, W.v2f(capN, gv(f)))
    for K, f in m.CAP_FILES.items()
    if 20 <= K <= 45 and K not in (38, 39)
}
Rc = Z.curve({K: v for K, v in cap.items() if v[0] > 0.09})
ok = abs(d4[33][0] / R3(d4[33][1]) - 1.2868) < 1e-4 and abs(R3(d3[33][1]) / d3[33][0] - 1) < 1e-9
print(f"control: R_4/R_3 at K_b 33 {d4[33][0] / R3(d4[33][1]):.4f}, reference exact: {ok}")
if not ok:
    print("READING q1274: VOID (control)")
    sys.exit(2)
print(
    "K_b  v^2   c(0)    c(1)  c(2)    c(3)    c(4)   | c(2)-c(1)  c(3)-c(2)  c(4)-c(3) | second differences (1,2,3), (2,3,4)"
)
for K in W.TARGET:
    x = d4[K][1]
    rn = Rn(x)
    c = [Rc(x) / rn if Rc(x) else float("nan"), 1.0, Ri(x) / rn, R3(x) / rn, d4[K][0] / rn]
    inc = [c[2] - c[1], c[3] - c[2], c[4] - c[3]]
    print(
        f"{K:3d} {x:6.1f} {c[0]:6.3f} {c[1]:5.2f} {c[2]:6.3f} {c[3]:7.3f} {c[4]:7.3f} | {inc[0]:7.3f} {inc[1]:9.3f} {inc[2]:9.3f} | {inc[1] - inc[0]:+.3f} {inc[2] - inc[1]:+.3f}"
    )
print("READING q1274 (no verdict)")
