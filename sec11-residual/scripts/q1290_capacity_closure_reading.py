"""q1290 (8 Oct 2026, EXPLORATORY reading, no item): does the capacity residual (quintic spline, stored amplitudes of the c60 families, K_b 20 to 45 except 38 and 39) follow the closure R = sigma Da(v^2) (1 - 14.6 / v^2) of amendments 2069 and 2070? Prints R_cap, v^2, Da, and sigma_cap(K_b) = R_cap / [Da (1 - 14.6 / v^2)], to compare with the amplitudes sigma_inf = 3.09, 5.55, 7.90 of k = 1, 2, 3 (a capacity value 0.14 to 0.20 of the NPMLE one would be 0.45 to 0.6).
Control (first): the Da routine reproduces Da(115) = 0.21082e-9 within 1e-4 relative and the capacity residual at K_b 36 is 0.5456e-9 (q1250_out.txt, quintic) within 0.001e-9.
Usage: py q1290_capacity_closure_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, sys
import numpy as np
import mpmath as mp

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1250_quintic_residual as Q
    import q1271_linear_in_k_blind as Z
    import q1220_borel_pade_births_law as T
mp.mp.dps = 30
Bf = T.machine(T.coef_series(), 8, 8)[0]


def Da(v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(T.ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


C, m = Z.C, Z.m
capN = Q.make(lambda a: C.Q.N_of(a, *C.CAPL), C.X, 5)
gv = lambda f: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))
rows = []
for K, f in sorted(m.CAP_FILES.items()):
    if 20 <= K <= 45 and K not in (38, 39):
        a = gv(f)
        r = (a - Q.root(capN, K)) * 1e9
        v2 = Z.v2f(capN, a)
        rows.append((K, r, v2))
r36 = [r for K, r, v in rows if K == 36][0]
ok = abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082 and abs(r36 - 0.5456) < 0.001
print(f"control: Da(115) {Da(115.0):.5f}, capacity R(36) {r36:.4f}: {ok}")
if not ok:
    print("READING q1290: VOID (control)")
    sys.exit(2)
print("K_b   v^2     R_cap(1e-9)   Da(1e-9)    sigma = R/Da   sigma/(1 - 14.6/v^2)")
for K, r, v in rows:
    d = Da(v)
    print(f"{K:3d} {v:7.2f} {r:12.4f} {d:11.5f} {r / d:10.4f} {r / d / (1 - 14.6 / v):12.4f}")
print("READING q1290 (exploratory, no verdict)")
