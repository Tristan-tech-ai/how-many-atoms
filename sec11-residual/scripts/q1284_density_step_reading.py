"""q1284 (8 Oct 2026, a reading, no item): does the central-difference step H = 1e-3 of the density rho = dN_half/da in the root a_med change a_med by an amount of the size of the common offset (6e-12 in the amplitude, q1283)? a_med of the NPMLE, immune and k = 3 functionals at K_b 40, 44, 48, 52 with H = 1e-3, 1e-4, 1e-5 (the same quintic splines).
Control (first): a_med (H = 1e-3) of the NPMLE at K_b 44 equals the value of q1279_amed.json within 1e-12.
Usage: py q1284_density_step_reading.py  (from temporal_within/centre_scripts/)"""

import json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np
import q1250_quintic_residual as Q
import q1271_linear_in_k_blind as Z

C, N = Q.C, Q.N
sn = [s for s in json.load(open("c33_immc_D146.json"))["stages"] if s["conv"] and s["D"] == 92.0][0]
pi_ = np.array(sn["p"][: sn["nfree"]])
pi_ = pi_[pi_ <= 72]
Nf = {
    1: Q.make(lambda a: Q.N4(a, N.LAW), N.pn, 5),
    2: Q.make(lambda a: Q.N4(a, (0.22573, 0.06806)), pi_, 5),
    3: Z.nhalf(3, "c33_k3c.json", 92.0, 5),
}
ref = json.load(open("q1279_amed.json"))
Q.H = 1e-3
ok = abs(Q.root(Nf[1], 44) - ref["1"]["44"]) < 1e-12
print(
    f"control: NPMLE a_med K_b 44 (H = 1e-3) {Q.root(Nf[1], 44):.12f} against the table {ref['1']['44']:.12f}: {ok}"
)
if not ok:
    print("READING q1284: VOID (control)")
    sys.exit(2)
for k in (1, 2, 3):
    row = []
    for K in (40, 44, 48, 52):
        Q.H = 1e-3
        a0 = Q.root(Nf[k], K)
        d = []
        for H in (1e-4, 1e-5):
            Q.H = H
            d.append((Q.root(Nf[k], K) - a0) * 1e9)
        row.append(f"K_b {K}: {d[0]:+.5f}, {d[1]:+.5f}")
    print(f"k = {k}: a_med(H) - a_med(1e-3) in 1e-9 for H = 1e-4, 1e-5: " + "; ".join(row))
Q.H = 1e-3
print("READING q1284 (no verdict)")
