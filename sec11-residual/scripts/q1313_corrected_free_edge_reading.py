"""q1313 (8 Oct 2026, EXPLORATORY reading, no item, no verdict; after the mechanism test q1312 PASS): the stored free-edge transitions k = 1, 2, 3, 4, 5 at K_b 38 to 52 (files u1279) corrected for the early stop of the locator: a_corr = a_t + r^2 x (last secant step), r = (last step) / (previous step) from the stored history (formula validated at 33 events of k = 3, 4, 5 by q1312, bands 1 to 4 percent). R = (a - a_med,quintic) 1e9; the closure R_pred = sigma Da(v^2) (1 + g / v^2) with the (sigma, g) of q1293 on K_b 14 to 40 (k = 1: 3.0087, -12.325; k = 2: 5.5832, -15.019; k = 3: 8.2010, -17.941; k = 4: 10.8222, -20.141; k = 5: 13.4703, -21.908).
Printed per k and K_b: shift (1e-12), R_old, R_corr, R_corr / R_pred and R_old / R_pred; at the end the mean and spread of R / R_pred for K_b 41 to 52 (old and corrected) and the mean ratio R_k / R_1 for the old and the corrected values at the K_b where both exist.
The formula was validated for the c35 lineage (k = 3, 4, 5); q626 (k = 1) and q622 (k = 2) contain the same stop rule (lines 200-208 and 181-189, text compared by grep), but the formula is not tested for them.
Control (first): the shifts of this script at k = 3, K_b 44, 46, 50 equal those of the reruns of q1310 within 5 percent; Da(115) = 0.21082.
Usage: py q1313_corrected_free_edge_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
import mpmath as mp

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1282_extended_decay_reading as D
    import q1220_borel_pade_births_law as T
Q, W, N5 = D.Q, D.W, D.N5
mp.mp.dps = 30
Bf = T.machine(T.coef_series(), 8, 8)[0]


def Da(v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(T.ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


SG = {
    1: (3.0087, -12.325),
    2: (5.5832, -15.019),
    3: (8.2010, -17.941),
    4: (10.8222, -20.141),
    5: (13.4703, -21.908),
}
FILE = {
    1: "q607_u1279_K{K}",
    2: "c19_u1279_K{K}",
    3: "q607_u1279_k3_K{K}",
    4: "q607_u1279_k4_K{K}",
    5: "q607_u1279_k5_K{K}",
}


def shift(f):
    j = json.load(open(f + ".json"))
    sec = [x for x in j["hist"] if "it" in x and "secs" not in x]
    a = [x["a"] for x in sec]
    if len(a) < 3:
        return None
    st = [a[i + 1] - a[i] for i in range(len(a) - 1)]
    r = st[-1] / st[-2]
    return r * r * st[-1]


# control against the reruns of q1310
ok = (
    all(
        abs(
            (
                json.load(open(f"q607_u1310_k3_K{K}.json"))["a_t"]
                - json.load(open(f"q607_u1279_k3_K{K}.json"))["a_t"]
            )
            / shift(f"q607_u1279_k3_K{K}")
            - 1
        )
        < 0.05
        for K in (44, 46, 50)
    )
    and abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082
)
print(f"control: shifts of this script equal the reruns of q1310 within 5 percent, Da(115): {ok}")
if not ok:
    print("READING q1313: VOID (control)")
    sys.exit(2)
res = {}
for k in (1, 2, 3, 4, 5):
    print(f"k = {k}")
    print("  K_b   shift(1e-12)   R_old(1e-9)   R_corr(1e-9)   R_corr/R_pred   R_old/R_pred")
    res[k] = {}
    for K in range(38, 53):
        f = FILE[k].format(K=K)
        a = D.loc(f)
        if a is None:
            continue
        sh = shift(f)
        if sh is None:
            print(f"  {K:3d}  no secant history")
            continue
        am = Q.root(N5[k], K)
        Ro = (a - am) * 1e9
        Rc = (a + sh - am) * 1e9
        v2 = W.v2f(N5[k], a)
        rp = SG[k][0] * Da(v2) * (1 + SG[k][1] / v2)
        res[k][K] = (Ro, Rc, rp, v2)
        print(f"  {K:3d} {sh * 1e12:12.3f} {Ro:13.5f} {Rc:14.5f} {Rc / rp:14.4f} {Ro / rp:13.4f}")
print("mean R / R_pred for K_b 41 to 52 and its spread (old; corrected):")
for k in res:
    o = [res[k][K][0] / res[k][K][2] for K in res[k] if K >= 41]
    c = [res[k][K][1] / res[k][K][2] for K in res[k] if K >= 41]
    if o:
        print(
            f"  k = {k}: events {len(o)}; old mean {np.mean(o):.4f} sd {np.std(o):.4f}; corrected mean {np.mean(c):.4f} sd {np.std(c):.4f}"
        )
print("ratio R_k / R_1 at equal K_b (old; corrected), K_b 41 to 52:")
for k in (2, 3, 4, 5):
    row = []
    for K in range(41, 53):
        if K in res[k] and K in res[1]:
            row.append((K, res[k][K][0] / res[1][K][0], res[k][K][1] / res[1][K][1]))
    print(f"  k = {k}: " + ", ".join(f"{K}: {o:.3f}/{c:.3f}" for K, o, c in row))
print("READING q1313 (exploratory, no verdict)")
