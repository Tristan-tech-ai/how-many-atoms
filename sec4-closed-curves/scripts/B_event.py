"""B_event: the linear-response (LR) event for one ring. F(eps) = max_t q(t) / |L_KK| with g - C = Pi q for the stationary
LR ring (harmonics up to 2M'); secant in ln eps from the copying-ring prediction. Each eps: one B_lr run (subprocess) at M'.
Usage: py B_event.py FUNC RP K PHASE EPS_PRED MPLUS TAG   (env passes to B_lr: SRC_S, SHAPE, NF, LAM)"""

import sys, os, subprocess, json, time
import mpmath as mp
import B_ana as A

mp.mp.dps = 80
FUNC, RP, K, PHASE, EP, MPL, TAG = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    sys.argv[4],
    mp.mpf(sys.argv[5]),
    int(sys.argv[6]),
    sys.argv[7],
)
Mp = K // 2 + MPL
HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
v = None


LOWREF = (
    os.environ.get("REF", "") == "low"
)  # reference density solved to K/2 only; harmonics above enter as residuals


def run(eps, M):
    m0 = K // 2 if (LOWREF and M > K // 2) else M
    f = os.path.join(
        HERE, "B_out", f"{TAG}_M{M}" + (f"_m0{m0}" if m0 != M else "") + f"_e{mp.nstr(eps, 12)}.json"
    )
    if not os.path.exists(f):
        env = dict(os.environ)
        env["OUT"] = f
        env["M0"] = str(m0)
        subprocess.run(
            ["py", "-u", os.path.join(HERE, "B_lr.py"), FUNC, RP, mp.nstr(eps, 15), str(M)],
            env=env,
            capture_output=True,
            text=True,
        )
    return A.load(f)


def F(x):
    global v
    eps = mp.exp(x)
    d, c, L = run(eps, Mp)
    A.RV = d["rv"]
    starts = [v] if v is not None else []
    RV = A.RV
    d0, c0, L0 = run(eps, K // 2)
    A.RV = RV
    starts.append(A.copy_ring(c0, K, PHASE)[1])  # the copying ring of the M = K/2 density at this eps
    for s0 in starts:
        ORB, vr, nr, g, qmax, tmax = A.lr_ring(L, c, K, PHASE, Mp, s0)
        if nr < mp.mpf("1e-50"):
            break
    v = vr
    val = qmax / abs(L[K // 2 - 1, K // 2 - 1])
    print(
        f"  eps {mp.nstr(eps, 12)}: max q/|L_KK| {mp.nstr(val, 6)} at {mp.nstr(mp.degrees(tmax), 5)} deg; LR residual {mp.nstr(nr, 2)}  [{time.time()-T0:.0f}s]",
        flush=True,
    )
    return val, vr, c, ORB


x0 = mp.log(EP)
x1 = x0 + mp.mpf(os.environ.get("DX", "0.0015"))
f0, *_ = F(x0)
f1, vr, c, ORB = F(x1)
for it in range(12):
    x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
    x0, f0 = x1, f1
    x1 = x2
    f1, vr, c, ORB = F(x1)
    if abs(f1) < mp.mpf("1e-6"):
        break
e = mp.exp(x1)
print(
    f"B_event {FUNC} rp {RP} K {K} {PHASE} M' {Mp} {TAG}: LR event eps {mp.nstr(e, 10)}; vs copying-ring {mp.nstr(EP, 10)}: {mp.nstr(100*(e/EP - 1), 4)} %; "
    f"ring {[mp.nstr(x, 7) for x in vr[:len(ORB)]]} G {[mp.nstr(mp.degrees(a), 7) for a in vr[len(ORB):]]}; b_j - c_j "
    f"{[mp.nstr(A.bj(ORB, vr, 2*j) - c[j-1], 4) for j in range(1, Mp + 1)]}  [{time.time()-T0:.0f}s]",
    flush=True,
)
