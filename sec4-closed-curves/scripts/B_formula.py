"""B_formula: the leading-order correction to the copying-ring event, term by term, from two B_lr runs at the copying-ring
eps (M = K/2 and M' = K/2 + 1) and one M = K/2 run at eps e^DX (for the slope). Also checks the OPUC affine relation away from the event.
Event condition at M' = K/2 + 1 (derivation in B_derivation.md, part (c)):
  lambda_K h_K = -L_(K,K+2) b_(K+2) - 2 sgn(pi_K) |g_(K+2)| + (pi_(K-2)/pi_K) g_(K+2),   g_(K+2) = lambda_(K+2) (b_(K+2) - c_(K+2))
with h_K = b_K - c_K^(K/2) on the copying ring; delta ln eps = h_K(event) / (d h_K / d ln eps).
Usage: py B_formula.py FUNC RP K PHASE EPS_PRED TAG   (env passes to B_lr)"""

import sys, os, subprocess
import mpmath as mp
import B_ana as A

mp.mp.dps = 80
FUNC, RP, K, PHASE, EP, TAG = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    sys.argv[4],
    mp.mpf(sys.argv[5]),
    sys.argv[6],
)
HERE = os.path.dirname(os.path.abspath(__file__))
DX = mp.mpf(os.environ.get("DX", "0.0015"))
k = K // 2


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


def pis(th, n):
    """cosine coefficients pi_0..pi_n of Pi(t) = prod (1 - cos(t - t_k)) by a trapezoid of 4n nodes (exact for degree < 4n)."""
    N = 4 * n
    P = [mp.fprod(1 - mp.cos(2 * mp.pi * s / N - tk) for tk in th) for s in range(N)]
    return [
        (1 if j == 0 else 2) * mp.fsum(P[s] * mp.cos(j * 2 * mp.pi * s / N) for s in range(N)) / N
        for j in range(n + 1)
    ]


d0, c0, L0 = run(EP, k)
d1, c1, L1 = run(EP, k + 1)
e2 = EP * mp.exp(DX)
d2, c2, L2 = run(e2, k)
ORB, v, n = A.copy_ring(c0, K, PHASE)
ORB2, v2, n2 = A.copy_ring(c2, K, PHASE, v)
bK, bK2 = A.bj(ORB, v, K), A.bj(ORB2, v2, K)
hK0, hK2 = bK - c0[-1], bK2 - c2[-1]
slope = (hK2 - hK0) / DX
lamK, lamK2, LK_K2, LK2_K = L1[k - 1, k - 1], L1[k, k], L1[k - 1, k], L1[k, k - 1]
bK2p = A.bj(ORB, v, K + 2)
cK2p = (-d1["rv"][k] / lamK2) if LOWREF else c1[k]
gK2 = lamK2 * (bK2p - cK2p)
th, w = A.atoms(ORB, v)
P = pis(th, K)
sp = mp.sign(P[K])
T_kap = -LK_K2 * bK2p / lamK
T_q = -2 * sp * abs(gK2) / lamK
T_pi = (P[K - 2] / P[K]) * gK2 / lamK
hev = T_kap + T_q + T_pi
print(
    f"B_formula {FUNC} rp {RP} K {K} {PHASE} at eps {mp.nstr(EP, 10)}: copy-ring residual {mp.nstr(n, 2)}; h_K(eps) {mp.nstr(hK0, 4)} (should be ~0)"
)
print(
    f"  lambda_K {mp.nstr(lamK, 8)}  lambda_K+2 {mp.nstr(lamK2, 8)}  ratio {mp.nstr(lamK2/lamK, 6)};  L_(K,K+2) {mp.nstr(LK_K2, 6)}  L_(K+2,K) {mp.nstr(LK2_K, 6)}"
    f"  L_(K,K+2)/sqrt(lam lam) {mp.nstr(LK_K2/mp.sqrt(lamK*lamK2), 5)}"
)
print(
    f"  c_K(K/2) {mp.nstr(c0[-1], 8)} b_K {mp.nstr(bK, 8)};  c_K+2 {mp.nstr(cK2p, 8)}  b_K+2 {mp.nstr(bK2p, 8)};  pi_K {mp.nstr(P[K], 5)} pi_(K-2)/pi_K {mp.nstr(P[K-2]/P[K], 5)}"
)
print(
    f"  terms of h_K at the event: kappa {mp.nstr(T_kap, 5)}  q-variation {mp.nstr(T_q, 5)}  Pi-asymmetry {mp.nstr(T_pi, 5)}  sum {mp.nstr(hev, 5)}"
)
print(
    f"  d h_K / d ln eps {mp.nstr(slope, 6)} ((K/2) c_K = {mp.nstr(k*c0[-1], 6)});  predicted offset {mp.nstr(100*(mp.exp(hev/slope) - 1), 5)} %"
    f"  [q only {mp.nstr(100*T_q/slope, 4)} %, kappa only {mp.nstr(100*T_kap/slope, 4)} %, Pi only {mp.nstr(100*T_pi/slope, 4)} %]"
)
# OPUC affine relation away from the event: c_K - b_K against alpha_(K-1) - tau times prod (1 - alpha_j^2)
tau = 1 if "X" in ORB else -1
for cc, bb, lab in ((c0, bK, "eps_pred"), (c2, bK2, "eps_pred e^DX")):
    m = A.moments(cc, K)
    al = A.verblunsky(m, K)
    pr = mp.fprod(1 - a**2 for a in al[: K - 1])
    print(
        f"  OPUC at {lab}: alpha_(K-1) {mp.nstr(al[K-1], 12)}; c_K - b_K {mp.nstr(cc[-1] - bb, 10)}; 2 (alpha - tau) prod(1 - alpha^2) "
        f"{mp.nstr(2*(al[K-1] - tau)*pr, 10)};  -2 (alpha - tau) prod {mp.nstr(-2*(al[K-1] - tau)*pr, 10)}"
    )
