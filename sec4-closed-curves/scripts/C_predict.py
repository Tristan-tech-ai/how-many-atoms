"""C_predict.py (02 Oct 2026): the copying-ring rule (q795 form) in deterministic-annealing language.
Fixed ellipse (RP, RM) and source width S; temperature T = 1/beta, component width sigma = sqrt(T/2) (exp(-beta |Y - y|^2) =
exp(-|Y - y|^2 / (2 sigma^2))). Lengths in units of sigma give the unit-variance NPMLE at (RP/sigma, RM/sigma, S/sigma); its full smooth
density c_2..c_K comes from q790 (subprocess, unchanged; FUNC rd, env SRC_S = S/sigma, SHAPE removed, LAM 0, args DEG 6 NA 96 RMAX 20
PREC 256 as q795 uses). The copying ring (moment solve copied as text from q795: D2-symmetric K-ring with b_j = c_j, j = 2..K-2) gives
b_K; the K-ring is lost at T_c where |c_K(T)| = |b_K(T)| (secant in log T); the signs of c_K and b_K at the crossing are printed
(consistent = same sign).
Usage: py C_predict.py RP RM S K PHASE T_GUESS          (PHASE vertex = atom at t = 0, flank = no atom at t = 0)
       py C_predict.py RP RM S K PHASE scan T1,T2,...   (prints c_K, b_K at each T)
       env CP_NA, CP_DEG override q790's NA, DEG (convergence check)."""

import sys, os, re, subprocess, time
import mpmath as mp

mp.mp.dps = 40
RP, RM, S = mp.mpf(sys.argv[1]), mp.mpf(sys.argv[2]), mp.mpf(sys.argv[3])
K = int(sys.argv[4])
PHASE = sys.argv[5]
PHASE = {"vertex": "major", "major": "major", "flank": "flank"}[PHASE]
Q790 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "centre_scripts", "q790_full_numeric.py"
)
T0 = time.time()
M = K // 2
NA = os.environ.get("CP_NA", "96")
DEG = os.environ.get("CP_DEG", "6")


def full_c(T):
    sig = mp.sqrt(T / 2)
    env = dict(os.environ)
    env.pop("SHAPE", None)
    env["LAM"] = "0"
    env["SRC_S"] = mp.nstr(S / sig, 30)
    out = subprocess.run(
        [
            "py",
            "-u",
            Q790,
            "rd",
            mp.nstr(RP / sig, 30),
            mp.nstr((RP - RM) / sig, 30),
            str(M),
            DEG,
            NA,
            "20",
            "256",
        ],
        capture_output=True,
        text=True,
        env=env,
    ).stdout
    c = {int(a): mp.mpf(b) for a, b in re.findall(r"K +(\d+): c_K +([-+0-9.e]+)", out)}
    if len(c) < M:
        raise SystemExit(f"q790 failed at T {T}: {out[-500:]}")
    return [c[2 * j] for j in range(1, M + 1)]


# ---- copied as text from q795 (layout, atoms, bj, ring, start) ----
def layout():
    if PHASE == "flank":  # 1305: no atom at t = 0 (atoms at odd multiples of pi/K)
        return (["Y"] + ["G"] * ((K - 2) // 4)) if K % 4 == 2 else ["G"] * (K // 4)
    if PHASE == "major":
        if K % 4 == 0:
            return ["X", "Y"] + ["G"] * ((K - 4) // 4)
        return ["X"] + ["G"] * ((K - 2) // 4)
    if K % 4 == 2:
        return ["Y"] + ["G"] * ((K - 2) // 4)
    return ["X", "Y"] + ["G"] * ((K - 4) // 4)


ORB = layout()
nG = ORB.count("G")


def atoms(v):
    ms = v[: len(ORB)]
    gs = v[len(ORB) :]
    th, w = [], []
    g = 0
    for o, m in zip(ORB, ms):
        if o == "X":
            th += [mp.mpf(0), mp.pi]
            w += [m] * 2
        elif o == "Y":
            th += [mp.pi / 2, 3 * mp.pi / 2]
            w += [m] * 2
        else:
            a = gs[g]
            g += 1
            th += [a, -a, mp.pi - a, mp.pi + a]
            w += [m] * 4
    return th, w


def bj(v, j):
    th, w = atoms(v)
    return 2 * mp.fsum(m * mp.cos(j * t) for t, m in zip(th, w))


def ring(c, v0):
    v = list(v0)

    def F(v):
        th, w = atoms(v)
        return [mp.fsum(w) - 1] + [bj(v, 2 * j) - c[j - 1] for j in range(1, M)]

    for it in range(60):
        f = F(v)
        n = max(abs(x) for x in f)
        if n < mp.mpf("1e-30"):
            break
        h = mp.mpf("1e-20")
        J = mp.matrix(len(f), len(v))
        for k in range(len(v)):
            vp = list(v)
            vp[k] += h
            fp = F(vp)
            for i in range(len(f)):
                J[i, k] = (fp[i] - f[i]) / h
        dv = mp.lu_solve(J, mp.matrix([-x for x in f]))
        lam = mp.mpf(1)
        for _ in range(30):
            vt = [v[k] + lam * dv[k] for k in range(len(v))]
            for k in range(len(ORB), len(vt)):
                a = mp.fmod(vt[k], mp.pi)
                a = a + mp.pi if a < 0 else a
                vt[k] = mp.pi - a if a > mp.pi / 2 else a
            if max(abs(x) for x in F(vt)) < n:
                break
            lam /= 2
        v = vt
    return v, n


off = mp.mpf(0) if PHASE == "major" else (mp.pi / K if PHASE == "flank" else mp.pi / 2)
gang = sorted({float(mp.fmod(off + 2 * mp.pi * k / K, 2 * mp.pi)) for k in range(K)})
gq = [mp.mpf(a) for a in gang if 1e-9 < a < float(mp.pi / 2) - 1e-9]
v = [mp.mpf(1) / K] * len(ORB) + gq[:nG]
V0 = list(v)
# ---- end of the copied part ----


def g(T):
    global v
    c = full_c(T)
    best = None
    for start in (v, V0):
        w, n = ring(c, start)
        if n < mp.mpf("1e-25") and min(w[: len(ORB)]) > 0:
            best = (w, n)
            break
    if best is None:
        return None, c, None, n
    v = best[0]
    bK = bj(v, K)
    return mp.log(abs(c[-1]) / abs(bK)), c, bK, best[1]


def show(T, f, c, bK, n):
    ms = v[: len(ORB)]
    gs = v[len(ORB) :]
    return (
        f"T {mp.nstr(T, 12)} sigma {mp.nstr(mp.sqrt(T/2), 10)}: c_K {mp.nstr(c[-1], 10)} b_K {mp.nstr(bK, 10) if bK is not None else 'none'}"
        f" ln|c/b| {mp.nstr(f, 6) if f is not None else 'none'}; c_2..c_(K-2) {[mp.nstr(x, 7) for x in c[:-1]]}; ring {ORB} masses "
        f"{[mp.nstr(m, 6) for m in ms]} G deg {[mp.nstr(mp.degrees(a), 7) for a in gs]}; moment residual {mp.nstr(n, 2)}  [{time.time()-T0:.0f}s]"
    )


print(
    f"C_predict rp {sys.argv[1]} rm {sys.argv[2]} s {sys.argv[3]} K {K} phase {sys.argv[5]} (q795 layout {PHASE}: {ORB}); q790 NA {NA} DEG {DEG}",
    flush=True,
)
if sys.argv[6] == "scan":
    for Ts in sys.argv[7].split(","):
        T = mp.mpf(Ts)
        f, c, bK, n = g(T)
        print(show(T, f, c, bK, n), flush=True)
    raise SystemExit
x0 = mp.log(mp.mpf(sys.argv[6]))
x1 = x0 + mp.mpf("0.01")
f0, *_ = g(mp.exp(x0))
f1, c, bK, n = g(mp.exp(x1))
if f0 is None or f1 is None:
    raise SystemExit("no copying ring at the starting temperatures")
for it in range(25):
    x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
    x0, f0 = x1, f1
    x1 = x2
    f1, c, bK, n = g(mp.exp(x1))
    if f1 is None:
        raise SystemExit(f"no copying ring at T {mp.nstr(mp.exp(x1), 10)}")
    print(f"  secant {it}: " + show(mp.exp(x1), f1, c, bK, n), flush=True)
    if abs(f1) < mp.mpf("1e-10"):
        break
T = mp.exp(x1)
cons = "consistent (same sign)" if c[-1] * bK > 0 else "INCONSISTENT (opposite signs)"
print(
    f"C_predict RESULT rp {sys.argv[1]} rm {sys.argv[2]} s {sys.argv[3]} K {K} {sys.argv[5]}: lost at T_c {mp.nstr(T, 10)} (sigma "
    f"{mp.nstr(mp.sqrt(T/2), 10)}, beta {mp.nstr(1/T, 10)}); c_K {mp.nstr(c[-1], 8)} b_K {mp.nstr(bK, 8)} {cons}; ring {ORB} masses "
    f"{[mp.nstr(m, 6) for m in v[:len(ORB)]]} G deg {[mp.nstr(mp.degrees(a), 7) for a in v[len(ORB):]]}; moment residual {mp.nstr(n, 2)}  "
    f"[{time.time()-T0:.0f}s]",
    flush=True,
)
