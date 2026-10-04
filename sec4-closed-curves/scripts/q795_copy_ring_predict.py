"""q795 (Amendment 1297): the copying-ring form of the comb rule, parameter-free. At a given eps, the full smooth density c_2..c_K
comes from q790 (run as a subprocess, unchanged). The copying ring is the D2-symmetric K-ring whose harmonics b_j = 2 sum m cos(j t)
equal c_j for j = 2, 4, ..., K-2 (K/2 - 1 equations, K/2 - 1 free masses and angles); its comb content b_K follows. The ring is lost
where |c_K(eps)| = |b_K(eps)| (secant in log eps). Ring layout from K and PHASE: 'major' = an atom at t = 0 (K = 0 mod 4: X, Y and
(K-4)/4 G orbits; K = 2 mod 4: X and (K-2)/4 G orbits); 'minor' = an atom at t = pi/2 (K = 2 mod 4: Y and (K-2)/4 G orbits).
Usage: py q795_copy_ring_predict.py FUNC RP K PHASE EPS_GUESS   (env SRC_S passes to q790 for FUNC rd)"""

import sys, os, re, subprocess, time
import mpmath as mp

mp.mp.dps = 40
FUNC, RP, K, PHASE, EG = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], mp.mpf(sys.argv[5])
HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
M = K // 2


def full_c(eps):
    out = subprocess.run(
        [
            "py",
            "-u",
            os.path.join(HERE, "q790_full_numeric.py"),
            FUNC,
            RP,
            mp.nstr(eps, 20),
            str(M),
            "6",
            "96",
            "20",
            "256",
        ],
        capture_output=True,
        text=True,
    ).stdout
    c = {int(a): mp.mpf(b) for a, b in re.findall(r"K +(\d+): c_K +([-+0-9.e]+)", out)}
    return [c[2 * j] for j in range(1, M + 1)]


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
        lam = mp.mpf(1)  # damped step, angles folded into (0, pi/2)
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


# regular ring as the first guess: atoms at multiples of 2 pi / K, offset so that the layout's vertex atom is present
off = mp.mpf(0) if PHASE == "major" else (mp.pi / K if PHASE == "flank" else mp.pi / 2)
gang = sorted({float(mp.fmod(off + 2 * mp.pi * k / K, 2 * mp.pi)) for k in range(K)})
gq = [mp.mpf(a) for a in gang if 1e-9 < a < float(mp.pi / 2) - 1e-9]
v = [mp.mpf(1) / K] * len(ORB) + gq[:nG]


V0 = list(v)


def g(eps):
    global v  # 1304: start from the last good ring, else the regular one;
    c = full_c(eps)
    best = None  # reject rings with a non-positive mass
    for start in (v, V0):
        w, n = ring(c, start)
        if n < mp.mpf("1e-25") and min(w[: len(ORB)]) > 0:
            best = (w, n)
            break
    if best is None:
        raise SystemExit(
            f"no copying ring at eps {mp.nstr(eps, 10)} (moment residual {mp.nstr(n, 3)}, masses {[mp.nstr(x, 4) for x in w[:len(ORB)]]})"
        )
    v = best[0]
    bK = bj(v, K)
    return mp.log(abs(c[-1]) / abs(bK)), c[-1], bK, best[1]


x0 = mp.log(EG)
x1 = x0 + mp.mpf("0.01")
f0, *_ = g(mp.exp(x0))
f1, cK, bK, n = g(mp.exp(x1))
for it in range(20):
    x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
    x0, f0 = x1, f1
    x1 = x2
    f1, cK, bK, n = g(mp.exp(x1))
    if abs(f1) < mp.mpf("1e-10"):
        break
ms = v[: len(ORB)]
gs = v[len(ORB) :]
print(
    f"q795 {FUNC} rp {RP} K {K} {PHASE}: lost at eps {mp.nstr(mp.exp(x1), 9)}; c_K {mp.nstr(cK, 8)}, b_K {mp.nstr(bK, 8)}; ring {ORB} masses "
    f"{[mp.nstr(m, 6) for m in ms]} G deg {[mp.nstr(mp.degrees(a), 7) for a in gs]}; moment residual {mp.nstr(n, 2)}  [{time.time()-T0:.0f}s]",
    flush=True,
)
