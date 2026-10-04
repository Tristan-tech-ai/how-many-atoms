"""B_ana (2 Oct 2026): analysis helpers, mpmath only, no file writes at import.
- copying ring: q795's layout / atoms / bj / ring copied AS TEXT, made re-entrant (K, PHASE passed in)
- Szego quadrature of the formal density (OPUC): para-orthogonal polynomial zeros and weights, Verblunsky coefficients
- linear-response ring: g_i = sum_j L_ij (b_2j - c_2j), i, j = 1..M'; stationary ring of g; max of g - C on [0, pi/2]
"""

import json
import mpmath as mp

mp.mp.dps = 80


def load(path):
    d = json.load(open(path))
    c = [mp.mpf(x) for x in d["c"]]
    L = mp.matrix([[mp.mpf(x) for x in row] for row in d["L"]])
    d["rv"] = [mp.mpf(x) for x in d.get("r", ["0"] * len(c))]
    if d.get("M0", d["M"]) == d["M"]:
        d["rv"] = [mp.mpf(0)] * len(c)  # fully solved reference: residuals are Newton noise
    return d, c, L


# ---- copying ring (q795 text, re-entrant) ----
def layout(K, PHASE):
    if PHASE == "flank":
        return (["Y"] + ["G"] * ((K - 2) // 4)) if K % 4 == 2 else ["G"] * (K // 4)
    if PHASE == "major":
        if K % 4 == 0:
            return ["X", "Y"] + ["G"] * ((K - 4) // 4)
        return ["X"] + ["G"] * ((K - 2) // 4)
    if K % 4 == 2:
        return ["Y"] + ["G"] * ((K - 2) // 4)
    return ["X", "Y"] + ["G"] * ((K - 4) // 4)


def atoms(ORB, v):
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


def bj(ORB, v, j):
    th, w = atoms(ORB, v)
    return 2 * mp.fsum(m * mp.cos(j * t) for t, m in zip(th, w))


def start(K, PHASE):
    ORB = layout(K, PHASE)
    nG = ORB.count("G")
    off = mp.mpf(0) if PHASE == "major" else (mp.pi / K if PHASE == "flank" else mp.pi / 2)
    gang = sorted({float(mp.fmod(off + 2 * mp.pi * k / K, 2 * mp.pi)) for k in range(K)})
    gq = [mp.mpf(a) for a in gang if 1e-9 < a < float(mp.pi / 2) - 1e-9]
    return ORB, [mp.mpf(1) / K] * len(ORB) + gq[:nG]


def newton(F, v0, tol=mp.mpf("1e-60"), nit=80, h=mp.mpf("1e-40"), nO=0, damp=True):
    v = list(v0)
    n = None
    for it in range(nit):
        f = F(v)
        n = max(abs(x) for x in f)
        if n < tol:
            break
        J = mp.matrix(len(f), len(v))
        for k in range(len(v)):
            vp = list(v)
            vp[k] += h
            fp = F(vp)
            for i in range(len(f)):
                J[i, k] = (fp[i] - f[i]) / h
        dv = mp.lu_solve(J, mp.matrix([-x for x in f]))
        lam = mp.mpf(1)
        for _ in range(40):
            vt = [v[k] + lam * dv[k] for k in range(len(v))]
            for k in range(nO, len(vt)):
                a = mp.fmod(vt[k], mp.pi)
                a = a + mp.pi if a < 0 else a
                vt[k] = mp.pi - a if a > mp.pi / 2 else a
            if not damp or max(abs(x) for x in F(vt)) < n:
                break
            lam /= 2
        v = vt
    return v, n


def copy_ring(c, K, PHASE, v0=None):
    """c[j-1] = c_2j. Ring with b_2j = c_2j for j = 1..K/2-1 and unit mass."""
    ORB, vs = start(K, PHASE)
    M = K // 2
    F = lambda v: [mp.fsum(atoms(ORB, v)[1]) - 1] + [bj(ORB, v, 2 * j) - c[j - 1] for j in range(1, M)]
    v, n = newton(F, v0 or vs, nO=len(ORB))
    return ORB, v, n


# ---- OPUC / Szego ----
def moments(c, n):
    """m_k = int e^{ikt} dmu for k = 0..n from cosine coefficients c[j-1] = c_2j (D2: odd moments zero)."""
    m = [mp.mpf(0)] * (n + 1)
    m[0] = mp.mpf(1)
    for k in range(2, n + 1, 2):
        if k // 2 - 1 < len(c):
            m[k] = c[k // 2 - 1] / 2
    return m


def monic_op(m, n):
    """monic orthogonal polynomial Phi_n (coefficients low -> high) for real symmetric moments m_k (|k| <= n)."""
    if n == 0:
        return [mp.mpf(1)]
    A = mp.matrix(n, n)
    rhs = mp.matrix(n, 1)
    for j in range(n):
        for k in range(n):
            A[j, k] = m[abs(k - j)]
        rhs[j] = -m[abs(n - j)]
    x = mp.lu_solve(A, rhs)
    return [x[k] for k in range(n)] + [mp.mpf(1)]


def verblunsky(m, N):
    return [-monic_op(m, n + 1)[0] for n in range(N)]  # alpha_n = -Phi_{n+1}(0) (real case)


def szego(c, K, tau):
    """K-node Szego quadrature exact for |j| <= K-1: zeros of z Phi_{K-1}(z) - tau Phi*_{K-1}(z); weights from moments 0..K-1."""
    m = moments(c, K)
    P = monic_op(m, K - 1)
    Ps = P[::-1]
    B = [mp.mpf(0)] + P  # z Phi_{K-1}
    for k in range(K):
        B[k] -= tau * Ps[k]
    roots = mp.polyroots(B[::-1], maxsteps=400, extraprec=400)
    ts = sorted([mp.arg(r) % (2 * mp.pi) for r in roots])
    mods = [abs(r) for r in roots]
    V = mp.matrix(2 * K - 1, K)
    rhs = mp.matrix(2 * K - 1, 1)  # moments j = 0..K-1, real and imaginary parts
    for j in range(K):
        for k in range(K):
            V[j, k] = mp.cos(j * ts[k])
        rhs[j] = m[j]
    for j in range(1, K):
        for k in range(K):
            V[K - 1 + j, k] = mp.sin(j * ts[k])
    w, res = mp.qr_solve(V, rhs)
    bK = 2 * mp.fsum(w[k] * mp.cos(K * ts[k]) for k in range(K))
    return ts, [w[k] for k in range(K)], bK, max(abs(x - 1) for x in mods)


def toeplitz_det(c, K):
    m = moments(c, K)
    T = mp.matrix(K + 1, K + 1)
    for i in range(K + 1):
        for j in range(K + 1):
            T[i, j] = m[abs(i - j)]
    return mp.det(T)


# ---- linear-response ring ----
RV = None  # residual harmonics of the reference density (set by callers for M0 < M)


def lr_g(L, c, ORB, v, Mp):
    """harmonic coefficients g_i (cos 2it), i = 1..Mp, of the linear response of the KKT function to h = b - c (plus RV)."""
    h = [bj(ORB, v, 2 * j) - (c[j - 1] if j - 1 < len(c) else 0) for j in range(1, Mp + 1)]
    r = RV if RV is not None else [0] * Mp
    return [
        r[i - 1] + mp.fsum(L[i - 1, j - 1] * h[j - 1] for j in range(1, Mp + 1)) for i in range(1, Mp + 1)
    ]


def gfun(g, t):
    return mp.fsum(g[i - 1] * mp.cos(2 * i * t) for i in range(1, len(g) + 1))


def gder(g, t):
    return -mp.fsum(2 * i * g[i - 1] * mp.sin(2 * i * t) for i in range(1, len(g) + 1))


def lr_ring(L, c, K, PHASE, Mp, v0):
    ORB = layout(K, PHASE)
    nG = ORB.count("G")

    def reps(v):
        r = []
        g = 0
        for o in ORB:
            if o == "X":
                r.append(mp.mpf(0))
            elif o == "Y":
                r.append(mp.pi / 2)
            else:
                r.append(v[len(ORB) + g])
                g += 1
        return r

    def F(v):
        g = lr_g(L, c, ORB, v, Mp)
        r = reps(v)
        D0 = gfun(g, r[0])
        out = [mp.fsum(atoms(ORB, v)[1]) - 1] + [gfun(g, t) - D0 for t in r[1:]]
        out += [gder(g, v[len(ORB) + k]) for k in range(nG)]
        return out

    v, n = newton(F, v0, nO=len(ORB), nit=30, damp=False)
    g = lr_g(L, c, ORB, v, Mp)
    D0 = gfun(g, reps(v)[0])
    th, w = atoms(ORB, v)
    # g - D0 = Pi(t) q(t) with Pi = prod_k (1 - cos(t - t_k)) >= 0; validity <=> max q <= 0. Sample q off the atoms.
    NS = 720
    qs = []
    for s in range(NS):
        t = mp.pi / 2 * (s + mp.mpf(1) / 3) / NS
        Pi = mp.fprod(1 - mp.cos(t - tk) for tk in th)
        qs.append(((gfun(g, t) - D0) / Pi, t))
    qmax, tmax = max(qs, key=lambda x: x[0])
    return ORB, v, n, g, qmax, tmax
