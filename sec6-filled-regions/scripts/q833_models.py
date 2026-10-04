"""q833 (03 Oct 2026): TEXT COPY of q832 with two more modes: quartet (X pair mass mx/2 at (+-a, 0) and Y pair my/2 at (0, +-b),
equations r(a,0) = C, d/dx1 r(a,0) = 0, r(0,b) = C, d/dx2 r(0,b) = 0, as q823) and inner2 (inner formal density with two harmonics,
g = 1 + d_1 cos 2th + d_2 cos 4th on rho = a_0 + a_1 cos 2th + a_2 cos 4th; harmonics 0, 2, 4 of r - C and of the radial derivative
vanish; the q825-readable print lists [a_0, a_1, a_2] and [d_1, d_2]). START: quartet a b mx my; inner2 a0 mi. q832's docstring follows.
q832 (03 Oct 2026, BACKLOG 64): TEXT COPY of q830 with FUNC cap added (capacity: D(q) = -int phi(y - q) log p(y) dy, constants
dropped, as q819/q820) and the outer curve on the mixed geometry of q790/q831 (env LAM, 0 = ellipse): z(t) = R e^(it) + eta (e^(-it)
+ LAM e^(3it)), R = rp - eps/2, eta = eps/(2 (1 + LAM)). q830's docstring follows.
q830 (03 Oct 2026, BACKLOG 63): TEXT COPY of q828 (LFP interior models) with a FUNC switch (env FUNC lfp | rd). FUNC rd: the
rate-distortion / NPMLE dual of q829 (source N(0, s^2 I), env SRC_S, default 2): KKT function D(q) = int phi(y - q) P_src/p dy,
C = its mean on the ellipse (= 1 at the optimum); every 'r' below reads as D for rd. q828's docstring follows.
q828 (03 Oct 2026, BACKLOG 62): the LFP (least favourable prior, bounded normal mean in 2D, squared loss, unit noise) interior
models on the filled ellipse, the counterparts of q818/q819 (centre atom), q820/q821 (pair) and q822 (inner formal density) for
capacity. Prior = (1 - m_in) x formal density f = 1 + sum c_2k cos 2kt on the ellipse (rp cos t, rm sin t) + an inner part of mass m_in:
  centre: an atom at 0;  pair: atoms m_in/2 at (+-a, 0);  inner: a formal density g = 1 + d_1 cos 2th on rho = a_0 + a_1 cos 2th.
Posterior mean delta = N/p; risk r(q) = int phi(y - q) |delta(y) - q|^2 dy (q790 FUNC lfp, q827), polar Gauss-Legendre x trapezoid grid.
Equations: harmonics 2..2M of r on the ellipse vanish; C = mean of r on the ellipse (the Bayes risk); and
  centre: r(0) = C;  pair: r(a, 0) = C, d/dx1 r(a, 0) = 0;  inner: harmonics 0, 2 of r - C and of the radial derivative of r on the
  inner curve vanish.
Reports: centre: m_in, C, Hessian of r at 0 (H11, H22); pair: a, m_in, C, Hessian at (a, 0), the largest r - C on the minor axis
(0, b), b in [0, 1.5], and its b, r(0) - C; inner: m_in, C, "inner radius a_0..: [..]; inner d_1..: [..]" (q825-readable).
Usage: py q828_lfp_models.py MODE RP "rm1,rm2,..." [START...] [M] ; MODE centre (START m0), pair (START a m0), inner (START a0 m0)
Env: NR (160), NA (192), RMAX (14), NI (64)."""

import sys, os, math
import numpy as np

MODE, RP = sys.argv[1], float(sys.argv[2])
RMS = [float(x) for x in sys.argv[3].split(",")]
nst = {"centre": 1, "pair": 2, "inner": 2, "quartet": 4, "inner2": 2}[MODE]
ST = [float(x) for x in sys.argv[4 : 4 + nst]]
M = int(sys.argv[4 + nst]) if len(sys.argv) > 4 + nst else 6
NR = int(os.environ.get("NR", "160"))
NA = int(os.environ.get("NA", "192"))
RMAX = float(os.environ.get("RMAX", "14"))
NI = int(os.environ.get("NI", "64"))
xr, wr = np.polynomial.legendre.leggauss(NR)
rho = (xr + 1) * RMAX / 2
wrho = wr * RMAX / 2
ang = 2 * np.pi * np.arange(NA) / NA
Y1 = (rho[:, None] * np.cos(ang)[None, :]).ravel()
Y2 = (rho[:, None] * np.sin(ang)[None, :]).ravel()
WY = (wrho[:, None] * rho[:, None] * np.full((1, NA), 2 * np.pi / NA)).ravel()
t = 2 * np.pi * np.arange(NA) / NA
cosk = np.array([np.cos(2 * k * t) for k in range(1, M + 1)])
th = 2 * np.pi * np.arange(NI) / NI
c2 = np.cos(2 * th)
c4 = np.cos(4 * th)
FUNC = os.environ.get("FUNC", "lfp")
S2 = float(os.environ.get("SRC_S", "2")) ** 2
LAM = float(os.environ.get("LAM", "0"))
PSRC = np.exp(-(Y1**2 + Y2**2) / (2 * S2)) / (2 * np.pi * S2)


def kern(q1, q2):
    return np.exp(-((Y1[:, None] - q1[None, :]) ** 2 + (Y2[:, None] - q2[None, :]) ** 2) / 2) / (2 * np.pi)


def inner_pts(u):
    if MODE == "centre":
        return np.array([0.0]), np.array([0.0]), np.array([u[M]])
    if MODE == "pair":
        m0, a = u[M], u[M + 1]
        return np.array([a, -a]), np.array([0.0, 0.0]), np.array([m0 / 2, m0 / 2])
    if MODE == "quartet":
        mx, my, a, b = u[M : M + 4]
        return (
            np.array([a, -a, 0.0, 0.0]),
            np.array([0.0, 0.0, b, -b]),
            np.array([mx / 2, mx / 2, my / 2, my / 2]),
        )
    if MODE == "inner2":
        d1, d2, a0, a1, a2, mi = u[M : M + 6]
        rr = a0 + a1 * c2 + a2 * c4
        g = 1 + d1 * c2 + d2 * c4
        return rr * np.cos(th), rr * np.sin(th), mi * g / NI
    d1, a0, a1, mi = u[M], u[M + 1], u[M + 2], u[M + 3]
    rr = a0 + a1 * c2
    g = 1 + d1 * c2
    return rr * np.cos(th), rr * np.sin(th), mi * g / NI


def risks(Kq, q1, q2, d1, d2):
    """r at points q from the kernel matrix Kq = phi(y - q) (Ny x nq)."""
    if FUNC in ("rd", "cap"):
        return Kq.T @ (WY * d1)  # d1 = P_src/p for rd, -log p for cap
    A = WY * (d1**2 + d2**2)
    return (
        Kq.T @ A - 2 * q1 * (Kq.T @ (WY * d1)) - 2 * q2 * (Kq.T @ (WY * d2)) + (q1**2 + q2**2) * (Kq.T @ WY)
    )


def state(u, S):
    Kc, z1, z2 = S
    f = 1 + u[:M] @ cosk
    i1, i2, iw = inner_pts(u)
    min_ = iw.sum()
    wo = (1 - min_) * f / NA
    Ki = kern(i1, i2)
    p = Kc @ wo + Ki @ iw
    if FUNC == "rd":
        return PSRC / p, None, (i1, i2, Ki)
    if FUNC == "cap":
        return -np.log(p), None, (i1, i2, Ki)
    n1 = Kc @ (z1 * wo) + Ki @ (i1 * iw)
    n2 = Kc @ (z2 * wo) + Ki @ (i2 * iw)
    return n1 / p, n2 / p, (i1, i2, Ki)


def r_at(q1, q2, d1, d2):
    q1 = np.atleast_1d(np.asarray(q1, float))
    q2 = np.atleast_1d(np.asarray(q2, float))
    return risks(kern(q1, q2), q1, q2, d1, d2)


def model(u, S):
    Kc, z1, z2 = S
    d1, d2, (i1, i2, Ki) = state(u, S)
    r = risks(Kc, z1, z2, d1, d2)
    C = r.mean()
    h = [2 * np.mean(r * cosk[k]) for k in range(M)]
    if MODE == "centre":
        e = [r_at(0.0, 0.0, d1, d2)[0] - C]
    elif MODE == "pair":
        a = u[M + 1]
        dd = 1e-4
        e = [
            r_at(a, 0.0, d1, d2)[0] - C,
            (r_at(a + dd, 0.0, d1, d2)[0] - r_at(a - dd, 0.0, d1, d2)[0]) / (2 * dd),
        ]
    elif MODE == "quartet":
        a, b = u[M + 2], u[M + 3]
        dd = 1e-4
        e = [
            r_at(a, 0.0, d1, d2)[0] - C,
            (r_at(a + dd, 0.0, d1, d2)[0] - r_at(a - dd, 0.0, d1, d2)[0]) / (2 * dd),
            r_at(0.0, b, d1, d2)[0] - C,
            (r_at(0.0, b + dd, d1, d2)[0] - r_at(0.0, b - dd, d1, d2)[0]) / (2 * dd),
        ]
    elif MODE == "inner2":
        ri = risks(Ki, i1, i2, d1, d2) - C
        a0, a1, a2 = u[M + 2], u[M + 3], u[M + 4]
        dd = 1e-5
        rr = a0 + a1 * c2 + a2 * c4
        rp_ = r_at((rr + dd) * np.cos(th), (rr + dd) * np.sin(th), d1, d2)
        rm_ = r_at((rr - dd) * np.cos(th), (rr - dd) * np.sin(th), d1, d2)
        dr = (rp_ - rm_) / (2 * dd)
        e = [np.mean(ri), np.mean(ri * c2), np.mean(ri * c4), np.mean(dr), np.mean(dr * c2), np.mean(dr * c4)]
    else:
        ri = risks(Ki, i1, i2, d1, d2) - C
        a0, a1 = u[M + 1], u[M + 2]
        dd = 1e-5
        rr = a0 + a1 * c2
        rp_ = r_at((rr + dd) * np.cos(th), (rr + dd) * np.sin(th), d1, d2)
        rm_ = r_at((rr - dd) * np.cos(th), (rr - dd) * np.sin(th), d1, d2)
        dr = (rp_ - rm_) / (2 * dd)
        e = [np.mean(ri), np.mean(ri * c2), np.mean(dr), np.mean(dr * c2)]
    return np.array(h + e), C, d1, d2


def ok(u):
    if MODE == "centre":
        return 0 <= u[M] < 0.9
    if MODE == "pair":
        return 0 <= u[M] < 0.9 and u[M + 1] > 1e-6
    if MODE == "quartet":
        return u[M] > 0 and u[M + 1] > 0 and u[M + 2] > 1e-3 and u[M + 3] > 1e-3
    if MODE == "inner2":
        return 0 < u[M + 5] < 0.6 and u[M + 2] > 0.02
    return 0 < u[M + 3] < 0.6 and u[M + 1] > 0.02


def newton(u, S):
    n = len(u)
    for it in range(100):
        F = model(u, S)[0]
        if np.max(np.abs(F)) < 1e-12:
            break
        J = np.zeros((n, n))
        for j in range(n):
            up = u.copy()
            up[j] += 1e-7
            J[:, j] = (model(up, S)[0] - F) / 1e-7
        step = np.linalg.lstsq(J, -F, rcond=None)[0]
        lam = 1.0
        while lam > 1e-6:
            un = u + lam * step
            if ok(un):
                Fn = model(un, S)[0]
                if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < np.max(np.abs(F)):
                    u = un
                    break
            lam /= 2
        else:
            break
    return u, np.max(np.abs(model(u, S)[0]))


if MODE == "centre":
    u = np.zeros(M + 1)
    u[M] = ST[0]
elif MODE == "pair":
    u = np.zeros(M + 2)
    u[M + 1], u[M] = ST[0], ST[1]
elif MODE == "quartet":
    u = np.zeros(M + 4)
    u[M + 2], u[M + 3], u[M], u[M + 1] = ST[0], ST[1], ST[2], ST[3]
elif MODE == "inner2":
    u = np.zeros(M + 6)
    u[M + 2], u[M + 5] = ST[0], ST[1]
else:
    u = np.zeros(M + 4)
    u[M + 1], u[M + 3] = ST[0], ST[1]
for rm in RMS:
    eps_ = RP - rm
    R_ = RP - eps_ / 2
    et_ = eps_ / (2 * (1 + LAM))
    z1 = (R_ + et_) * np.cos(t) + LAM * et_ * np.cos(3 * t)
    z2 = (R_ - et_) * np.sin(t) + LAM * et_ * np.sin(3 * t)
    S = (kern(z1, z2), z1, z2)
    u, res = newton(u, S)
    F, C, d1, d2 = model(u, S)
    fmin = (1 + u[:M] @ cosk).min()
    hd = 1e-3
    if MODE == "centre":
        c0 = r_at(0.0, 0.0, d1, d2)[0]
        H11 = (r_at(hd, 0.0, d1, d2)[0] + r_at(-hd, 0.0, d1, d2)[0] - 2 * c0) / hd**2
        H22 = (r_at(0.0, hd, d1, d2)[0] + r_at(0.0, -hd, d1, d2)[0] - 2 * c0) / hd**2
        print(
            f"{FUNC} rp {RP} rm {rm}: centre mass {u[M]:.10f}; C {C:.12f}; H11 {H11:+.6e}; H22 {H22:+.6e}; residual {res:.1e}; min f {fmin:.3f}",
            flush=True,
        )
    elif MODE == "pair":
        a = u[M + 1]
        c0 = r_at(a, 0.0, d1, d2)[0]
        H11 = (r_at(a + hd, 0.0, d1, d2)[0] + r_at(a - hd, 0.0, d1, d2)[0] - 2 * c0) / hd**2
        H22 = (r_at(a, hd, d1, d2)[0] + r_at(a, -hd, d1, d2)[0] - 2 * c0) / hd**2
        bb = np.linspace(0, 1.5, 151)
        Db = r_at(np.zeros_like(bb), bb, d1, d2) - C
        print(
            f"{FUNC} rp {RP} rm {rm}: pair a {a:.6f}, mass {u[M]:.8f}, C {C:.12f}; Hessian at (a,0) {H11:+.6e} {H22:+.6e}; minor-axis max "
            f"r - C {Db.max():+.3e} at b {bb[np.argmax(Db)]:.2f}; r(0) - C {Db[0]:+.3e}; residual {res:.1e}; min f {fmin:.3f}",
            flush=True,
        )
    elif MODE == "quartet":
        mx, my, a, b = u[M : M + 4]
        ca = r_at(a, 0.0, d1, d2)[0]
        cb = r_at(0.0, b, d1, d2)[0]
        Ha22 = (r_at(a, hd, d1, d2)[0] + r_at(a, -hd, d1, d2)[0] - 2 * ca) / hd**2
        Hb11 = (r_at(hd, b, d1, d2)[0] + r_at(-hd, b, d1, d2)[0] - 2 * cb) / hd**2
        D45 = r_at(a * np.cos(np.pi / 4), b * np.sin(np.pi / 4), d1, d2)[0] - C
        D0 = r_at(0.0, 0.0, d1, d2)[0] - C
        print(
            f"{FUNC} rp {RP} rm {rm}: X pair a {a:.6f} mass {mx:.8f}; Y pair b {b:.6f} mass {my:.8f}; C {C:.12f}; H22 at X {Ha22:+.4e}, "
            f"H11 at Y {Hb11:+.4e}; D45 - C {D45:+.3e}; D(0) - C {D0:+.3e}; residual {res:.1e}; min f {fmin:.3f}",
            flush=True,
        )
    elif MODE == "inner2":
        print(
            f"{FUNC} rp {RP} rm {rm}: inner mass {u[M + 5]:.8f}; C {C:.12f}; inner radius a_0..: [{u[M + 2]:.6f}, {u[M + 3]:.6f}, {u[M + 4]:.6f}]; "
            f"inner d_1..: [{u[M]:.6f}, {u[M + 1]:.6f}]; residual {res:.1e}; min f {fmin:.3f}",
            flush=True,
        )
    else:
        print(
            f"{FUNC} rp {RP} rm {rm}: inner mass {u[M + 3]:.8f}; C {C:.12f}; inner radius a_0..: [{u[M + 1]:.6f}, {u[M + 2]:.6f}]; "
            f"inner d_1..: [{u[M]:.6f}]; residual {res:.1e}; min f {fmin:.3f}",
            flush=True,
        )
