"""q653: Amendment 973. The 2D bulk law with its curvature term, fitted to ring depths.
Model M_c: dN/dd = rho, drho/dd = (s0/rho^2) [f1(rho) - c (1/(2 pi)) arctan(4 pi rho/(A - d))], f1 = the derived 1D law to rho^-10 (933).
Two free constants (N and rho at the first fitted ring); c fixed at 0 (M0, the 1D law) or 1 (M1, derived) or free (c_fit).
Usage: py q653_bulk2d_fit.py control              (1D capacity half-line with a fake radius A = 30: true c = 0)
       py q653_bulk2d_fit.py synth                (data made by M1 itself at A = 30: true c = 1)
       py q653_bulk2d_fit.py 2d FILE A ...        (2D states; rings with depth >= DMIN and radius >= RMIN, default 6)
"""

import sys, os, json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

PI = np.pi
s0 = 1 / (16 * PI**2)
k2 = 1 / (8 * PI**2)
k4 = 1 / (64 * PI**4) - 1 / (96 * PI**2)
c6 = 1 / (512 * PI**6) - 13 / (768 * PI**4)
c8 = 1 / (4096 * PI**8) - 17 / (1536 * PI**6) - 1 / (3072 * PI**4)
c10 = 1 / (32768 * PI**10) - 29 / (6144 * PI**8) - 109 / (368640 * PI**6)
f1 = lambda r: 1 + k2 / r**2 + k4 / r**4 + c6 / r**6 + c8 / r**8 + c10 / r**10
DMIN = float(os.environ.get("DMIN", 6.0))
RMIN = float(os.environ.get("RMIN", 6.0))
FORM = os.environ.get("FORM", "arctan")


def curve(A, c, d0, N0, rho0, dq):
    g = (lambda x: x) if FORM == "lin" else np.arctan  # FORM=lin: the alternative 2 rho/r, no arctan
    if FORM == "corr":  # 1007: arctan at r + S/rho, S = rho^2 rho'/s0 (one step)

        def rhs(d, y):
            r = A - d
            S = f1(y[1]) - c * np.arctan(4 * PI * y[1] / r) / (2 * PI)
            return [
                y[1],
                s0 / y[1] ** 2 * (f1(y[1]) - c * np.arctan(4 * PI * y[1] / (r + S / y[1])) / (2 * PI)),
            ]

    else:
        rhs = lambda d, y: [y[1], s0 / y[1] ** 2 * (f1(y[1]) - c * g(4 * PI * y[1] / (A - d)) / (2 * PI))]
    s = solve_ivp(rhs, (d0, dq[-1]), [N0, rho0], t_eval=dq, rtol=1e-11, atol=1e-13, method="DOP853")
    return s.y[0]


def fit(A, d, N, cfree=None):
    """least squares in N at the given depths; returns (rms, params)."""
    d0 = d[0]
    rho_g = (N[-1] - N[0]) / (d[-1] - d[0])
    if cfree is None:
        f = lambda p: curve(A, p[2], d0, p[0], p[1], d) - N
        p0 = [N[0], rho_g, 0.5]
    else:
        f = lambda p: curve(A, cfree, d0, p[0], p[1], d) - N
        p0 = [N[0], rho_g]
    r = least_squares(f, p0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return float(np.sqrt(np.mean(r.fun**2))), r.x


def report(tag, A, d, N):
    r0, _ = fit(A, d, N, 0.0)
    r1, _ = fit(A, d, N, 1.0)
    rc, p = fit(A, d, N)
    print(
        f"{tag}: {len(d)} rings, depth {d[0]:.2f}-{d[-1]:.2f}; rms M0 (1D law) {r0:.3e}, M1 (derived curvature) {r1:.3e}, "
        f"c free: c_fit = {p[2]:.4f}, rms {rc:.3e}",
        flush=True,
    )
    return r0, r1, p[2]


if sys.argv[1] == "control":
    d = np.array(json.load(open("c50_capc_D110.json"))["stages"][-1]["d"])
    N = np.arange(len(d)) + 0.5
    for A in (20.0, 25.0, 30.0):
        m = (d >= DMIN) & (A - d >= RMIN)
        report(f"CONTROL 1D half-line, fake A = {A}", A, d[m], N[m])
elif sys.argv[1] == "synth":
    d = np.array(json.load(open("c50_capc_D110.json"))["stages"][-1]["d"])
    N = np.arange(len(d)) + 0.5
    for A in (20.0, 25.0, 30.0):
        m = (d >= DMIN) & (A - d >= RMIN)
        dd = d[m]
        Ns = curve(A, 1.0, dd[0], N[m][0], 0.54, dd)  # synthetic M1 data at the same depths
        report(f"SYNTH c = 1, A = {A}", A, dd, Ns)
else:
    st = json.load(open(sys.argv[2]))["states"]
    As = [float(x) for x in sys.argv[3:]]
    for s in st:
        if not any(abs(s["A"] - a) < 1e-9 for a in As):
            continue
        A = s["A"]
        r = np.sort(np.array(s["r"]))[::-1]
        d = A - r
        N = np.arange(len(r)) + 0.5
        m = (d >= DMIN) & (r >= RMIN)
        report(f"2D A = {A} ({''.join(s['types'])})", A, d[m], N[m])
