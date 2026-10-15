"""q1608 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 62): the continuum-integral version of the clean free-block certificates N = 30, 60, 90 and 120: the integration window widened to 25 below the deepest atom and 25 above the wall (as q1597a for the polluted N = 30 and q1599a for the polluted N = 60).
Readings 98 and 99 widened the window for blocks whose level atom was polluted; reading 100 corrected the geometry (NOUT = N + 80, KC = N + 50) and reading 102 certified the clean blocks with the window of 14 and 15, where the window tail of the wall row, 4.3e-49, times the norm of C limits the continuum statement (1.1e-24 at N = 120). Here the start is the stored clean solution (q1601_z_N<N>.json), the arb Newton iteration of q1598 re-solves it with the wide window (own damped version, tolerance 1e-70), the solution is stored in q1608_z_N<N>.json, and certify() of q1598 runs the Krawczyk test (radii 1e-60, 1e-40, 1e-35, 1e-30, the 2C control, the finite-difference control). The strip and the Poisson bound are those of readings 94 and 98 and 99 (insensitive to the state at the 1e-10 level; they were computed for the N = 10, 30 and 60 states): trapezoid error 5.2e-119 at the spacing 0.025, window tail beyond 25 about 1e-135.
Predictions written before the run:
 P1 for each of the four blocks the first depth with the wide window differs from the stored clean value (window 15) by less than 1e-47;
 P2 the Krawczyk test passes at r = 1e-30 for N = 30, 60 and 90 and at r = 1e-35 for N = 120;
 P3 the largest component of C F(z*) of the wide-window block N = 120 is below 1e-58 (the window-15 value is 6.05e-59);
 P4 the 2C control fails for every block.
Usage: py q1608_wide_window_clean.py run (about 30 minutes in the background, one process), then py q1608_check.py.
"""

import os, sys, json, time, types

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
os.environ.setdefault("Q1583_NIN", "30")
src = (
    open(os.path.join(here, "q1583_ball_rows.py"), encoding="utf-8")
    .read()
    .replace("ctx.prec = 170", "ctx.prec = int(os.environ.get('Q1594_PREC', '300'))")
)
assert "Q1594_PREC" in src
src = src.replace(
    "ymin = -(float(self.dout[-1]) + 14.0); n = int((15.0 - ymin) / 0.025) + 1",
    "ymin = -(float(self.dout[-1]) + PADLO); n = int((PADHI - ymin) / 0.025) + 1",
)
assert "PADLO" in src
m = types.ModuleType("q1583m")
m.PADLO = 25.0
m.PADHI = 25.0
m.__file__ = os.path.join(here, "q1583_ball_rows.py")
argv = sys.argv
sys.argv = ["x"]
exec(compile(src, "q1583_ball_rows_prec.py", "exec"), m.__dict__)
sys.argv = argv
from flint import arb, arb_mat, ctx
import numpy as np

C0, NU, D0 = 0.0065, -0.093773586, 0.001159841


def up(b):
    return b.abs_upper()


def zfile(N):
    return os.path.join(here, "q1608_z_N%d.json" % N)


def zfile_narrow(N):
    return os.path.join(here, "q1601_z_N%d.json" % N)


def save_z(N, z):
    json.dump([t.str(90, radius=False) for t in z], open(zfile(N), "w"))


def load_z(N):
    return [arb(s) for s in json.load(open(zfile(N)))]


def newton_own(WA, z, n, tol, maxit):
    hist = []
    for it in range(maxit):
        r, lq, C = WA.resid(z)
        rn = max(abs(float(t.mid())) for t in r)
        hist.append(rn)
        if rn < tol:
            break
        J = WA.jac(z)
        dz = J.solve(arb_mat(n, 1, r))
        lam = arb(1)
        for k in range(8):
            zt = [z[i] - lam * dz[i, 0].mid() for i in range(n)]
            rt = max(abs(float(t.mid())) for t in WA.resid(zt)[0])
            if rt < rn or it == 0 and k == 7:
                break
            lam = lam / 2
        z = zt
        print("  Newton update", it + 1, "residual before %.2e" % rn, "damping", float(lam), flush=True)
    return z, hist


def formal_tail(N0, N):
    kk = np.arange(N0 + 1, N + 1)
    r0 = m.solve_rho(kk + NU)
    nua = arb(repr(NU))
    d0a = arb(repr(D0))
    ca = arb(repr(C0))
    rho = [m.solve_rho_a(arb(int(k)) + nua, r0[i]) for i, k in enumerate(kk)]
    return [m.dlaw_a(r) + d0a for r in rho], [ca / r for r in rho]


def set_block(N):
    m.NIN = N
    m.NOUT = N + 80
    m.KC = N + 50


def stage_30():
    if os.path.exists(zfile(30)):
        return
    set_block(30)
    Wd = m.Wall(NU, D0, C0)
    zd, itd, nrd = Wd.solve(m.start(NU, D0, C0))
    WA = m.WallB(NU, D0, C0)
    z, hist = m.newton_arb(WA, zd, tol=1e-70, maxit=30)
    print("N = 30 solved, residual", hist[-1], flush=True)
    save_z(30, z)


def stage_next(N0, N):
    """solve the block N from the stored solution of the block N0"""
    if os.path.exists(zfile(N)):
        return
    zp = load_z(N0)
    set_block(N)
    WA = m.WallB(NU, D0, C0)
    dt, wt = formal_tail(N0, N)
    z0 = zp[:N0] + dt + zp[N0 : 2 * N0] + wt + [zp[2 * N0]]
    t0 = time.time()
    z, hist = newton_own(WA, z0, 2 * N + 1, 1e-70, 40)
    print(
        "N = %d: arb Newton residuals" % N,
        ["%.2e" % h for h in hist],
        "time",
        round(time.time() - t0, 1),
        flush=True,
    )
    save_z(N, z)
    json.dump(dict(hist=[float(h) for h in hist]), open(os.path.join(here, "q1608_hist_N%d.json" % N), "w"))


def krawczyk(WA, z, C, CF, r, n):
    rb = arb(repr(r))
    Z = [z[i] + arb(0, rb) for i in range(n)]
    JZ = WA.jac(Z)
    CJ = C * JZ
    M = arb_mat(n, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = (1 if i == j else 0) - CJ[i, j]
    delta = arb_mat(n, 1, [arb(0, rb)] * n)
    Md = M * delta
    Kd = [-CF[i, 0] + Md[i, 0] for i in range(n)]
    q = max(float(sum((up(M[i, j]) for j in range(n)), arb(0)).abs_upper()) for i in range(n))
    rr = up(arb(0, rb))
    inside = all(bool(up(Kd[i]) < rr) for i in range(n))
    return inside, q, Kd


def certify(N):
    out_path = os.path.join(here, "q1608_res_N%d.json" % N)
    if os.path.exists(out_path):
        return
    t0 = time.time()
    n = 2 * N + 1
    set_block(N)
    WA = m.WallB(NU, D0, C0)
    z = load_z(N)
    F, lq, C00 = WA.resid(z)
    Fv = arb_mat(n, 1, F)
    J0 = WA.jac(z)
    Jm = arb_mat(n, n, [J0[i, j].mid() for i in range(n) for j in range(n)])
    Ci = Jm.inv()
    C = arb_mat(n, n, [Ci[i, j].mid() for i in range(n) for j in range(n)])
    CF = C * Fv
    out = dict(
        N=N,
        prec=int(ctx.prec),
        d1_50=z[0].str(60, radius=False),
        Frad=max(float(t.rad()) for t in F),
        Fmax=max(float(up(t)) for t in F),
        CFmax=max(float(up(CF[i, 0])) for i in range(n)),
        Cnorm_inf=max(float(sum((up(C[i, j]) for j in range(n)), arb(0)).abs_upper()) for i in range(n)),
        Jrad=max(float(J0[i, j].rad()) for i in range(n) for j in range(n)),
    )
    eps = arb("1e-30")
    worst = 0.0
    for j in (0, 1, 10, 30, N, N + 1, N + 30, 2 * N - 1, 2 * N):
        zp = list(z)
        zp[j] = zp[j] + eps
        Fp = WA.resid(zp)[0]
        for i in range(n):
            worst = max(
                worst,
                float(abs(((Fp[i] - F[i]) / eps).mid() - J0[i, j].mid()))
                / max(1.0, float(abs(J0[i, j].mid()))),
            )
    out["fd_vs_analytic_rel"] = worst
    print(
        "N = %d: finite-difference control %.3e; ||C||_inf %.3e; largest component of C F %.3e"
        % (N, worst, out["Cnorm_inf"], out["CFmax"]),
        flush=True,
    )
    rows = []
    for e in (-60, -40, -35, -30):
        r = float("1e%d" % e)
        inside, q, Kd = krawczyk(WA, z, C, CF, r, n)
        rows.append(
            dict(r=r, inside=inside, q=q, K1_up=float(up(Kd[0])), Kmax_up=max(float(up(t)) for t in Kd))
        )
        print(
            "  N = %d r %.0e: inside %s, norm %.3e, first-depth bound %.3e"
            % (N, r, inside, q, rows[-1]["K1_up"]),
            flush=True,
        )
    out["rows"] = rows
    C2 = arb_mat(n, n, [2 * C[i, j] for i in range(n) for j in range(n)])
    ins2, q2, _ = krawczyk(WA, z, C2, C2 * Fv, 1e-40, n)
    out["ctrl_2C"] = dict(inside=ins2, q=q2)
    print("  N = %d control 2C at r = 1e-40: inside %s, norm %.6f" % (N, ins2, q2), flush=True)
    out["time"] = time.time() - t0
    json.dump(out, open(out_path, "w"), indent=1)


def run():
    t0 = time.time()
    for N in (30, 60, 90, 120):
        if not os.path.exists(zfile(N)):
            set_block(N)
            WA = m.WallB(NU, D0, C0)
            z0 = [arb(s) for s in json.load(open(zfile_narrow(N)))]
            z, hist = newton_own(WA, z0, 2 * N + 1, 1e-70, 8)
            print(
                "N = %d wide window: arb Newton residuals" % N,
                ["%.2e" % h for h in hist],
                "time",
                round(time.time() - t0, 1),
                flush=True,
            )
            save_z(N, z)
            json.dump(
                dict(hist=[float(h) for h in hist]), open(os.path.join(here, "q1608_hist_N%d.json" % N), "w")
            )
        certify(N)
        print("certificate", N, "ready", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    run()
