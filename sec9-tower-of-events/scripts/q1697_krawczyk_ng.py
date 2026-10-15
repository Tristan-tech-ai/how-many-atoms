# q1697 (10 Oct 2026): copy of q1598_krawczyk_free90_120.py that reads q1696_ball_rows_ng.py and writes q1697_* files. Made by scratch/make_q1696.py.
"""q1598 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 55): Krawczyk certificates for the FREE wall blocks of N = 90 and N = 120 atoms (181 and 241 unknowns), law atoms N+1..N+40 frozen, level atom KC = N + 30, 300 bits, trapezoid spacing 0.025, and the sequence d_1(N) for N = 10, 30, 60, 90, 120.
Readings 96 and 97 certified the free blocks N = 30 and 60 (first depth +7.73277e-12 and +3.14166e-16 when the next 20 and 30 atoms are freed). This script continues the sequence. Each stage starts from the previous solution for the atoms already solved and the formal bulk law for the new atoms (the own damped Newton iteration of q1596), and every solution is stored in q1598_z_N<N>.json with 90 digits, so that a stopped run resumes at the next stage (the N = 30 stage uses the double-precision Newton start of q1595, the N = 60 stage is recomputed because q1596 did not store its solution). The Krawczyk test (as q1596) is run for N = 90 and N = 120 with the radii 1e-60, 1e-40, 1e-35 and 1e-30 and the 2C control; the finite-difference control uses nine columns.
Predictions written before the run (from the registration text):
 P1 both new blocks converge to a residual below 1e-60 within 12 updates;
 P2 the Krawczyk test passes at r = 1e-35 for N = 90 and at r = 1e-40 for N = 120;
 P3 |d_1(90) - d_1(60)| < 3.14e-17 (a tenth of the previous difference);
 P4 |d_1(120) - d_1(90)| < |d_1(90) - d_1(60)|;
 P5 negative control: with C replaced by 2 C the test fails for both blocks at r = 1e-40.
Usage: py q1598_krawczyk_free90_120.py run (one process; hours of CPU are not needed, about an hour in total), then py q1598_check.py.
"""

import os, sys, json, time, types

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
os.environ.setdefault("Q1583_NIN", "30")
src = (
    open(os.path.join(here, "q1696_ball_rows_ng.py"), encoding="utf-8")
    .read()
    .replace("ctx.prec = 170", "ctx.prec = int(os.environ.get('Q1594_PREC', '300'))")
)
assert "Q1594_PREC" in src
m = types.ModuleType("q1696m")
m.__file__ = os.path.join(here, "q1696_ball_rows_ng.py")
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
    return os.path.join(here, "q1697_z_N%d.json" % N)


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
    m.NOUT = N + 40
    m.KC = N + 30


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
    json.dump(dict(hist=[float(h) for h in hist]), open(os.path.join(here, "q1697_hist_N%d.json" % N), "w"))


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
    out_path = os.path.join(here, "q1697_res_N%d.json" % N)
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
    stage_30()
    print("stage 30 ready", round(time.time() - t0, 1), flush=True)
    stage_next(30, 60)
    print("stage 60 ready", round(time.time() - t0, 1), flush=True)
    stage_next(60, 90)
    print("stage 90 ready", round(time.time() - t0, 1), flush=True)
    certify(90)
    print("certificate 90 ready", round(time.time() - t0, 1), flush=True)
    stage_next(90, 120)
    print("stage 120 ready", round(time.time() - t0, 1), flush=True)
    certify(120)
    print("certificate 120 ready", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    run()
