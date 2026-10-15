"""gpu_walker: a COPY-by-subclassing of the production walker (src/q1411_fast_walk.py, FastGB band solver) whose numerical kernel runs on the GPU:
  residual: 320-bit fixed-point multi-limb kernels (c30_fx_residual.FXRes), agreement with arb 192 bits 1.5e-55
  Jacobian: double-double kernels (c10_dd_jacobian.DDJac), agreement with arb 1.4e-31 (relative to the row maximum)
  linear solves: dd LU on the GPU.
Walker logic (walk_family, locate, after_birth, after_split, Newton acceptance rule, events) is the production code, unchanged, except
signal(), the pitchfork bordered Newton (bordered_newton, dd bordered system) and the polynomial/Hermite predictor which stays as in production.
Usage: py gpu_walker.py STATE.json OUTDIR A_END HMAX [H]    (same arguments as q1411_fast_walk.py)"""

import sys, os, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, HERE)
import numpy as np
from flint import arb, arb_mat, ctx
import q1411_fast_walk as QW
from q1414_fast_gauss_band import FastGB
from c10_dd_jacobian import DDJac, arb_to_dd
from c30_fx_residual import FXRes

CNT = dict(J_builds=0, res_only=0, tJ=0.0, tres=0.0, solves=0, tsolve=0.0)


class _Sol:
    def __init__(self, arbs):
        self.a = arbs

    def __getitem__(self, ij):
        return self.a[ij[0]]


class DDMat:
    """Jacobian kept as dd numpy array (n, n, 2) with a GPU LU solve; interface used by the unmodified FastG.newton / Walker.tangent."""

    def __init__(self, F, Jd):
        self.F = F
        self.Jd = Jd

    def nrows(self):
        return self.Jd.shape[0]

    def solve(self, B):
        n = self.Jd.shape[0]
        rhs = np.array([arb_to_dd(B[i, 0]) for i in range(n)])
        t0 = time.perf_counter()
        x = self.F.dd.solve(self.Jd, rhs)
        CNT["tsolve"] += time.perf_counter() - t0
        CNT["solves"] += 1
        return _Sol([arb(float(x[i, 0])) + arb(float(x[i, 1])) for i in range(n)])


class GPUF(FastGB):
    def __init__(self, prec=192, H=0.05, pad=9):
        super().__init__(prec, H=H, pad=pad)
        self.fx = FXRes(Hf=H, pad=pad)
        self.dd = DDJac(Hf=H, pad=pad)

    def residual_and_jacobian(self, A, u, w, we, wc, want_J=True, y=None):
        if y is None:
            y = self.grid(A)
        N = len(y)
        A = arb(A.mid()) if False else A
        t0 = time.perf_counter()
        res, s = self.fx.residual(A, u, w, we, wc, y[0], N)
        self.last_sums = s
        if not want_J:
            CNT["res_only"] += 1
            CNT["tres"] += time.perf_counter() - t0
            return res, None
        _, Jd = self.dd.build(A, u, w, we, wc, y[0], N)
        CNT["J_builds"] += 1
        CNT["tJ"] += time.perf_counter() - t0
        return res, DDMat(self, Jd)


def _gpuf_update(self, A, u, w, we, wc, y, r):
    """one Newton correction dv (arb list) for the arb residual list r at the point (u, w, we, wc): dd Jacobian build and dd LU solve on the GPU"""
    N = len(y)
    _, Jd = self.dd.build(A, u, w, we, wc, y[0], N)
    rd = np.array([arb_to_dd(-t) for t in r])
    x = self.dd.solve(Jd, rd)
    return [(arb(float(x[i, 0])) + arb(float(x[i, 1]))).mid() for i in range(len(r))]


GPUF.update = _gpuf_update


class FXFG(FastGB):
    """backend fx: residual, Jacobian and LU solve all in 320-bit fixed point on the GPU (c33_fx_full)"""

    def __init__(self, prec=192, H=0.05, pad=9):
        super().__init__(prec, H=H, pad=pad)
        from c33_fx_full import FXF

        self.fxf = FXF(Hf=H, pad=pad)

    def residual_and_jacobian(self, A, u, w, we, wc, want_J=True, y=None):
        if y is None:
            y = self.grid(A)
        t0 = time.perf_counter()
        res, J = self.fxf.residual_and_jacobian(A, u, w, we, wc, want_J=want_J, y=y)
        if want_J:
            CNT["J_builds"] += 1
            CNT["tJ"] += time.perf_counter() - t0
        else:
            CNT["res_only"] += 1
            CNT["tres"] += time.perf_counter() - t0
        return res, J

    def update(self, A, u, w, we, wc, y, r):
        _, J = self.fxf.residual_and_jacobian(A, u, w, we, wc, want_J=True, y=y)
        sol = J.solve(arb_mat([[-t.mid()] for t in r]))
        return [sol[i, 0].mid() for i in range(J.nrows())]


class GPUWalker(QW.Walker):
    def __init__(self, prec=192, outdir="gpuwalk", H=0.05, pad=9, backend="dd"):
        super().__init__(prec, outdir, H=H, pad=pad)
        self.backend = backend
        self.F = GPUF(prec, H=H, pad=pad) if backend == "dd" else FXFG(prec, H=H, pad=pad)

    def signal(self, A, u, w, we, wc, y):
        """even K: D(0) - C ; odd K: D''(0)  (fixed-point GPU sums; same quantities as Walker.signal)"""
        F = self.F
        N = len(y)
        if self.backend == "fx":
            s, _ = F.fxf.run(A, u, w, we, wc, y[0], N, want_J=False)
        else:
            s = F.fx.sums(A, u, w, we, wc, y[0], N)
        if wc is None:
            return s["centre"][0] - s["DA"]
        return s["centre"][2]

    def bordered_newton(self, A0, u, w, we, eps, steps=40, verbose=False, growth=5.0):
        if self.backend == "fx":
            return self.bordered_newton_fx(A0, u, w, we, eps, steps=steps, verbose=verbose, growth=growth)
        F = self.F
        m = len(u)
        n = 2 * m + 1
        A = arb(A0.mid())
        y = F.grid(A)
        v = [arb(t.mid()) for t in (list(u) + list(w) + [we])]
        dl = arb("1e-25")

        def system(v, A):
            up, wp, wep = v[:m], v[m : 2 * m], v[2 * m]
            r, J = F.residual_and_jacobian(A, up, wp, wep, None, y=y)
            rp, _ = F.residual_and_jacobian(A + dl, up, wp, wep, None, want_J=False, y=y)
            rm, _ = F.residual_and_jacobian(A - dl, up, wp, wep, None, want_J=False, y=y)
            M = np.zeros((n + 1, n + 1, 2))
            M[:n, :n] = J.Jd
            for i in range(n):
                M[i, n] = arb_to_dd((rp[i] - rm[i]) / (2 * dl))
            M[n, 0] = (1.0, 0.0)
            rhs = [-t.mid() for t in r] + [-(v[0] - eps)]
            sc = 1 / eps  # row m scaled by 1/eps (the row D'(u_1) vanishes with u_1)
            for j in range(n + 1):
                x = (arb(float(M[m, j, 0])) + arb(float(M[m, j, 1]))) * sc
                M[m, j] = arb_to_dd(x)
            rhs[m] = rhs[m] * sc
            return r, M, rhs

        def dsolve(M, rhs):
            t0 = time.perf_counter()
            x = F.dd.solve(M, np.array([arb_to_dd(t) for t in rhs]))
            CNT["tsolve"] += time.perf_counter() - t0
            CNT["solves"] += 1
            return [(arb(float(x[i, 0])) + arb(float(x[i, 1]))).mid() for i in range(n + 1)]

        nrm = lambda d: max(abs(t.mid()) for t in d)
        r, M, rhs = system(v, A)
        dz = dsolve(M, rhs)
        nd = nrm(dz)
        res = nrm(r)
        k = 0
        while k < steps and nd > arb("1e-40") and res > arb("1e-45"):
            lam = arb(1)
            done = False
            for _ in range(10):
                vn = [v[i] + lam * dz[i] for i in range(n)]
                An = A + lam * dz[n]
                if (
                    all(vn[i] > 0 for i in range(m, 2 * m + 1))
                    and vn[0] > 0
                    and all(vn[i] < vn[i + 1] for i in range(m - 1))
                    and vn[m - 1] < An
                ):
                    rn, Mn, rhsn = system(vn, An)
                    dzn = dsolve(Mn, rhsn)
                    ndn = nrm(dzn)
                    if ndn <= growth * nd or nrm(rn) < arb("1e-40"):
                        v, A, r, dz, nd, res, done = vn, An, rn, dzn, ndn, nrm(rn), True
                        break
                lam /= 2
            if not done:
                break
            k += 1
            if verbose:
                QW.say(
                    f"      bordered it {k}: lam {float(lam):.3g}, residual {float(res):.2e}, next step {float(nd):.2e}, A = {A.str(15, radius=False)}, u_1 = {float(v[0].mid()):.4f}"
                )
        return v[:m], v[m : 2 * m], v[2 * m], A, res, k


def _bordered_newton_fx(self, A0, u, w, we, eps, steps=40, verbose=False, growth=5.0):
    """pitchfork branch of the even system, fixed-point backend: bordered matrix (n+1)^2 from the device Jacobian, border column dF/dA (central difference 1e-25), row u_1 = eps"""
    import cupy as cp
    from c30_fx_residual import to_limbs, from_limbs
    from c33_fx_full import FXMat

    F = self.F
    m = len(u)
    n = 2 * m + 1
    A = arb(A0.mid())
    y = F.grid(A)
    v = [arb(t.mid()) for t in (list(u) + list(w) + [we])]
    dl = arb("1e-25")

    def system(v, A):
        up, wp, wep = v[:m], v[m : 2 * m], v[2 * m]
        r, J = F.residual_and_jacobian(A, up, wp, wep, None, y=y)
        rp, _ = F.residual_and_jacobian(A + dl, up, wp, wep, None, want_J=False, y=y)
        rm, _ = F.residual_and_jacobian(A - dl, up, wp, wep, None, want_J=False, y=y)
        Jh = cp.asnumpy(J.J)
        M = np.zeros((n + 1, n + 1, Jh.shape[2]), dtype=np.uint32)
        M[:n, :n] = Jh
        for i in range(n):
            M[i, n] = to_limbs((rp[i] - rm[i]) / (2 * dl))
        M[n, 0] = to_limbs(arb(1))
        rhs = [-t.mid() for t in r] + [-(v[0] - eps)]
        sc = 1 / eps
        for j in range(n + 1):
            M[m, j] = to_limbs(from_limbs(M[m, j]) * sc)
        rhs[m] = rhs[m] * sc
        return r, M, rhs

    def dsolve(M, rhs):
        mat = FXMat(F.fxf, cp.asarray(M), n + 1)
        t0 = time.perf_counter()
        x = mat.solve_limbs(np.stack([to_limbs(t) for t in rhs]))
        CNT["tsolve"] += time.perf_counter() - t0
        CNT["solves"] += 1
        return [from_limbs(x[i]) for i in range(n + 1)]

    nrm = lambda d: max(abs(t.mid()) for t in d)
    r, M, rhs = system(v, A)
    dz = dsolve(M, rhs)
    nd = nrm(dz)
    res = nrm(r)
    k = 0
    while k < steps and nd > arb("1e-40") and res > arb("1e-45"):
        lam = arb(1)
        done = False
        for _ in range(10):
            vn = [v[i] + lam * dz[i] for i in range(n)]
            An = A + lam * dz[n]
            if (
                all(vn[i] > 0 for i in range(m, 2 * m + 1))
                and vn[0] > 0
                and all(vn[i] < vn[i + 1] for i in range(m - 1))
                and vn[m - 1] < An
            ):
                rn, Mn, rhsn = system(vn, An)
                dzn = dsolve(Mn, rhsn)
                ndn = nrm(dzn)
                if ndn <= growth * nd or nrm(rn) < arb("1e-40"):
                    v, A, r, dz, nd, res, done = vn, An, rn, dzn, ndn, nrm(rn), True
                    break
            lam /= 2
        if not done:
            break
        k += 1
        if verbose:
            QW.say(
                f"      bordered it {k}: lam {float(lam):.3g}, residual {float(res):.2e}, next step {float(nd):.2e}, A = {A.str(15, radius=False)}, u_1 = {float(v[0].mid()):.4f}"
            )
    return v[:m], v[m : 2 * m], v[2 * m], A, res, k


GPUWalker.bordered_newton_fx = _bordered_newton_fx


if __name__ == "__main__":
    st, outdir, Aend, hmax = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    H = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    ctx.prec = 192
    W = GPUWalker(192, outdir, H=H, backend=os.environ.get("GW_BACKEND", "dd"))
    A, u, w, we, wc = QW.load(st, W.F)
    t0 = time.perf_counter()
    u, w, we, wc, res, k = W.solve_at(A, u, w, we, wc)
    QW.say(f"start re-solved: residual {float(res):.1e}")
    out = W.run(A, (u, w, we, wc), Aend, hmax=hmax)
    CNT["wall_s"] = time.perf_counter() - t0
    CNT["states_saved"] = W.nsave
    CNT["returned"] = out[0]
    CNT["A_final"] = float(out[1])
    print("COUNTERS", json.dumps(CNT))
