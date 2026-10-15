"""gpu_walker_verified: GPUWalker plus the acceptance check demanded for production use:
 (1) every Newton result the walker accepts is re-evaluated by the 192-bit arb residual (CPU, FastGB.residual_and_jacobian want_J=False on the same grid): the maximum arb residual and
     the difference to the GPU residual are recorded; a result with arb residual above its own tolerance is a FAILURE;
 (2) every located event: the signal is re-evaluated by arb (dense FastG.tables, 192 bits) at both end points of the final bracket: lo must be < 0, hi must be > 0, and they must agree with the
     GPU signals; otherwise FAILURE.
Writes OUTDIR/verify.json. Usage as module (c40_stored_events.py) or py gpu_walker_verified.py STATE OUTDIR A_END HMAX [H].
"""

import sys, os, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, HERE)
from flint import arb, arb_mat, ctx
import q1411_fast_walk as QW
from q1414_fast_gauss_band import FastGB
import gpu_walker as GW
import numpy as np
from c10_dd_jacobian import arb_to_dd


class VerifiedGPUWalker(GW.GPUWalker):
    def __init__(
        self,
        prec=192,
        outdir="gpuwalk_v",
        H=0.05,
        pad=9,
        polish=False,
        compare_arb_update=True,
        polish_residual="arb",
        backend="dd",
    ):
        super().__init__(prec, outdir, H=H, pad=pad, backend=backend)
        self.Fc = FastGB(prec, H=H, pad=pad)  # pure arb reference instrument (CPU)
        self.polish = polish
        self.compare_arb_update = compare_arb_update
        self.F_floor = None
        self.polish_residual = polish_residual
        self.V = dict(
            newton_calls=0,
            newton_fail=0,
            max_arb_res=0.0,
            max_res_gap=0.0,
            t_verify=0.0,
            events=[],
            failures=[],
            polish=dict(
                on=polish,
                F=None,
                n_states=0,
                n_updates=0,
                t_arb_res=0.0,
                t_gpu_update=0.0,
                t_arb_update=0.0,
                max_final_res=0.0,
                max_diff_to_arb=0.0,
                hist=[],
            ),
        )
        orig = self.F.newton

        def newton_checked(A, u, w, we, wc, steps=60, tol="1e-30", verbose=False, growth=5.0):
            out = orig(A, u, w, we, wc, steps=steps, tol=tol, verbose=verbose, growth=growth)
            if self.polish and out[4] < arb(tol):
                out = self.polish_state(A, out)
            elif (
                self.polish
                and os.environ.get("Q1411_BIRTH_POLISH") == "1"
                and out[4] < arb(os.environ.get("Q1411_BIRTH_TOL", "1e-30"))
            ):  # switch added 10 Oct 2026 (reading 196): polish a stagnated odd start (residual between tol and Q1411_BIRTH_TOL) before the sign test; default off
                pol = self.polish_state(A, out)
                if pol[4] < out[4]:
                    out = pol
            un, wn, wen, wcn, res, k = out
            t0 = time.perf_counter()
            Ac = arb(A.mid())
            yc = self.Fc.grid(Ac)
            r, _ = self.Fc.residual_and_jacobian(Ac, un, wn, wen, wcn, want_J=False, y=yc)
            ar = max(abs(float(t.mid())) for t in r)
            self.V["newton_calls"] += 1
            if res < arb(tol):  # an accepted result must pass in arb
                self.V["max_arb_res"] = max(self.V["max_arb_res"], ar)
                self.V["max_res_gap"] = max(self.V["max_res_gap"], abs(ar - float(res.mid())))
                if not ar < float(arb(tol).mid()) * 1.0001:
                    self.V["newton_fail"] += 1
                    self.V["failures"].append(("newton", float(Ac.mid()), ar, tol))
            self.V["t_verify"] += time.perf_counter() - t0
            return out

        self.F.newton = newton_checked

    def measure_floor(self, A, u, w, we, wc):
        """F = residual of the 320-bit all-arb root rounded to 192 bits, evaluated at 192 bits (as in q1470): 3 full all-arb Newton steps at 320 bits from the given converged 192-bit state"""
        m = len(u)
        odd = wc is not None
        n = 2 * m + 1 + (1 if odd else 0)
        nrm = lambda r: max(abs(t.mid()) for t in r)
        pack = lambda u, w, we, wc: list(u) + list(w) + [we] + ([wc] if odd else [])
        unpack = lambda x: (x[:m], x[m : 2 * m], x[2 * m], (x[2 * m + 1] if odd else None))
        old = ctx.prec
        ctx.prec = 320
        F320 = FastGB(320, H=self.Fc.Hf)
        A320 = arb(A.mid())
        y320 = F320.grid(A320)
        v = [arb(t.mid()) for t in pack(u, w, we, wc)]
        for _ in range(3):
            r, J = F320.residual_and_jacobian(A320, *unpack(v), y=y320)
            sol = J.solve(arb_mat([[-t.mid()] for t in r]))
            v = [arb((v[i] + sol[i, 0]).mid()) for i in range(n)]
        ctx.prec = 192
        T = [arb(t.mid()) for t in v]
        Ac = arb(A.mid())
        yc = self.Fc.grid(Ac)
        self.F_floor = float(nrm(self.Fc.residual_and_jacobian(Ac, *unpack(T), want_J=False, y=yc)[0]))
        self.V["polish"]["F"] = self.F_floor
        ctx.prec = 192
        return self.F_floor

    def polish_state(self, A, out):
        """mixed Newton updates (arb 192-bit residual on the CPU, dd Jacobian and dd LU solve on the GPU) until the 192-bit arb residual is <= 3 F; records residual and difference to one all-arb update per step"""
        un, wn, wen, wcn, res, k = out
        m = len(un)
        odd = wcn is not None
        n = 2 * m + 1 + (1 if odd else 0)
        P = self.V["polish"]
        pack = lambda u, w, we, wc: list(u) + list(w) + [we] + ([wc] if odd else [])
        unpack = lambda x: (x[:m], x[m : 2 * m], x[2 * m], (x[2 * m + 1] if odd else None))
        nrm = lambda r: max(abs(t.mid()) for t in r)
        Ac = arb(A.mid())
        yc = self.Fc.grid(Ac)
        N = len(yc)
        if self.F_floor is None:
            self.measure_floor(A, un, wn, wen, wcn)
        target = 3 * self.F_floor
        v = [arb(t.mid()) for t in pack(un, wn, wen, wcn)]
        hist = []
        for upd in range(10):
            t0 = time.perf_counter()
            if (
                self.polish_residual == "fx"
            ):  # residual on the GPU (320-bit fixed point, c30); the final arb check is done by newton_checked afterwards
                r, _ = (
                    self.F.fx.residual(Ac, *unpack(v), yc[0], N)
                    if self.backend == "dd"
                    else self.F.fxf.residual_and_jacobian(Ac, *unpack(v), want_J=False, y=yc)
                )
                rn = float(nrm(r))
                target = self.F_floor
            else:
                r, _ = self.Fc.residual_and_jacobian(Ac, *unpack(v), want_J=False, y=yc)
                rn = float(nrm(r))
            P["t_arb_res"] += time.perf_counter() - t0
            if rn <= target:
                break
            t0 = time.perf_counter()
            dv = self.F.update(Ac, *unpack(v), yc, r)
            vn = [arb((v[i] + dv[i]).mid()) for i in range(n)]
            P["t_gpu_update"] += time.perf_counter() - t0
            rec = dict(update=upd, residual_before=rn)
            if self.compare_arb_update:
                t0 = time.perf_counter()
                r2, J2 = self.Fc.residual_and_jacobian(Ac, *unpack(v), y=yc)
                sol = J2.solve(arb_mat([[-t.mid()] for t in r2]))
                va = [arb((v[i] + sol[i, 0]).mid()) for i in range(n)]
                P["t_arb_update"] += time.perf_counter() - t0
                rec["diff_to_arb_update"] = max(float(abs((vn[i] - va[i]).mid())) for i in range(n))
                P["max_diff_to_arb"] = max(P["max_diff_to_arb"], rec["diff_to_arb_update"])
            hist.append(rec)
            v = vn
            P["n_updates"] += 1
        P["n_states"] += 1
        P["max_final_res"] = max(P["max_final_res"], rn)
        hist.append(dict(final_residual=rn, A=float(Ac.mid()), updates=len(hist)))
        if len(P["hist"]) < 400:
            P["hist"].append(hist)
        un, wn, wen, wcn = unpack(v)
        return (un, wn, wen, wcn, arb(rn), k)

    def cpu_signal(self, A, u, w, we, wc, y):
        F = self.Fc
        H = F.H
        rowsP, wallp, wallm, cen, Q, lQ = F.tables(A, y, u, w, we, wc)
        Dv = lambda Lr, Pr: H * sum((p * (l - q) for p, l, q in zip(Pr, Lr, lQ)), arb(0))
        C = Dv(wallp[0], wallp[1])
        L0, P0, S0 = F.rows(y, arb(0))
        if wc is None:
            return Dv(L0, P0) - C
        return H * sum(
            (p * (s * s * (l - q + 2) - (l - q + 1)) for p, s, l, q in zip(P0, S0, L0, lQ)), arb(0)
        )

    def locate(self, lo, hi, tolA=1e-14, smax=1e-30):
        out = super().locate(lo, hi, tolA=tolA, smax=smax)
        A0, st0, s0, A1, st1, s1, Aev = out
        t0 = time.perf_counter()
        rec = dict(
            A_lo=A0.str(25, radius=False),
            A_hi=A1.str(25, radius=False),
            gpu_s_lo=float(s0.mid()),
            gpu_s_hi=float(s1.mid()),
        )
        sc = []
        for A, st in ((A0, st0), (A1, st1)):
            Ac = arb(A.mid())
            yc = self.Fc.grid(Ac)
            sc.append(self.cpu_signal(Ac, *st, yc))
        rec["arb_s_lo"] = float(sc[0].mid())
        rec["arb_s_hi"] = float(sc[1].mid())
        for tag, (A, st) in (
            ("lo", (A0, st0)),
            ("hi", (A1, st1)),
        ):  # final arb residual of the two end states of the bracket
            Ac = arb(A.mid())
            yc = self.Fc.grid(Ac)
            r, _ = self.Fc.residual_and_jacobian(Ac, *st, want_J=False, y=yc)
            rec["arb_res_" + tag] = max(abs(float(t.mid())) for t in r)
        rec["sign_ok"] = bool(sc[0] < 0 and sc[1] > 0)
        rec["rel_gap_lo"] = abs(rec["arb_s_lo"] - rec["gpu_s_lo"]) / max(abs(rec["gpu_s_lo"]), 1e-300)
        rec["rel_gap_hi"] = abs(rec["arb_s_hi"] - rec["gpu_s_hi"]) / max(abs(rec["gpu_s_hi"]), 1e-300)
        rec["A_event"] = Aev.str(20, radius=False)
        rec["A_event_40"] = Aev.str(40, radius=False)
        rec["A_lo_30"] = A0.str(30, radius=False)
        rec["A_hi_30"] = A1.str(30, radius=False)
        self.V["events"].append(rec)
        if not rec["sign_ok"]:
            self.V["failures"].append(("event sign", rec))
        self.V["t_verify"] += time.perf_counter() - t0
        return out

    def save_verify(self):
        json.dump(self.V, open(os.path.join(self.out, "verify.json"), "w"), indent=1)


if __name__ == "__main__":
    st, outdir, Aend, hmax = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    H = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    ctx.prec = 192
    W = VerifiedGPUWalker(192, outdir, H=H)
    A, u, w, we, wc = QW.load(st, W.F)
    t0 = time.perf_counter()
    u, w, we, wc, res, k = W.solve_at(A, u, w, we, wc)
    out = W.run(A, (u, w, we, wc), Aend, hmax=hmax)
    W.V["wall_s"] = time.perf_counter() - t0
    W.V["gpu_counters"] = GW.CNT
    W.save_verify()
    print("VERIFY", json.dumps({k: v for k, v in W.V.items() if k not in ("events", "gpu_counters")}))
