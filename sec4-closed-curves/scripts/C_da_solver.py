"""C_da_solver.py (02 Oct 2026): an independent solver for mass-constrained deterministic annealing (Rose, Gurewitz, Fox,
PRL 65 (1990) 945) with squared-error distortion and codevectors confined to the ellipse z(t) = (RP cos t, RM sin t), source
Y ~ N(0, S^2 I), temperature T = 1/beta. Free energy F = -(1/beta) E_Y log sum_j m_j exp(-beta |Y - y_j|^2); at beta = 1/2 this is the
NPMLE with unit-variance components (q789's problem). Written without reading or importing q789's code paths beyond its header.

Stationarity (unnormalised masses; maximising E log p - sum m gives sum m = 1 by itself, checked): D(t) = E_Y[e(Y, z(t))/p(Y)] = 1 at
every codevector, D'(t) = 0 at free codevectors, D <= 1 on the curve (KKT; the problem is concave in the mixture, so a KKT-valid state
is the optimum). e(Y, z) = exp(-beta |Y - z|^2), p = sum_k m_k e(Y, z_k).

Stages.
1. grid: the mass-constrained DA fixed point m_g <- m_g E_Y[e(Y, z_g)/p(Y)] (Blahut-Arimoto at fixed codevectors) on NG + 1 points of
   the first quadrant (4 NG points on the curve), float64, D2-folded; the mass is then clustered into orbits.
2. polish: Newton on orbit masses and G angles with the analytic Jacobian (bilinear forms, no finite differences) in arb ball
   arithmetic (C_PREC bits, default 128).
3. events: D(0) - 1 and D(pi/2) - 1 at empty vertices (births), D'' at every codevector (splits), interior maxima of D - 1 (G births).
4. trace: cooling in ln T with a linear predictor; when an event function turns positive, the crossing is bracketed and found by the
   Illinois method on that function along the ring's own branch (each evaluation is a polished solve); then the new structure is
   built from the event kind and polished below T_c.
Quadrature: tensor Gauss-Hermite with own node generator (orthonormal recurrence + Newton in mpmath), folded to one quadrant using
the D2 symmetry of p: E f = sum_{i,l > 0} om_i om_l Fx_i Fy_l / P_il with Fx_i(a) = exp(-beta (u_i - a)^2) + exp(-beta (u_i + a)^2).
Usage:
  py C_da_solver.py grid RP RM S T [NGH] [NG] [ITERS]
  py C_da_solver.py polish RP RM S T NGH ORBS MASSES GDEG [OUT.json]     (ORBS like XYGG; MASSES per atom, comma list; GDEG comma list)
  py C_da_solver.py trace RP RM S T_START T_END NGH START.json OUTDIR [DLNT]"""

import sys, os, time, json, math
import numpy as np
import mpmath as mp
from flint import arb, arb_mat, ctx

say = lambda *a: print(*a, flush=True)
PREC = int(os.environ.get("C_PREC", "128"))
ctx.prec = PREC
T00 = time.time()


def gh_pos(n, dps):
    """Positive nodes and weights of the n-point Gauss-Hermite rule (weight exp(-x^2)), n even."""
    mp.mp.dps = dps
    a = [mp.sqrt(mp.mpf(2) / (k + 1)) for k in range(n)]
    b = [mp.sqrt(mp.mpf(k) / (k + 1)) for k in range(n)]
    p0 = mp.pi ** (-mp.mpf(1) / 4)

    def ev(x):
        pm, pk = mp.mpf(0), p0
        for k in range(n):
            pm, pk = pk, a[k] * x * pk - b[k] * pm
        return pk, pm

    g = np.polynomial.hermite.hermgauss(n)[0]
    xs, ws = [], []
    for x0 in sorted(g[g > 0]):
        x = mp.mpf(float(x0))
        for _ in range(60):
            pn, pn1 = ev(x)
            dx = pn / (mp.sqrt(2 * n) * pn1)
            x -= dx
            if abs(dx) < mp.mpf(10) ** (-dps + 6):
                break
        pn, pn1 = ev(x)
        xs.append(x)
        ws.append(1 / (n * pn1**2))
    err = abs(2 * mp.fsum(ws) - mp.sqrt(mp.pi))
    err2 = abs(4 * mp.fsum(w * x**2 for x, w in zip(xs, ws)) - mp.sqrt(mp.pi))
    return xs, ws, err, err2


def tostr(x, d=40):
    return x.str(d, radius=False)


# ---------------------------------------------------------------- grid stage (float64)
def grid_da(rp, rm, s, T, ngh=200, ng=1024, iters=20000, m=None, every=0):
    x, w = np.polynomial.hermite.hermgauss(ngh)
    k = x > 0
    x, w = x[k], w[k]
    om = w / np.sqrt(np.pi)
    k = om > 1e-120
    x, om = x[k], om[k]
    u = np.sqrt(2) * s * x
    be = 1 / T
    t = np.linspace(0, np.pi / 2, ng + 1)
    X = rp * np.cos(t)
    Yc = rm * np.sin(t)
    mult = np.full(ng + 1, 4.0)
    mult[0] = mult[-1] = 2.0
    Kx = np.exp(2 * be * np.outer(u, X) - be * X**2) + np.exp(
        -2 * be * np.outer(u, X) - be * X**2
    )  # factor exp(-beta u^2) cancels in e/p
    Ky = np.exp(2 * be * np.outer(u, Yc) - be * Yc**2) + np.exp(-2 * be * np.outer(u, Yc) - be * Yc**2)
    OM = np.outer(om, om)
    if m is None:
        m = np.full(ng + 1, 1 / mult.sum())
    for it in range(iters):
        P = (Kx * (m * mult / 4)) @ Ky.T
        Rm = OM / P
        D = np.einsum("ig,ig->g", Kx, Rm @ Ky)
        m = m * D
        if every and it % every == 0:
            say(f"  grid it {it}: max D - 1 {D.max() - 1:+.3e}, mass {np.dot(mult, m):.15f}")
    return t, m, mult, D


def clusters(t, m, mult, rel=1e-3):
    """Contiguous runs of grid points with m > rel * max m; returns orbit list, per-atom masses, G angles (rad)."""
    on = m > rel * m.max()
    runs = []
    i = 0
    n = len(m)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    orbs, ms, gs = [], [], []
    for i, j in runs:
        idx = np.arange(i, j + 1)
        w = m[idx] * mult[idx] / 4  # per-quadrant-atom mass of each grid orbit
        if i == 0:
            orbs.append("X")
            ms.append(m[0] + 2 * m[1 : j + 1].sum())  # atoms at t and -t merge into the vertex atom
        elif j == n - 1:
            orbs.append("Y")
            ms.append(m[n - 1] + 2 * m[i : n - 1].sum())
        else:
            orbs.append("G")
            ms.append(m[idx].sum())
            gs.append(float(np.dot(w, t[idx]) / w.sum()))
    order = sorted(range(len(orbs)), key=lambda q: "XYG".index(orbs[q]))
    gsorted = sorted(gs)
    return [orbs[q] for q in order], [ms[q] for q in order], gsorted, runs


# ---------------------------------------------------------------- arb engine
class Engine:
    def __init__(self, rp, rm, s, ngh):
        dps = int(PREC * 0.30103) + 15
        xs, ws, e1, e2 = gh_pos(ngh, dps)
        sq2 = arb(2).sqrt()
        spi = arb.pi().sqrt()
        self.pi = arb.pi()
        self.rp, self.rm, self.s = arb(rp), arb(rm), arb(s)
        self.u = [sq2 * self.s * arb(mp.nstr(x, dps - 3)) for x in xs]
        self.om = [arb(mp.nstr(w, dps - 3)) / spi for w in ws]
        self.H = H = len(self.u)
        self.ngh = ngh
        self.OM = [[self.om[i] * self.om[l] for l in range(H)] for i in range(H)]
        self.gherr = (e1, e2)

    def set_T(self, T):
        self.T = T if isinstance(T, arb) else arb(str(T))
        self.beta = 1 / self.T

    def atoms(self, orbs, m, a):
        pi = self.pi
        out = []
        gi = 0
        for o, kd in enumerate(orbs):
            if kd == "X":
                out += [(arb(0), o, 0, -1), (pi, o, 0, -1)]
            elif kd == "Y":
                out += [(pi / 2, o, 0, -1), (3 * pi / 2, o, 0, -1)]
            else:
                A = a[gi]
                out += [(A, o, 1, gi), (-A, o, -1, gi), (pi - A, o, -1, gi), (pi + A, o, 1, gi)]
                gi += 1
        return out

    def build(self, orbs, m, a, deriv=True):
        """Kernel matrices of the state: A (H x K), B (K x H), derivatives in t_k, P, R = OM/P, Q = OM/P^2."""
        at = self.atoms(orbs, m, a)
        be = self.beta
        H = self.H
        u = self.u
        S = {"at": at, "K": len(at)}
        xk = [self.rp * t.cos() for t, *_ in at]
        yk = [self.rm * t.sin() for t, *_ in at]
        A = [[(-be * (u[i] - xk[k]) ** 2).exp() for k in range(len(at))] for i in range(H)]
        B = [[(-be * (u[l] - yk[k]) ** 2).exp() for l in range(H)] for k in range(len(at))]
        mk = [m[o] for _, o, _, _ in at]
        P = (
            arb_mat(H, len(at), [A[i][k] * mk[k] for i in range(H) for k in range(len(at))])
            * arb_mat(len(at), H, [x for row in B for x in row])
        ).tolist()
        OM = self.OM
        R = [[OM[i][l] / P[i][l] for l in range(H)] for i in range(H)]
        S.update(A=A, B=B, P=P, R=R, mk=mk, xk=xk, yk=yk)
        if deriv:
            S["Q"] = [[R[i][l] / P[i][l] for l in range(H)] for i in range(H)]
            dxk = [-self.rp * t.sin() for t, *_ in at]
            dyk = [self.rm * t.cos() for t, *_ in at]
            S["Ad"] = [[2 * be * (u[i] - xk[k]) * dxk[k] * A[i][k] for k in range(len(at))] for i in range(H)]
            S["Bd"] = [[2 * be * (u[l] - yk[k]) * dyk[k] * B[k][l] for l in range(H)] for k in range(len(at))]
        return S

    def qv(self, t):
        """Query vectors at curve parameter t: a0, a1, a2 (x part: value, d/dt, d2/dt2), b0, b1, b2 (y part)."""
        be = self.beta
        x = self.rp * t.cos()
        y = self.rm * t.sin()
        dx = -self.rp * t.sin()
        dy = self.rm * t.cos()
        out = []
        for c, dc, ddc in ((x, dx, -x), (y, dy, -y)):
            v0, v1, v2 = [], [], []
            for uu in self.u:
                em = (-be * (uu - c) ** 2).exp()
                ep = (-be * (uu + c) ** 2).exp()
                f1 = 2 * be * ((uu - c) * em - (uu + c) * ep)
                f2 = (4 * be * be * (uu - c) ** 2 - 2 * be) * em + (4 * be * be * (uu + c) ** 2 - 2 * be) * ep
                v0.append(em + ep)
                v1.append(f1 * dc)
                v2.append(f2 * dc * dc + f1 * ddc)
            out += [v0, v1, v2]
        return out

    @staticmethod
    def bil(a, M, b):
        H = len(a)
        return sum((a[i] * sum((M[i][l] * b[l] for l in range(H)), arb(0)) for i in range(H)), arb(0))

    def Dfam(self, S, t, order=0):
        """D, D', D'' at t (order 0, 1, 2)."""
        a0, a1, a2, b0, b1, b2 = self.qv(t)
        R = S["R"]
        H = self.H
        Rb0 = [sum((R[i][l] * b0[l] for l in range(H)), arb(0)) for i in range(H)]
        D = sum((a0[i] * Rb0[i] for i in range(H)), arb(0))
        if order == 0:
            return [D]
        Rb1 = [sum((R[i][l] * b1[l] for l in range(H)), arb(0)) for i in range(H)]
        D1 = sum((a1[i] * Rb0[i] + a0[i] * Rb1[i] for i in range(H)), arb(0))
        if order == 1:
            return [D, D1]
        Rb2 = [sum((R[i][l] * b2[l] for l in range(H)), arb(0)) for i in range(H)]
        D2 = sum((a2[i] * Rb0[i] + 2 * a1[i] * Rb1[i] + a0[i] * Rb2[i] for i in range(H)), arb(0))
        return [D, D1, D2]

    def reps(self, orbs, a):
        pi = self.pi
        out = []
        gi = 0
        for kd in orbs:
            if kd == "X":
                out.append(arb(0))
            elif kd == "Y":
                out.append(pi / 2)
            else:
                out.append(a[gi])
                gi += 1
        return out

    def residual(self, orbs, m, a, S=None):
        S = S or self.build(orbs, m, a, deriv=False)
        F = []
        Fg = []
        for kd, t in zip(orbs, self.reps(orbs, a)):
            if kd == "G":
                D, D1 = self.Dfam(S, t, 1)
                F.append(D - 1)
                Fg.append(D1)
            else:
                F.append(self.Dfam(S, t, 0)[0] - 1)
        return F + Fg, S

    def resjac(self, orbs, m, a):
        S = self.build(orbs, m, a, deriv=True)
        H = self.H
        at = S["at"]
        K = len(at)
        nO = len(orbs)
        nG = len(a)
        rp_ = self.reps(orbs, a)
        qvs = [self.qv(t) for t in rp_]
        R = S["R"]
        # pairs: (row index, a vector, b vector); rows 0..nO-1: D - 1 at reps; rows nO..: D' at G reps
        pairs = []
        F = [arb(0)] * (nO + nG)
        gidx = [o for o, kd in enumerate(orbs) if kd == "G"]
        for o in range(nO):
            a0, a1, a2, b0, b1, b2 = qvs[o]
            pairs.append((o, a0, b0))
        for g, o in enumerate(gidx):
            a0, a1, a2, b0, b1, b2 = qvs[o]
            pairs += [(nO + g, a1, b0), (nO + g, a0, b1)]
        for r, av, bv in pairs:
            F[r] += self.bil(av, R, bv)
        for o in range(nO):
            F[o] -= 1
        # columns B_k o b and Bd_k o b for every pair and atom
        cols = []
        B, Bd, A, Ad = S["B"], S["Bd"], S["A"], S["Ad"]
        for p, (r, av, bv) in enumerate(pairs):
            for k in range(K):
                cols.append([B[k][l] * bv[l] for l in range(H)])
                cols.append([Bd[k][l] * bv[l] for l in range(H)] if at[k][2] != 0 else [arb(0)] * H)
        V = arb_mat(H, len(cols), [cols[c][l] for l in range(H) for c in range(len(cols))])
        QV = (arb_mat(H, H, [x for row in S["Q"] for x in row]) * V).tolist()
        n = nO + nG
        J = [[arb(0)] * n for _ in range(n)]
        mG = [m[o] for o in gidx]
        for p, (r, av, bv) in enumerate(pairs):
            for k in range(K):
                t_, o, sg, gi = at[k]
                c0 = 2 * (p * K + k)
                c1 = c0 + 1
                T00 = sum((av[i] * A[i][k] * QV[i][c0] for i in range(H)), arb(0))
                J[r][o] -= T00
                if sg != 0:
                    T10 = sum((av[i] * Ad[i][k] * QV[i][c0] for i in range(H)), arb(0))
                    T01 = sum((av[i] * A[i][k] * QV[i][c1] for i in range(H)), arb(0))
                    J[r][nO + gi] -= mG[gi] * sg * (T10 + T01)
        for g, o in enumerate(gidx):  # the evaluation point moves with the G angle
            D, D1, D2 = self.Dfam(S, a[g], 2)
            J[o][nO + g] += D1
            J[nO + g][nO + g] += D2
        return F, J, S

    def solve(self, orbs, m, a, itmax=40, tol=1e-32, quiet=True):
        nO = len(orbs)
        nG = len(a)
        v = list(m) + list(a)
        n = nO + nG
        hist = []
        for it in range(itmax):
            F, J, S = self.resjac(orbs, v[:nO], v[nO:])
            nr = max(abs(f.mid()) for f in F)
            hist.append(nr)
            if not quiet:
                say(f"    newton {it}: residual {nr.str(3, radius=False)}  [{time.time() - T00:.0f}s]")
            if nr < arb(tol):
                return v[:nO], v[nO:], nr, it, True
            Jm = arb_mat(n, n, [J[i][j].mid() for i in range(n) for j in range(n)])
            try:
                dv = Jm.solve(arb_mat(n, 1, [-(f.mid()) for f in F]))
            except Exception as e:
                return v[:nO], v[nO:], nr, it, False
            lam = arb(1)
            ok = False
            for _ in range(25):
                vt = [(v[j] + lam * dv[j, 0]).mid() for j in range(n)]
                if all(x > 0 for x in vt[:nO]) and all(0 < x < self.pi / 2 for x in vt[nO:]):
                    Ft, _ = self.residual(orbs, vt[:nO], vt[nO:])
                    if max(abs(f.mid()) for f in Ft) < nr:
                        ok = True
                        break
                lam = lam / 2
            if not ok:
                return v[:nO], v[nO:], nr, it, False
            v = vt
        F, _ = self.residual(orbs, v[:nO], v[nO:])
        nr = max(abs(f.mid()) for f in F)
        return v[:nO], v[nO:], nr, itmax, nr < arb(tol)

    def scan(self, S, n=720):
        """D - 1 on n + 1 points of [0, pi/2] (matrix form)."""
        H = self.H
        ts = [self.pi / 2 * arb(j) / n for j in range(n + 1)]
        Avs, Bvs = [], []
        be = self.beta
        for t in ts:
            x = self.rp * t.cos()
            y = self.rm * t.sin()
            Avs.append([(-be * (uu - x) ** 2).exp() + (-be * (uu + x) ** 2).exp() for uu in self.u])
            Bvs.append([(-be * (uu - y) ** 2).exp() + (-be * (uu + y) ** 2).exp() for uu in self.u])
        RB = (
            arb_mat(H, H, [x for row in S["R"] for x in row])
            * arb_mat(H, n + 1, [Bvs[q][l] for l in range(H) for q in range(n + 1)])
        ).tolist()
        return ts, [sum((Avs[q][i] * RB[i][q] for i in range(H)), arb(0)) - 1 for q in range(n + 1)]

    def events(self, orbs, m, a, S=None, nscan=720):
        """Event functions (> 0 means the state is not KKT-valid): vertex births, splits (D'' at codevectors), interior G births."""
        S = S or self.build(orbs, m, a, deriv=False)
        ev = {}
        pi = self.pi
        if "X" not in orbs:
            ev["X birth (t = 0)"] = (self.Dfam(S, arb(0), 0)[0] - 1, arb(0))
        if "Y" not in orbs:
            ev["Y birth (t = 90)"] = (self.Dfam(S, pi / 2, 0)[0] - 1, pi / 2)
        gi = 0
        for o, (kd, t) in enumerate(zip(orbs, self.reps(orbs, a))):
            D, D1, D2 = self.Dfam(S, t, 2)
            ev[f"{kd}{o} split"] = (D2, t)
        ts, dd = self.scan(S, nscan)
        atoms_t = self.reps(orbs, a)
        best = None
        for q in range(1, nscan):
            if dd[q] > dd[q - 1] and dd[q] >= dd[q + 1]:
                if min(abs((ts[q] - x).mid()) for x in atoms_t) < pi / nscan * 2:
                    continue  # next to a codevector: covered by D''
                t = ts[q]
                for _ in range(30):  # refine the interior maximum, Newton on D'
                    D, D1, D2 = self.Dfam(S, t, 2)
                    if D2 >= 0:
                        break
                    dt = D1 / D2
                    t = t - dt
                    if abs(dt.mid()) < 1e-30:
                        break
                if not (0 < t < pi / 2):
                    continue
                D = self.Dfam(S, t, 0)[0]
                if best is None or D - 1 > best[0]:
                    best = (D - 1, t)
        if best is not None:
            ev["G birth (interior max)"] = best
        smax = max(dd[q] for q in range(nscan + 1))
        return ev, smax


def fmt_state(eng, orbs, m, a):
    pi = eng.pi
    tot = sum((x * (2 if kd in "XY" else 4) for kd, x in zip(orbs, m)), arb(0))
    return (
        f"orbits {''.join(orbs)} (K {sum(2 if kd in 'XY' else 4 for kd in orbs)}); masses per atom ["
        + ", ".join(tostr(x, 12) for x in m)
        + "]; G deg ["
        + ", ".join(tostr(x * 180 / pi, 12) for x in a)
        + f"]; total mass - 1 {tostr(tot - 1, 3)}"
    )


def dump(path, eng, orbs, m, a, extra=None):
    d = {
        "rp": tostr(eng.rp),
        "rm": tostr(eng.rm),
        "s": tostr(eng.s),
        "T": tostr(eng.T),
        "ngh": eng.ngh,
        "prec": PREC,
        "orbs": "".join(orbs),
        "m": [tostr(x) for x in m],
        "G_rad": [tostr(x) for x in a],
        "G_deg": [tostr(x * 180 / eng.pi, 30) for x in a],
    }
    if extra:
        d.update(extra)
    json.dump(d, open(path, "w"), indent=1)


def polish_cli(args):
    RP, RM, S, T, NGH, ORBS, MS, GD = args[:8]
    out = args[8] if len(args) > 8 else None
    eng = Engine(RP, RM, S, int(NGH))
    eng.set_T(T)
    orbs = list(ORBS)
    m = [arb(x) for x in MS.split(",")]
    a = [arb(x) * eng.pi / 180 for x in GD.split(",")] if GD else []
    say(
        f"C_da_solver polish rp {RP} rm {RM} s {S} T {T} NGH {NGH} prec {PREC}; GH check {mp.nstr(eng.gherr[0], 3)} {mp.nstr(eng.gherr[1], 3)}; start {fmt_state(eng, orbs, m, a)}"
    )
    m, a, nr, it, ok = eng.solve(orbs, m, a, quiet=False)
    say(
        f"polished ({'converged' if ok else 'NOT converged'}, {it} iterations, residual {nr.str(3, radius=False)}): {fmt_state(eng, orbs, m, a)}  [{time.time() - T00:.0f}s]"
    )
    ev, smax = eng.events(orbs, m, a)
    for k, (val, t) in ev.items():
        say(f"  event function {k}: {val.str(6, radius=False)} at {float((t*180/eng.pi).mid()):.6f} deg")
    say(
        f"  scan max(D - 1) on 721 points of [0, 90] deg: {smax.str(6, radius=False)}  [{time.time() - T00:.0f}s]"
    )
    if out:
        dump(
            out,
            eng,
            orbs,
            m,
            a,
            {"residual": tostr(nr, 5), "events": {k: tostr(v, 10) for k, (v, t) in ev.items()}},
        )


def grid_cli(args):
    RP, RM, S, T = [float(x) for x in args[:4]]
    NGH = int(args[4]) if len(args) > 4 else 200
    NG = int(args[5]) if len(args) > 5 else 1024
    IT = int(args[6]) if len(args) > 6 else 20000
    say(
        f"C_da_solver grid rp {RP} rm {RM} s {S} T {T}: NGH {NGH}, {NG + 1} grid orbits ({4*NG} points on the curve), {IT} iterations"
    )
    t, m, mult, D = grid_da(RP, RM, S, T, NGH, NG, IT, every=max(1, IT // 10))
    for rel in (1e-2, 1e-3, 1e-4):
        orbs, ms, gs, runs = clusters(t, m, mult, rel)
        say(
            f"  clusters at {rel:g} x max mass: {''.join(orbs)} (K {sum(2 if k in 'XY' else 4 for k in orbs)}); masses {[f'{x:.6f}' for x in ms]}; G deg "
            f"{[f'{np.degrees(g):.4f}' for g in gs]}; runs (deg) {[(round(np.degrees(t[i]), 3), round(np.degrees(t[j]), 3)) for i, j in runs]}"
        )
    pk = [i for i in range(1, len(m) - 1) if m[i] > m[i - 1] and m[i] >= m[i + 1]]
    say(
        f"  local maxima of the grid mass (deg, mass): {[(round(np.degrees(t[i]), 3), float(f'{m[i]:.3e}')) for i in pk][:40]}; max D - 1 {D.max() - 1:+.3e}  [{time.time() - T00:.0f}s]"
    )
    return t, m, mult, D


# ---------------------------------------------------------------- cooling trace
TOL = arb("1e-28")


def sort_state(orbs, m, a):
    vx = [(kd, x) for kd, x in zip(orbs, m) if kd != "G"]
    gm = [x for kd, x in zip(orbs, m) if kd == "G"]
    gs = sorted(zip(a, gm), key=lambda p: float(p[0].mid()))
    vx.sort(key=lambda p: "XY".index(p[0]))
    return [k for k, _ in vx] + ["G"] * len(gs), [x for _, x in vx] + [x for _, x in gs], [t for t, _ in gs]


def interp(v0, v1, x0, x1, x):
    return [(p + (q - p) * (x - x0) / (x1 - x0)).mid() for p, q in zip(v0, v1)]


class Tracer:
    def __init__(self, eng, log):
        self.eng = eng
        self.log = log

    def at(self, lnT, orbs, v, nscan=720):
        """Polish the structure at T = exp(lnT) from v; returns (m, a, ok, residual, events, scanmax)."""
        eng = self.eng
        eng.set_T(lnT.exp())
        nO = len(orbs)
        m, a, nr, it, ok = eng.solve(orbs, v[:nO], v[nO:])
        if not ok:
            return m, a, False, nr, None, None
        ev, smax = eng.events(orbs, m, a, nscan=nscan)
        return m, a, True, nr, ev, smax

    def root(self, key, orbs, xa, va, fa, xb, vb, fb):
        """Illinois on the event function `key` in x = ln T, fa <= 0 (valid side), fb > 0; v interpolated as the start of each solve."""
        side = 0
        bis = fa is None  # undefined at the valid end: plain bisection
        for it in range(60):
            if abs((xb - xa).mid()) < arb("1e-10"):
                break
            x = ((xa + xb) / 2).mid() if bis else (xb - fb * (xb - xa) / (fb - fa)).mid()
            if not (min(xa, xb) < x < max(xa, xb)):
                x = ((xa + xb) / 2).mid()
            v0 = interp(va, vb, xa, xb, x)
            m, a, ok, nr, ev, smax = self.at(x, orbs, v0)
            if not ok:
                x = ((xa + xb) / 2).mid()
                v0 = interp(va, vb, xa, xb, x)
                m, a, ok, nr, ev, smax = self.at(x, orbs, v0)
                if not ok:
                    self.log(f"    root: solve failed at T {tostr(x.exp(), 12)}")
                    break
            f = ev[key][0] if key in ev else None
            v = m + a
            self.log(
                f"    root {key}: T {tostr(x.exp(), 14)}  f {f.str(5, radius=False) if f is not None else 'undefined (valid)'}  (residual {nr.str(2, radius=False)})"
            )
            if f is not None and f > TOL:
                xb, vb, fb = x, v, f
                if side == 1 and not bis:
                    fa = fa / 2
                side = 1
            else:
                xa, va = x, v
                if f is None:
                    bis = True
                else:
                    fa = f
                    if side == -1 and not bis:
                        fb = fb / 2
                side = -1
        return xa, va, xb, vb

    def new_structure(self, key, orbs, m, a, tpeak):
        """Candidate post-event structures (list of (orbs, m, a)) from the event kind."""
        eng = self.eng
        pi = eng.pi
        out = []
        nO = len(orbs)
        gl = [o for o, kd in enumerate(orbs) if kd == "G"]
        if "birth" in key:
            kd = "X" if key.startswith("X") else ("Y" if key.startswith("Y") else "G")
            for mu in ("1e-6", "1e-5", "1e-4", "1e-3"):
                if kd == "G":
                    out.append(sort_state(orbs + ["G"], m + [arb(mu)], a + [tpeak]))
                else:
                    out.append(sort_state(orbs + [kd], m + [arb(mu)], a))
        else:
            o = int(key.split()[0][1:])
            kd = orbs[o]
            for dd in ("1", "2", "3", "5", "0.5"):
                d = arb(dd) * pi / 180
                rest_o = [x for q, x in enumerate(orbs) if q != o]
                rest_m = [x for q, x in enumerate(m) if q != o]
                if kd == "X":
                    out.append(sort_state(rest_o + ["G"], rest_m + [m[o] / 2], a + [d]))
                elif kd == "Y":
                    out.append(sort_state(rest_o + ["G"], rest_m + [m[o] / 2], a + [pi / 2 - d]))
                else:
                    g = gl.index(o)
                    ra = [x for q, x in enumerate(a) if q != g]
                    out.append(
                        sort_state(
                            rest_o + ["G", "G"], rest_m + [m[o] / 2, m[o] / 2], ra + [a[g] - d, a[g] + d]
                        )
                    )
        return out


def trace_cli(args):
    RP, RM, S, TS, TE, NGH, START, OUTDIR = args[:8]
    DL = arb(args[8]) if len(args) > 8 else arb("0.01")
    os.makedirs(OUTDIR, exist_ok=True)
    logf = open(os.path.join(OUTDIR, "trace.log"), "a")

    def log(s):
        say(s)
        logf.write(s + "\n")
        logf.flush()

    eng = Engine(RP, RM, S, int(NGH))
    tr = Tracer(eng, log)
    pi = eng.pi
    st = json.load(open(START))
    orbs = list(st["orbs"])
    m = [arb(x) for x in st["m"]]
    a = [arb(x) for x in st["G_rad"]]
    x = arb(TS).log()
    xend = arb(TE).log()
    log(
        f"C_da_solver trace rp {RP} rm {RM} s {S}: T {TS} -> {TE}, NGH {NGH}, prec {PREC}, step {tostr(DL, 4)} in ln T; start {START}  [{time.strftime('%H:%M:%S')}]"
    )
    m, a, ok, nr, ev, smax = tr.at(x, orbs, m + a)
    if not ok or any(val > TOL for val, _ in ev.values()):
        log(
            f"start state not valid: ok {ok}, events {[(k, tostr(v, 5)) for k, (v, _) in ev.items()] if ev else None}"
        )
        return
    log(
        f"start T {tostr(x.exp(), 10)}: {fmt_state(eng, orbs, m, a)}; max event function {max(v for v, _ in ev.values()).str(4, radius=False)}"
    )
    hist = [(x, m + a)]
    dl = DL
    nev = 0
    while x > xend:
        xn = (x - dl).mid()
        v = hist[-1][1]
        vp = interp(hist[-2][1], hist[-1][1], hist[-2][0], hist[-1][0], xn) if len(hist) > 1 else v
        mm, aa, ok, nr, ev, smax = tr.at(xn, orbs, vp)
        if not ok:
            dl = (dl / 2).mid()
            log(
                f"  solve failed at T {tostr(xn.exp(), 10)} (residual {nr.str(2, radius=False)}); step -> {tostr(dl, 3)}"
            )
            if dl < arb("1e-7"):
                log("  step below 1e-7: stopping (solver wall)")
                break
            continue
        pos = {k: (val, t) for k, (val, t) in ev.items() if val > TOL}
        if not pos:
            x = xn
            hist.append((x, mm + aa))
            hist = hist[-3:]
            dl = min(DL, (dl * arb("1.5")).mid())
            log(
                f"  T {tostr(x.exp(), 10)} valid: {fmt_state(eng, orbs, mm, aa)}; events {', '.join(f'{k} {v.str(3, radius=False)}' for k, (v, _) in ev.items())}  [{time.time() - T00:.0f}s]"
            )
            continue
        # an event between T(x) (valid) and T(xn) (invalid)
        log(
            f"  T {tostr(xn.exp(), 10)} NOT valid: positive {', '.join(f'{k} {v.str(4, radius=False)} at {float((t*180/pi).mid()):.4f} deg' for k, (v, t) in pos.items())}"
        )
        _, _, _, _, ev0, _ = tr.at(x, orbs, hist[-1][1])
        best = None
        for k in pos:
            xa, va, xb, vb = tr.root(
                k, orbs, x, hist[-1][1], ev0[k][0] if k in ev0 else None, xn, mm + aa, pos[k][0]
            )
            if best is None or xa > best[1]:
                best = (k, xa, va, xb, vb)
        k, xa, va, xb, vb = best
        nO = len(orbs)
        nev += 1
        _, _, _, _, evb, _ = tr.at(xb, orbs, vb)
        tpk = evb[k][1]
        log(
            f"EVENT {nev}: ring {''.join(orbs)} (K {sum(2 if q in 'XY' else 4 for q in orbs)}) lost at T in ({tostr(xb.exp(), 14)}, {tostr(xa.exp(), 14)}]; "
            f"T_c {tostr(((xa + xb)/2).exp(), 12)}, beta_c {tostr((-(xa + xb)/2).exp(), 12)}; firing {k} at {float((tpk*180/pi).mid()):.4f} deg; "
            f"ring at the valid end: {fmt_state(eng, orbs, va[:nO], va[nO:])}"
        )
        dump(
            os.path.join(OUTDIR, f"event{nev}_pre.json"),
            eng,
            orbs,
            va[:nO],
            va[nO:],
            {"T_valid": tostr(xa.exp()), "T_invalid": tostr(xb.exp()), "firing": k},
        )
        done = False
        for d in ("2e-4", "1e-3", "3e-3", "1e-2", "3e-2"):
            xn2 = (xa - arb(d)).mid()
            for o2, m2, a2 in tr.new_structure(k, orbs, va[:nO], va[nO:], tpk):
                mm, aa, ok, nr, ev2, smax = tr.at(xn2, o2, m2 + a2)
                if not ok:
                    continue
                sep = min(
                    [abs((t - pi / 2).mid()) for t in aa]
                    + [abs(t.mid()) for t in aa]
                    + [abs((p - q).mid()) for p in aa for q in aa if p is not q]
                    + [arb(1)]
                )
                if sep < arb("1e-4"):
                    continue
                bad = {kk: vv for kk, (vv, _) in ev2.items() if vv > TOL}
                log(
                    f"    candidate {''.join(o2)} at T {tostr(xn2.exp(), 10)}: {fmt_state(eng, o2, mm, aa)}; {'VALID' if not bad else 'not valid: ' + str({kk: vv.str(3, radius=False) for kk, vv in bad.items()})}"
                )
                if not bad:
                    orbs, m, a = o2, mm, aa
                    x = xn2
                    hist = [(x, mm + aa)]
                    dl = DL
                    done = True
                    break
            if done:
                break
        if not done:
            log("  no valid post-event structure found: stopping")
            break
        log(
            f"  after event {nev}: T {tostr(x.exp(), 10)} {fmt_state(eng, orbs, m, a)}  [{time.time() - T00:.0f}s]"
        )
        dump(os.path.join(OUTDIR, f"event{nev}_post.json"), eng, orbs, m, a)
    v = hist[-1][1]
    nO = len(orbs)
    log(
        f"trace ended at T {tostr(x.exp(), 10)}: {fmt_state(eng, orbs, v[:nO], v[nO:])}  [{time.strftime('%H:%M:%S')}]"
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "grid":
        grid_cli(sys.argv[2:])
    elif mode == "polish":
        polish_cli(sys.argv[2:])
    elif mode == "trace":
        trace_cli(sys.argv[2:])
