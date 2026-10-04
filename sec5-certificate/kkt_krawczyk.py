"""kkt_krawczyk: KKT system with rigorous enclosures, Krawczyk existence test, and the curve certificate.
Built on quadrature_bounds (global trapezoid grid, rigorous quadrature bounds). No top-level computation; writes nothing.
"""

import math, time
from flint import arb, arb_mat, ctx
import quadrature_bounds as FC

PM = None


def pm():
    global PM
    if PM is None:
        PM = arb(0, 1)
    return PM


def ball(c, err):
    """c plus the ball [-err, err] (err an arb upper bound)."""
    return c + A_(err) * arb(0, 1)


def A_(x):
    return x if isinstance(x, arb) else arb(x)


def sum_w(geo, m):
    s = arb(0)
    k = 0
    if geo.X:
        s += 2 * m[k]
        k += 1
    if geo.Y:
        s += 2 * m[k]
        k += 1
    for i in range(geo.nG):
        s += 4 * m[k + i]
    return s


def system(geo, grid, v, a, rho=0.5, jac=True, log=None):
    """F(v) (list of arb) and, if jac, J (arb_mat) for every v in the ball vector v; all quadrature errors included."""
    t0 = time.time()
    nG, nO = geo.nG, geo.nO
    n = nG + nO + 1
    G, m, C = geo.split(v)
    atoms = geo.atoms(G, m)
    sw = sum_w(geo, m)
    mdev = abs(sw - 1).upper()
    mmin = min(x.lower() for x in m)
    (AL, BL), (Ath, Bth), (Am, Bm) = FC.strip_consts(geo, a, msum_dev=mdev, mmin=mmin)
    if jac:
        Lam, aux = FC.lam_grid(grid, geo, atoms, keepS=True)
    else:
        Lam = FC.lam_grid(grid, geo, atoms)
        aux = None
    if log:
        log(f"  Lam grid {grid.M}^2 done [{time.time() - t0:.1f}s]")
    reps = geo.reps(G)
    order = 2
    WB = [FC.weights(grid, geo, t, order) for t in reps]
    cL = FC.contract_multi(grid, Lam, WB, order)
    eL = [FC.cauchy_coef_err(geo, (AL, BL), k, a, grid.h, grid.Y, rho) for k in range(order + 1)]
    Fc = []  # Taylor coefficients of F(t) = |x|^2/2 - I[L](x(t)) at each rep, k = 0..2
    for t, c in zip(reps, cL):
        hn = FC.half_norm2_series(geo, t, order)
        Fc.append([ball(hn[k] - c[k], eL[k]) for k in range(order + 1)])
    Fv = [Fc[o][0] - C for o in range(nO)] + [Fc[nO - nG + g][1] for g in range(nG)] + [sw - 1]
    if not jac:
        return Fv, None, Fc
    J = [[arb(0)] * n for _ in range(n)]
    for name, Gm in FC.deriv_grids(grid, geo, atoms, G, aux):
        cG = FC.contract_multi(grid, Gm, WB, 1)
        if name.startswith("th"):
            g = int(name[2:])
            col = g
            AB = (Ath, Bth)
        else:
            o = int(name[1:])
            col = nG + o
            AB = (Am, Bm)
        e0 = FC.cauchy_coef_err(geo, AB, 0, a, grid.h, grid.Y, rho)
        e1 = FC.cauchy_coef_err(geo, AB, 1, a, grid.h, grid.Y, rho)
        for o in range(nO):
            J[o][col] = -ball(cG[o][0], e0)
        for g2 in range(nG):
            J[nO + g2][col] = -ball(cG[nO - nG + g2][1], e1)
        del Gm
    for g in range(nG):  # evaluation-point terms
        o = nO - nG + g
        J[o][g] += Fc[o][1]  # d/dtheta of D(x(theta)) at the own atom
        J[nO + g][g] += 2 * Fc[o][2]
    for o in range(nO):
        J[o][n - 1] = arb(-1)
    k = 0
    if geo.X:
        J[n - 1][nG + k] = arb(2)
        k += 1
    if geo.Y:
        J[n - 1][nG + k] = arb(2)
        k += 1
    for g in range(nG):
        J[n - 1][nG + k + g] = arb(4)
    if log:
        log(f"  Jacobian done [{time.time() - t0:.1f}s]")
    return Fv, arb_mat(J), Fc


def mid_vec(v):
    return [arb(x.mid()) for x in v]


def krawczyk(geo, grid, vhat, r, a, rho=0.5, log=None, F0=None):
    """Krawczyk test on the box vhat +- r (componentwise). Returns (ok, details)."""
    n = len(vhat)
    if F0 is None:
        F0, _, _ = system(geo, grid, vhat, a, rho=rho, jac=False, log=log)
    B = [vh + A_(ri) * arb(0, 1) for vh, ri in zip(vhat, r)]
    _, JB, _ = system(geo, grid, B, a, rho=rho, jac=True, log=log)
    Jm = arb_mat(n, n, [arb(JB[i, j].mid()) for i in range(n) for j in range(n)])
    Yi = Jm.inv()
    Y = arb_mat(n, n, [arb(Yi[i, j].mid()) for i in range(n) for j in range(n)])
    YF = Y * arb_mat(n, 1, F0)
    Mx = arb_mat(n, n, [(arb(1) if i == j else arb(0)) for i in range(n) for j in range(n)]) - Y * JB
    D = arb_mat(n, 1, [A_(ri) * arb(0, 1) for ri in r])
    Kd = -YF + Mx * D  # K(B) - vhat
    ok = True
    rows = []
    for i in range(n):
        ub = abs(Kd[i, 0]).upper()
        rows.append((float(ub), float(A_(r[i]))))
        if not (ub < A_(r[i])):
            ok = False
    normM = max(sum(float(abs(Mx[i, j]).upper()) for j in range(n)) for i in range(n))
    return ok, dict(
        rows=rows,
        normM=normM,
        YF=[float(abs(YF[i, 0]).upper()) for i in range(n)],
        F0=[float(abs(x).upper()) for x in F0],
        sv=None,
        J=JB,
    )


def newton_point(geo, grid, v, a, steps=4, log=None, tol=None):
    """plain Newton on midpoints (not part of the proof; it only produces vhat)."""
    v = mid_vec(v)
    n = len(v)
    for it in range(steps):
        F, J, _ = system(geo, grid, v, a, jac=True, log=None)
        res = max(float(abs(x).upper()) for x in F)
        if log:
            log(f"  newton {it}: |F| <= {res:.3e}")
        if tol and res < tol:
            break
        Jm = arb_mat(n, n, [arb(J[i, j].mid()) for i in range(n) for j in range(n)])
        dv = Jm.solve(arb_mat(n, 1, [arb(x.mid()) for x in F]))
        v = [arb((v[i] - dv[i, 0]).mid()) for i in range(n)]
    F, _, _ = system(geo, grid, v, a, jac=False)
    if log:
        log(f"  newton final: |F| <= {max(float(abs(x).upper()) for x in F):.3e}")
    return v, F


# ---------------------------------------------------------------- curve certificate
def poly_eval(cf, s):
    acc = arb(0)
    for c in reversed(cf):
        acc = acc * s + c
    return acc


def poly_d2(cf):
    return [cf[k] * k * (k - 1) for k in range(2, len(cf))]


def rem_bounds(Mrho, rho, delta, m):
    """Taylor remainder bounds on |s| <= delta for the function and its second derivative (Cauchy, radius rho)."""
    x = A_(delta) / A_(rho)
    r0 = Mrho * x ** (m + 1) / (1 - x)
    s2 = arb(0)
    k = m + 1
    while True:
        term = arb(k * (k - 1)) * x ** (k - 2)
        s2 += term
        if k > m + 400 or float(term.upper()) < 1e-300:
            break
        k += 1
    tail = (
        2 * arb(k * (k + 1)) * x ** (k - 1) / (1 - x) ** 3
    )  # sum_{j>k} j(j-1)x^{j-2} <= (k+1)k x^{k-1}(1+x)/(1-x)^3
    r2 = Mrho / (A_(rho) ** 2) * (s2 + tail)
    return r0.upper(), r2.upper()


def curve_certify(
    geo,
    grid,
    Lam,
    C,
    a,
    atom_ts,
    atom_r,
    atom_d,
    t_lo,
    t_hi,
    nT,
    m,
    rho,
    rhoq=0.5,
    AL_BL=None,
    log=None,
    maxdep=30,
    t_hi_exact=None,
    atom_ts_arb=None,
):
    """Certify D(x(t)) - C <= 0 on [t_lo, t_hi] (floats), with equality only at the atoms.
    atom_ts (floats): centres theta^_k of the atoms in the range; atom_r: radius of each atom's enclosure (exact atom in
    [theta^ - r, theta^ + r]); atom_d: margin delta_k. On I_k = [theta^ - r - delta_k, theta^ + r + delta_k] we certify
    (D - C)'' < 0 (then D - C <= 0 there, since D - C and its derivative vanish at the exact atom, by the KKT equations or by
    symmetry at t = 0, pi/2); on the rest we certify D - C < 0. Taylor polynomials of order m on nT equal intervals.
    Returns (ok, report)."""
    T_lo = float(t_lo)
    T_hi = float(t_hi)
    width = (T_hi - T_lo) / nT
    delta = A_(width) / 2 * (1 + arb("1e-9"))  # covers float rounding of breakpoints
    AL, BL = AL_BL
    Mrho = FC.taylor_remainder_M(geo, AL, BL, rho, grid.Y)
    R0, R2 = rem_bounds(Mrho, rho, delta, m)
    eq0 = FC.cauchy_coef_err(geo, (AL, BL), 0, a, grid.h, grid.Y, rhoq)
    rq = A_(rhoq)
    eq2 = (
        2
        * FC.quad_err(AL, BL, geo.rp * rq.cosh(), geo.rp * rq.cosh(), geo.rp * rq.sinh(), a, grid.h, grid.Y)
        / rq**2
    )
    err0 = (R0 + eq0).upper()
    err2 = (R2 + eq2).upper()
    if log:
        log(
            f"  curve [{T_lo:.6f}, {T_hi:.6f}]: {nT} Taylor intervals, half-width {width/2:.4g}, order {m}, rho {rho}: "
            f"remainder {float(R0):.2e} (f), {float(R2):.2e} (f''); quadrature {float(eq0):.2e} (f), {float(eq2):.2e} (f'')"
        )
    centres_f = [T_lo + width * (i + 0.5) for i in range(nT)]
    centres = [A_(T_lo) + A_(width) * (i + arb(0.5)) for i in range(nT)]
    WB = [FC.weights(grid, geo, t, m) for t in centres]
    coefs = []
    for b0 in range(0, nT, 6):
        coefs += FC.contract_multi(grid, Lam, WB[b0 : b0 + 6], m)
    polys = []
    for i, t0 in enumerate(centres):
        hn = FC.half_norm2_series(geo, t0, m)
        cf = [hn[k] - coefs[i][k] for k in range(m + 1)]
        cf[0] -= C
        polys.append((cf, poly_d2(cf)))
    Iks = []
    tarb = atom_ts_arb if atom_ts_arb is not None else [A_(float(t)) for t in atom_ts]
    for t, ta, r, d in zip(atom_ts, tarb, atom_r, atom_d):
        lo = max(T_lo, float(t) - float(r) - float(d))
        hi = min(T_hi, float(t) + float(r) + float(d))
        # rigorous containment of the atom box, with the arb midpoint of the atom (R2 item 5.6, gap 2)
        assert (A_(ta) - A_(r) >= A_(lo)) or lo == T_lo, "atom box not inside I_k"
        assert (A_(ta) + A_(r) <= A_(hi)) or hi == T_hi, "atom box not inside I_k"
        Iks.append((lo, hi))
    bps = sorted(set([T_lo, T_hi] + [T_lo + width * i for i in range(1, nT)] + [x for I in Iks for x in I]))
    ok = True
    worst = []
    vmax = None
    d2max = None
    npc = 0
    viol = []
    nfree = 0
    natom = 0
    for u, w in zip(bps[:-1], bps[1:]):
        if w <= u:
            continue
        mid = 0.5 * (u + w)
        i = min(nT - 1, max(0, int((mid - T_lo) // width)))
        kind = "atom" if any(lo <= u and w <= hi for lo, hi in Iks) else "free"
        if kind == "free":
            nfree += 1
        else:
            natom += 1
        cf, d2 = polys[i]
        c0 = centres[i]
        # the last piece is closed at the exact endpoint (R2 item 5.6, gap 1: fl(pi/2) lies 6.1e-17 below pi/2); the Taylor disc of
        # the last interval has half-width delta = width/2 (1 + 1e-9), which covers the sliver
        wb = A_(w) if (t_hi_exact is None or w != T_hi) else A_(t_hi_exact).union(A_(w))
        stack = [(A_(u), wb, 0)]
        while stack:
            a_, b_, dep = stack.pop()
            npc += 1
            s = ((a_ + b_) / 2 - c0) + ((b_ - a_) / 2) * arb(0, 1)
            if kind == "free":
                val = poly_eval(cf, s) + A_(err0) * arb(0, 1)
                if val.upper() < 0:
                    vmax = val.upper() if vmax is None else max(vmax, val.upper())
                    continue
                if val.lower() > 0:  # certified violation: D - C > 0 on this whole piece
                    ok = False
                    viol.append((float(a_.mid()), float(b_.mid()), float(val.lower())))
                    continue
            else:
                v2 = poly_eval(d2, s) + A_(err2) * arb(0, 1)
                if v2.upper() < 0:
                    d2max = v2.upper() if d2max is None else max(d2max, v2.upper())
                    continue
                if (
                    v2.lower() > 0
                ):  # certified convexity inside an atom zone: zone too wide (or not a maximum)
                    ok = False
                    worst.append(("atom d2>0", float(a_.mid()), float(b_.mid())))
                    continue
            if dep >= maxdep or npc > 200000:
                ok = False
                worst.append((kind, float(a_.mid()), float(b_.mid())))
                continue
            mm = (a_ + b_) / 2
            stack += [(a_, mm, dep + 1), (mm, b_, dep + 1)]
    return ok, dict(
        nfree=nfree,
        natom=natom,
        err0=float(err0),
        err2=float(err2),
        max_free=float(vmax) if vmax is not None else None,
        max_d2=float(d2max) if d2max is not None else None,
        worst=worst[:10],
        pieces=npc,
        Iks=Iks,
        violations=len(viol),
        viol=(viol[:3] + sorted(viol, key=lambda x: -x[2])[:3]),
    )
