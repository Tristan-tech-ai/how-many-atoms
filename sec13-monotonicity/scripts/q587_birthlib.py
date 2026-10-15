"""Q587: library (no main block, no writes at import) for the 200-bit centre birth, written 30 Sep 19:31:43 (date stamp); copied from q586 v2,
which certified the K = 31 state at a = 24.3272637 (|F| 1e-49, centre mass 4.19879e-3, all three signals negative). Design as q492's
young-family continuation: at fixed a, the centre mass c is stepped up (factor 1.5, grown to 2, halved on failure) with a secant
predictor in (p, v) against c; the old system (J pair positions, J - 1 free pair masses) is solved by Newton in arb at each fixed c;
at the sign change of s0 = g(0)/C - 1, regula falsi in c; then the caller polishes the full centre-mode system.
"""

from flint import arb, arb_mat


def F_fixed(I, u, J, c):
    p = u[:J]
    v = u[J : 2 * J - 1] + [arb(1) - c - sum(u[J : 2 * J - 1])]
    gp, _ = I.gs(p, v, c, p)
    return [g1 for g0, g1, g2 in gp] + [g0 - 1 for g0, g1, g2 in gp[:-1]]


def newton_fixed(I, p, v, c, tol=1e-40, itmax=25):
    J = len(p)
    u = [arb(float(t)) for t in p] + [arb(float(t)) for t in v[:-1]]
    Fu = F_fixed(I, u, J, c)
    r = max(abs(t.mid()) for t in Fu)
    for it in range(itmax):
        if r < arb(tol):
            break
        h = arb("1e-30")
        n = len(u)
        Jm = arb_mat(n, n)
        for k in range(n):
            up = list(u)
            um = list(u)
            up[k] += h
            um[k] -= h
            Fp = F_fixed(I, up, J, c)
            Fm = F_fixed(I, um, J, c)
            for i in range(n):
                Jm[i, k] = ((Fp[i] - Fm[i]) / (2 * h)).mid()
        step = Jm.solve(arb_mat([[(-t).mid()] for t in Fu]))
        lam = arb(1)
        moved = False
        for _ in range(30):
            un = [u[i] + lam * step[i, 0].mid() for i in range(n)]
            Fn = F_fixed(I, un, J, c)
            rn = max(abs(t.mid()) for t in Fn)
            if rn < r:
                u, Fu, r, moved = un, Fn, rn, True
                break
            lam /= 2
        if not moved:
            break
    p = u[:J]
    v = u[J : 2 * J - 1] + [arb(1) - c - sum(u[J : 2 * J - 1])]
    return p, v, float(r)


def born_centre(I, p_old, v_old, say=print):
    """I = IMP at the target a; (p_old, v_old) = polished old family there (floats, pair masses summing to 1).
    Returns (p, v) floats of the fixed-c solution at s0 = 0 (pair masses sum to 1 - c), or None."""
    s0 = lambda p, v, c: float((I.gs(p, v, c, [arb(0)])[0][0][0] - 1).mid())
    fl = lambda xs: [float(t.mid()) for t in xs]
    path = [(0.0, list(p_old), list(v_old), s0([arb(x) for x in p_old], [arb(x) for x in v_old], arb(0)))]
    if path[0][3] <= 0:
        say("   born_centre: s0 <= 0 on the old family, no birth here")
        return None

    def solve_at(c):
        if len(path) >= 2:
            (ca, pa, va, _), (cb, pb, vb, _) = path[-2], path[-1]
            lam = (c - cb) / (cb - ca)
            p0 = [y + lam * (y - x) for x, y in zip(pa, pb)]
            v0 = [y + lam * (y - x) for x, y in zip(va, vb)]
        else:
            p0, v0 = path[-1][1], path[-1][2]
        v0 = [x * (1 - c) / sum(v0) for x in v0]
        return newton_fixed(I, p0, v0, arb(c))

    c, fac, bracket = 1e-4, 1.5, None
    for it in range(80):
        pc, vc, rc = solve_at(c)
        if rc > 1e-30:
            fac = 1 + (fac - 1) / 2
            say(f"   born_centre: c = {c:.3e} failed, factor -> {fac:.3f}")
            if fac < 1.01:
                return None
            c = path[-1][0] * fac if path[-1][0] > 0 else c / 2
            continue
        sc = s0(pc, vc, arb(c))
        path.append((c, fl(pc), fl(vc), sc))
        if sc < 0:
            bracket = (path[-2], path[-1])
            break
        if it >= 3 and fac < 2.0:
            fac = min(2.0, 1 + (fac - 1) * 1.5)
        c = c * fac
    if bracket is None:
        return None
    lo, hi = bracket
    for it in range(14):
        cn = lo[0] - lo[3] * (hi[0] - lo[0]) / (hi[3] - lo[3])
        pc, vc, rc = solve_at(cn)
        sc = s0(pc, vc, arb(cn))
        path.append((cn, fl(pc), fl(vc), sc))
        if abs(sc) < 1e-22:
            break
        if sc > 0:
            lo = path[-1]
        else:
            hi = path[-1]
    say(f"   born_centre: c = {path[-1][0]:.6e}, s0 = {path[-1][3]:+.1e}")
    return path[-1][1], path[-1][2]


# ---- added 30 Sep 20:08:06 (date stamp): the split counterpart, q492's fixed-eps young pair --------------------------------------------------
def F_eps(I, u, J, eps):
    """no-centre system with the inner (young) pair pinned at +-eps: unknowns p_2..p_J and J - 1 free masses; equations g'(p_j) = 0 for
    j >= 2 and g(p_j)/C = 1 for all pairs but the last (the young pair's g' equation is the one left out)."""
    p = [arb(eps)] + u[: J - 1]
    v = u[J - 1 : 2 * J - 2] + [arb(1) - sum(u[J - 1 : 2 * J - 2])]
    gp, _ = I.gs(p, v, None, p)
    return [g1 for g0, g1, g2 in gp[1:]] + [g0 - 1 for g0, g1, g2 in gp[:-1]]


def newton_eps(I, p, v, eps, tol=1e-40, itmax=25):
    J = len(p)
    u = [arb(float(t)) for t in p[1:]] + [arb(float(t)) for t in v[:-1]]
    Fu = F_eps(I, u, J, eps)
    r = max(abs(t.mid()) for t in Fu)
    for it in range(itmax):
        if r < arb(tol):
            break
        h = arb("1e-30")
        n = len(u)
        Jm = arb_mat(n, n)
        for k in range(n):
            up = list(u)
            um = list(u)
            up[k] += h
            um[k] -= h
            Fp = F_eps(I, up, J, eps)
            Fm = F_eps(I, um, J, eps)
            for i in range(n):
                Jm[i, k] = ((Fp[i] - Fm[i]) / (2 * h)).mid()
        step = Jm.solve(arb_mat([[(-t).mid()] for t in Fu]))
        lam = arb(1)
        moved = False
        for _ in range(30):
            un = [u[i] + lam * step[i, 0].mid() for i in range(n)]
            Fn = F_eps(I, un, J, eps)
            rn = max(abs(t.mid()) for t in Fn)
            if rn < r:
                u, Fu, r, moved = un, Fn, rn, True
                break
            lam /= 2
        if not moved:
            break
    p = [arb(eps)] + u[: J - 1]
    v = u[J - 1 : 2 * J - 2] + [arb(1) - sum(u[J - 1 : 2 * J - 2])]
    return p, v, float(r)


def split_centre(I, p_old, v_old, eps0=0.05, say=print):
    """I = IMP at the target a (past a centre split); (p_old, v_old) = polished old family there WITH its centre atom (pair masses v_old,
    centre 1 - sum). The centre atom is replaced by a young pair at +-eps carrying its mass; eps is stepped up (factor 1.3) with a secant
    predictor; the sign change of h(eps) = g'(eps)/C (positive while the pair wants to open) is refined by regula falsi.
    Returns (p, v) floats (no centre), or None."""
    fl = lambda xs: [float(t.mid()) if hasattr(t, "mid") else float(t) for t in xs]
    vc = 1 - sum(v_old)
    J = len(p_old) + 1

    def h(p, v):
        return float((I.gs(p, v, None, [p[0]])[0][0][1]).mid())

    path = []

    def solve_at(eps):
        if len(path) >= 2:
            (ea, pa, va, _), (eb, pb, vb, _) = path[-2], path[-1]
            lam = (eps - eb) / (eb - ea)
            p0 = [y + lam * (y - x) for x, y in zip(pa, pb)]
            v0 = [y + lam * (y - x) for x, y in zip(va, vb)]
        elif path:
            p0, v0 = path[-1][1], path[-1][2]
        else:
            p0, v0 = [eps] + list(p_old), [vc] + list(v_old)
        p0[0] = eps
        return newton_eps(I, p0, v0, eps)

    eps, fac, bracket = eps0, 1.3, None
    for it in range(60):
        pe, ve, re = solve_at(eps)
        if re > 1e-30:
            fac = 1 + (fac - 1) / 2
            say(f"   split_centre: eps = {eps:.4f} failed (|F| {re:.0e}), factor -> {fac:.3f}")
            if fac < 1.01 or not path:
                return None
            eps = path[-1][0] * fac
            continue
        he = h(pe, ve)
        path.append((eps, fl(pe), fl(ve), he))
        say(f"   split_centre: eps = {eps:.4f}, g'(eps)/C = {he:+.2e}")
        if len(path) >= 2 and he * path[-2][3] < 0:
            bracket = (path[-2], path[-1])
            break
        eps *= fac
        if eps > 1.5:
            return None
    if bracket is None:
        return None
    lo, hi = bracket
    for it in range(20):
        en = lo[0] - lo[3] * (hi[0] - lo[0]) / (hi[3] - lo[3])
        pe, ve, re = solve_at(en)
        he = h(pe, ve)
        path.append((en, fl(pe), fl(ve), he))
        if abs(he) < 1e-24:
            break
        if he * lo[3] > 0:
            lo = path[-1]
        else:
            hi = path[-1]
    say(f"   split_centre: eps = {path[-1][0]:.6f}, g'/C = {path[-1][3]:+.1e}")
    return path[-1][1], path[-1][2]
