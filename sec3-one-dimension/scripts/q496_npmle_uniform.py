"""Q496 (EXPLORATORY): number of atoms K(a) of the POPULATION NPMLE of a Gaussian location mixture when the data density
is uniform on [-a, a]. Question: does K grow like a^(4/3) (the law found here for the amplitude-constrained capacity and
the bounded-normal-mean least-favourable prior), like a, or like the deterministic upper bound C0 (xmax - xmin)^2 of
Polyanskiy-Wu (arXiv 2008.08244, Theorem 1; papers/polyanskiy_wu_2008.08244.txt lines 146-150)?

PROBLEM. maximise l(pi) = int f0(x) log (phi * pi)(x) dx over probability measures pi, f0 = 1/(2a) on [-a, a]. Concave;
KKT: D(theta) = int f0(x) phi(x - theta)/(phi * pi)(x) dx <= 1, equality on supp(pi). D is analytic and -> 0 at
infinity, so pi is discrete and finite. For a <= sqrt(3) the solution is the point mass at 0 (D(t) = sinh(a t)/(a t)
exp(-t^2/2) <= 1 iff a^2 <= 3), which is the exact starting point of the continuation.

METHOD (v2, replaces the draft entirely).
 * Symmetric prior: atoms +-p_j (j = 1..n) with weight v_j each, optional centre atom 0 with weight c; K = 2n (+1).
 * Quadrature: composite Gauss-Legendre in x on [0, a] (panels of width <= 0.5, q = 16 nodes per panel), folded by
   symmetry: D(t) = (1/2a) int_0^a [phi(x - t) + phi(x + t)]/m(x) dx. The only non-smoothness of f0 is at +-a, which
   are panel ends. Doubling check: q = 32.
 * Unknowns z = (p, v, c); equations D(p_j) - 1 = 0, D'(p_j) = 0, (D(0) - 1 = 0). Square; sum of weights = 1 follows
   from the identity sum_k w_k D(theta_k) = 1. Newton with the ANALYTIC Jacobian
   J = -Rows diag(W/m^2) Cols^T + diag(D'(p_k)) [rows D(p_k)] + diag(D''(p_k)) [rows D'(p_k)],
   Rows = (G_k, H_k, 2 phi), Cols = (v_j H_j, G_j, phi), G_j = phi(x - p_j) + phi(x + p_j), H_j = dG_j/dp_j;
   2-norm backtracking, Levenberg-Marquardt fallback. Weights may go negative inside Newton (only m > 0 is enforced), so
   that a dying atom shows up as a sign change of its weight.
 * Continuation in a from a = 1 (point mass at 0) on the reporting grid a = 1, 1.25, 1.5, ... (step 0.25); inside a
   step the seed is the linear extrapolation of the last two solutions with the same structure (or the previous solution
   with positions rescaled by a_new/a_old when there is only one); failed steps are halved.
 * KKT scan after every solve: D - 1 on theta in [0, a + 6] (step 0.004), local extrema refined by Newton on D'.
   A structure change is needed when (i) an off-atom local maximum of D - 1 exceeds TOL_VIOL = 1e-12, or (ii) D''(p_k) > 0
   at an atom (the atom has become a local minimum of D: it splits), or (iii) a weight is negative (the atom dies).
   The change point a* is located by bisection (to 1e-4) with the old structure, then applied: new centre atom, new pair
   at the off-atom maximum, split of an atom into two, or removal. Each event is logged with a*, type and location.
 * Signals: dip_g = -min (D - 1) over the interior of each gap between consecutive atoms (including the gap around 0);
   S = min_g dip_g (the flattest gap) and where it is; off = largest off-atom local maximum of D - 1; min weight;
   D''(0).

PRE-REGISTRATION (written before the main sweep; only development runs at a <= 6 were made before this text):
 (P1) Exponent test. Let A_res be the largest reporting-grid a such that every grid point in [1, A_res] passes (P3).
      On the grid points of the upper half [(1 + A_res)/2, A_res] fit K = c a^p + b by least squares with p free
      (scipy curve_fit) and with p fixed at 1, 4/3 and 2 (linear least squares in c, b). Report c, b, p and the rms of
      each fit. The same four fits are repeated on the event points (a*_i, K just after the i-th change) in the same
      range as a secondary test. Reading rule: the fixed exponent with the smallest rms is named; if the rms of two fixed
      exponents are within a factor 1.5 of each other the test is called undecided between them (K is a step function
      with steps of 1 or 2, so rms below about 0.4 carries no further information).
 (P2) Controls. (a) At a = 1 and a = 2, an independent method: EM on a fine theta grid (step 0.002 on
      [-(a + 1.5), a + 1.5], not symmetrised), x by the midpoint rule with 800 points on [-a, a] (not Gauss-Legendre),
      SQUAREM-accelerated, run until the grid KKT gap max D - 1 < 1e-9 or 20000 cycles; clusters of grid mass (runs with
      mass > 1e-7 separated by gaps > 0.05) give K, centroids and masses, which must agree with the Newton atoms and
      weights to 1e-4 (same K; max |atom difference| and max |weight difference| < 1e-4). (b) The analytic Jacobian
      matches central differences (step 1e-6) to relative 1e-6 at a = 6. (c) Quadrature doubling (q = 16 -> 32 per
      panel) changes atoms and weights by less than 1e-10 at every integer a reported (checked during the sweep).
 (P3) Resolvability rule. Noise floor nu(a) = max(max |F| at convergence, |change of S under q-doubling|, |change of S
      when D - 1 is re-evaluated with the quadrature nodes in reversed summation order|). A count is reported as resolved
      only if S(a) >= 1e3 nu(a); A_res stops at the first failure and nothing beyond it enters (P1).
 No claim beyond the numbers is attached to this exploratory run.

SOLVER NOTES ADDED DURING THE SWEEP (method only; the pre-registration above is unchanged):
 * a centre split is a pitchfork (pair separation ~ sqrt(a - a*)); at a* + 1e-4 Newton falls into the merged solution,
   so the split is applied at a* + 0.02. From a = 11.37 on the plain Newton seed (the old maximum of D) fails because the
   centre is flat; the young pair is then followed along its branch with the half-separation s held fixed and a free
   (analytic dF/da), from s = 0.01 until a(s) passes a* + 0.02 (split_centre_path). Each path point solves the full
   system at its own a(s), so the split is applied at the path end (releasing s by Newton stalled at 1e-9 at a = 24.06).
 * a centre birth at a = 28.37 failed the same way (young weight 1e-9); births then use birth_centre_path: the young
   centre weight c held fixed (1e-7, x1.5 per step) and a free, applied at the path end a(c) >= a* + 0.02. The path
   start a(c = 1e-7) is recorded as a_star_path: the bisection a* is late by about 1e-12/(d viol/da) because a birth is
   flagged only when the violation exceeds TOL_VIOL (0.002 at a = 28.37).
 * at a = 29.57 (beyond A_res) D(0) - 1 crosses 0 at 29.5695 and D''(p_1) at 29.5718, both signals ~1e-13; the birth
   is flagged only above TOL_VIOL, so the split flag came first. Priority rule: with no centre atom and D(0) - 1 > 0,
   the centre birth is applied first. Counts beyond A_res are accepted only where the arb check (mode mp) confirms them.
 * ARB STAGE (after the double sweep stopped at 29.57): mode mp polishes stored double rows in arb (160 bits) and
   re-measures every dip, D'' at the atoms and 12 points per gap; mode mpsweep continues in arb (128 bits) from the double
   row at 29.5 with the same events handled by arb bisection and arb paths, and a full arb KKT check on the 0.25 grid.
   Validated on 28.0 -> 28.75 against the double sweep (same K, same dips to 4 digits, birth a* 28.37104 vs 28.37106).
   It stopped at a = 33.5: at 33.6487 D''(0) crosses 0 and by 33.70 D'' > 0 also at p_1 and p_2; the K = 46, 47, 48
   candidate structures did not converge from local seeds, so the count beyond 33.65 is not resolved.
   Fallbacks: multi-seed Newton, then free-support EM + Newton. Any solution is accepted only if it passes the KKT scan,
   and the optimum is unique, so the route to it does not affect the result.

Usage:
  py q496_npmle_uniform.py control           (P2a, P2b)
  py q496_npmle_uniform.py sweep AMAX [resume] (main sweep from a = 1; writes results_q496_npmle_uniform.json)
  py q496_npmle_uniform.py fit               (P1 on the stored sweep)
  py q496_npmle_uniform.py analyse           (post-hoc, not pre-registered: event spacing, signal decay, gap law)
  py q496_npmle_uniform.py mp A1 A2 ...      (arb check of stored double rows; appends to results_q496_npmle_uniform_mp.json)
  py q496_npmle_uniform.py mpsweep A0 A1 [resume]  (arb continuation; results_q496_npmle_uniform_mpsweep.json, *_mp.log)
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    try:
        if int(os.environ.get(_v, "99")) > 4:
            os.environ[_v] = "4"
    except ValueError:
        os.environ[_v] = "4"
import sys, json, time
import numpy as np

say = lambda *a: print(*a, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_q496_npmle_uniform.json")
S2P = 1 / np.sqrt(2 * np.pi)
TOL_VIOL = 1e-12
RES_OK = 1e-12


# ------------------------------------------------------------------------------------------------ quadrature
class Quad:
    """composite Gauss-Legendre on [0, a], weights include the fold and the density 1/(2a)"""

    def __init__(self, a, q=16, hp=0.5):
        self.a, self.q = a, q
        npan = max(1, int(np.ceil(a / hp - 1e-9)))
        t, w = np.polynomial.legendre.leggauss(q)
        e = np.linspace(0.0, a, npan + 1)
        mid = 0.5 * (e[:-1] + e[1:])
        half = 0.5 * (e[1:] - e[:-1])
        self.x = (mid[:, None] + half[:, None] * t[None, :]).ravel()
        self.W = (half[:, None] * w[None, :]).ravel() / (2 * a)


def unpack(z, n, centre):
    return z[:n], z[n : 2 * n], (z[2 * n] if centre else None)


def mixture(x, p, v, c):
    m = np.zeros_like(x)
    if len(p):
        m = v @ (
            S2P
            * (np.exp(-0.5 * (x[None, :] - p[:, None]) ** 2) + np.exp(-0.5 * (x[None, :] + p[:, None]) ** 2))
        )
    if c is not None:
        m = m + c * S2P * np.exp(-0.5 * x * x)
    return m


def system(z, n, centre, Q, jac=True):
    """F and analytic J. Returns None if m <= 0 somewhere."""
    x, W = Q.x, Q.W
    p, v, c = unpack(z, n, centre)
    phi0 = S2P * np.exp(-0.5 * x * x)
    if n:
        u1 = x[None, :] - p[:, None]
        u2 = x[None, :] + p[:, None]
        E1 = S2P * np.exp(-0.5 * u1 * u1)
        E2 = S2P * np.exp(-0.5 * u2 * u2)
        G = E1 + E2
        H = u1 * E1 - u2 * E2
        m = v @ G
    else:
        m = np.zeros_like(x)
    if centre:
        m = m + c * phi0
    if not np.all(m > 0):
        return None
    r = W / m
    F = []
    aux = dict(m=m)
    if n:
        D0 = G @ r
        D1 = H @ r
        D2 = ((u1 * u1 - 1) * E1 + (u2 * u2 - 1) * E2) @ r
        F += [D0 - 1, D1]
        aux.update(D0=D0, D1=D1, D2=D2)
    if centre:
        Dc = 2 * phi0 @ r
        F.append([Dc - 1])
        aux["Dc"] = Dc
    F = np.concatenate(F)
    if not jac:
        return F, None, aux
    rows = ([G, H] if n else []) + ([2 * phi0[None, :]] if centre else [])
    cols = ([v[:, None] * H, G] if n else []) + ([phi0[None, :]] if centre else [])
    Rw = np.vstack(rows) * (r / m)[None, :]
    J = -Rw @ np.vstack(cols).T
    if n:
        idx = np.arange(n)
        J[idx, idx] += D1
        J[n + idx, idx] += D2
    return F, J, aux


def valid(z, n, centre):
    p = z[:n]
    if n and (p[0] <= 1e-9 or np.any(np.diff(p) <= 1e-9)):
        return False
    return True


def newton(z, n, centre, Q, maxit=80, tol=1e-15):
    out = system(z, n, centre, Q)
    if out is None:
        return z, np.inf
    F, J, _ = out
    s2 = F @ F
    mu = 1e-6
    for it in range(maxit):
        if np.abs(F).max() < tol:
            break
        ok = False
        try:
            dz = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(J, F, rcond=None)[0]
        lam = 1.0
        for _ in range(40):
            zn = z - lam * dz
            if valid(zn, n, centre):
                o = system(zn, n, centre, Q)
                if o is not None and o[0] @ o[0] < s2:
                    z, (F, J, _), s2, ok = zn, o, o[0] @ o[0], True
                    break
            lam *= 0.5
        if not ok:  # Levenberg-Marquardt on the 2-norm
            JTJ = J.T @ J
            g = J.T @ F
            dg = np.diag(JTJ).copy() + 1e-300
            for _ in range(30):
                try:
                    dz = np.linalg.solve(JTJ + mu * np.diag(dg), g)
                except np.linalg.LinAlgError:
                    mu *= 10
                    continue
                zn = z - dz
                if valid(zn, n, centre):
                    o = system(zn, n, centre, Q)
                    if o is not None and o[0] @ o[0] < s2:
                        z, (F, J, _), s2, ok = zn, o, o[0] @ o[0], True
                        mu = max(mu / 10, 1e-12)
                        break
                mu *= 10
        if not ok:
            break
    return z, float(np.abs(F).max())


# ------------------------------------------------------------------------------------------------ D and scans
def Dvals(th, z, n, centre, Q, order=0, reverse=False):
    p, v, c = unpack(z, n, centre)
    x, W = Q.x, Q.W
    if reverse:
        x, W = x[::-1].copy(), W[::-1].copy()
    m = mixture(x, p, v, c)
    r = W / m
    th = np.atleast_1d(np.asarray(th, float))
    out = [np.empty(len(th)) for _ in range(order + 1)]
    for s in range(0, len(th), 1500):
        t = th[s : s + 1500]
        u1 = x[None, :] - t[:, None]
        u2 = x[None, :] + t[:, None]
        E1 = S2P * np.exp(-0.5 * u1 * u1)
        E2 = S2P * np.exp(-0.5 * u2 * u2)
        out[0][s : s + 1500] = (E1 + E2) @ r
        if order >= 1:
            out[1][s : s + 1500] = (u1 * E1 - u2 * E2) @ r
        if order >= 2:
            out[2][s : s + 1500] = ((u1 * u1 - 1) * E1 + (u2 * u2 - 1) * E2) @ r
    return out


def refine(t0, z, n, centre, Q, lo, hi):
    """Newton on D'(t) = 0 from t0, kept in [lo, hi]"""
    t = np.array(t0, float)
    for _ in range(8):
        _, d1, d2 = Dvals(t, z, n, centre, Q, order=2)
        step = np.where(np.abs(d2) > 0, d1 / np.where(d2 == 0, 1, d2), 0.0)
        t = np.clip(t - step, lo, hi)
    return t


def scan(z, n, centre, Q, dth=0.004, reverse=False):
    a = Q.a
    p, v, c = unpack(z, n, centre)
    atoms = np.concatenate([[0.0] if centre else [], p])
    grid = np.arange(0.0, a + 6.0, dth)
    Dg = Dvals(grid, z, n, centre, Q, reverse=reverse)[0] - 1
    ext = np.concatenate([[Dg[1]], Dg, [Dg[-1] - 1]])  # symmetric at 0, decreasing at the far end
    imax = np.where((ext[1:-1] >= ext[:-2]) & (ext[1:-1] > ext[2:]))[0]
    imin = np.where((ext[1:-1] <= ext[:-2]) & (ext[1:-1] < ext[2:]))[0]
    tmax = (
        refine(grid[imax], z, n, centre, Q, np.maximum(grid[imax] - dth, 0), grid[imax] + dth)
        if len(imax)
        else np.array([])
    )
    tmin = (
        refine(grid[imin], z, n, centre, Q, np.maximum(grid[imin] - dth, 0), grid[imin] + dth)
        if len(imin)
        else np.array([])
    )
    dmax = Dvals(tmax, z, n, centre, Q, reverse=reverse)[0] - 1 if len(tmax) else np.array([])
    dmin = Dvals(tmin, z, n, centre, Q, reverse=reverse)[0] - 1 if len(tmin) else np.array([])
    near = lambda t: np.min(np.abs(atoms - t)) if len(atoms) else np.inf
    off = [(float(d), float(t)) for t, d in zip(tmax, dmax) if near(t) > 1e-5]
    viol, where = max(off) if off else (-np.inf, np.nan)
    # gaps: (0 or -p1 .. p1), (p1, p2), ...; dip = -min(D - 1) in the open gap
    edges = list(atoms) if centre else ([-p[0]] + list(p) if n else [])
    dips = []
    for g in range(len(edges) - 1):
        lo, hi = max(edges[g], 0.0), edges[g + 1]
        sel = (tmin > lo + 1e-6) & (tmin < hi - 1e-6)
        if centre is False and g == 0:
            sel = (tmin >= 0) & (tmin < hi - 1e-6)
        if np.any(sel):
            k = np.argmin(dmin[sel])
            dips.append((float(-dmin[sel][k]), float(tmin[sel][k])))
    off_grid = Dg[np.array([near(t) > 0.05 for t in grid])]
    # no gap at all (K = 1): the signal is how far D - 1 stays below 0 at distance > 0.05 from the atom
    S, Swhere = min(dips) if dips else (float(-off_grid.max()), np.nan)
    return dict(
        viol=float(viol),
        viol_at=float(where),
        S=float(S),
        S_at=float(Swhere),
        dips=dips,
        off_grid=float(off_grid.max()) if len(off_grid) else np.nan,
        offmax=off,
    )


def atom_curv(z, n, centre, Q):
    p, v, c = unpack(z, n, centre)
    d2p = Dvals(p, z, n, centre, Q, order=2)[2] if n else np.array([])
    d2c = float(Dvals([0.0], z, n, centre, Q, order=2)[2][0])
    return d2p, d2c


def loglik(z, n, centre, Q):
    p, v, c = unpack(z, n, centre)
    return float(2 * Q.W @ np.log(mixture(Q.x, p, v, c)))


def K_of(n, centre):
    return 2 * n + (1 if centre else 0)


# ------------------------------------------------------------------------------------------------ structure changes
def diagnose(z, n, centre, Q):
    """returns (needs_change, info) for a converged solution"""
    p, v, c = unpack(z, n, centre)
    sc = scan(z, n, centre, Q)
    d2p, d2c = atom_curv(z, n, centre, Q)
    wmin = min(list(v) + ([c] if centre else [])) if (n or centre) else 1.0
    reasons = []
    if wmin < 0:
        reasons.append("death")
    if n and np.max(d2p) > 0:
        reasons.append("split")
    if centre and d2c > 0:
        reasons.append("split0")
    if sc["viol"] > TOL_VIOL:
        reasons.append("viol")
    D0m1 = float(Dvals([0.0], z, n, centre, Q)[0][0] - 1) if not centre else None
    return reasons, dict(sc=sc, d2p=d2p, d2c=d2c, wmin=wmin, D0m1=D0m1)


def apply_change(z, n, centre, Q, reasons, info):
    """one structure change; returns (z, n, centre, event description)"""
    p, v, c = unpack(z, n, centre)
    p = list(p)
    v = list(v)
    sc = info["sc"]
    if "death" in reasons:
        allw = list(v) + ([c] if centre else [])
        k = int(np.argmin(allw))
        if k == len(v):
            ev = dict(type="death-centre", at=0.0)
            centre = False
            c = None
        else:
            ev = dict(type="death-pair", at=float(p[k]))
            del p[k]
            del v[k]
    elif "split0" in reasons:
        lim = 0.5 * p[0] if p else 3.0
        cand = [(d, t) for d, t in sc["offmax"] if 0 < t < lim]
        s = max(cand)[1] if cand else 0.02
        s = max(s, 0.005)
        ev = dict(type="split-centre", at=0.0, new=s)
        p = [s] + p
        v = [c / 2] + v
        centre = False
        c = None
    elif "split" in reasons and not centre and info.get("D0m1") is not None and info["D0m1"] > 0:
        # v4 priority rule (a = 29.57): D(0) - 1 has already crossed 0 (centre birth) but is still below TOL_VIOL
        # while D''(p_1) crosses 0 a little later; the birth comes first
        ev = dict(type="birth-centre", at=0.0, note="priority over split")
        centre = True
        c = 1e-9
    elif "split" in reasons:
        k = int(np.argmax(info["d2p"]))
        pk = p[k]
        left = p[k - 1] if k > 0 else (0.0 if centre else -pk)
        right = p[k + 1] if k + 1 < len(p) else pk + 6.0
        lo_c = [t for d, t in sc["offmax"] if 0.5 * (left + pk) < t < pk]
        hi_c = [t for d, t in sc["offmax"] if pk < t < 0.5 * (pk + right)]
        t1 = max(lo_c) if lo_c else pk - 0.01
        t2 = min(hi_c) if hi_c else pk + 0.01
        ev = dict(type="split-pair", at=float(pk), new=[float(t1), float(t2)])
        p = p[:k] + [t1, t2] + p[k + 1 :]
        v = v[:k] + [v[k] / 2, v[k] / 2] + v[k + 1 :]
    else:  # birth at the largest off-atom local maximum
        t = sc["viol_at"]
        if t < 1e-6 and not centre:
            ev = dict(type="birth-centre", at=0.0)
            centre = True
            c = 1e-9
        else:
            j = int(np.searchsorted(p, t))
            where = "outer" if j == len(p) else "bulk"
            ev = dict(type=f"birth-{where}", at=float(t))
            p.insert(j, t)
            v.insert(j, 1e-9)
    n = len(p)
    zn = np.concatenate([np.array(p), np.array(v)] + ([[c]] if centre else []))
    return zn, n, centre, ev


def relax_newton(z, n, centre, Q, blocks=40, per=100):
    """fallback after a structure change: free-support EM for the fixed-K symmetric mixture (monotone in the
    likelihood; w_k <- w_k D(t_k), t_k <- t_k + D'(t_k)/D(t_k)), with a Newton attempt after every block"""
    p, v, c = unpack(z.copy(), n, centre)
    v = np.maximum(v, 1e-4)
    c = max(c, 1e-4) if centre else None
    s = 2 * v.sum() + (c if centre else 0.0)
    v = v / s
    c = c / s if centre else None
    best = (None, np.inf)
    for b in range(blocks):
        for it in range(per):
            zz = np.concatenate([p, v] + ([[c]] if centre else []))
            if n:
                D0, D1 = Dvals(p, zz, n, centre, Q, order=1)
                v = v * D0
                p = p + D1 / D0
            if centre:
                c = c * Dvals([0.0], zz, n, centre, Q)[0][0]
        zz = np.concatenate([p, v] + ([[c]] if centre else []))
        zt, rt = newton(zz, n, centre, Q)
        if rt < best[1]:
            best = (zt, rt)
        if rt < RES_OK:
            return zt, rt
    return best


def dF_da(z, n, centre, a):
    """analytic derivative of F (the stationarity system) with respect to the half-width a, positions and weights
    fixed: D(t) = (1/2a) int_0^a g_t/m, so dD/da = g_t(a)/(2a m(a)) - D(t)/a; the same for D'(t) with h_t."""
    Q = Quad(a)
    F = system(z, n, centre, Q, jac=False)[0]
    p, v, c = unpack(z, n, centre)
    xa = np.array([a])
    ma = mixture(xa, p, v, c)[0]
    out = []
    if n:
        e1 = S2P * np.exp(-0.5 * (a - p) ** 2)
        e2 = S2P * np.exp(-0.5 * (a + p) ** 2)
        D0 = F[:n] + 1
        D1 = F[n : 2 * n]
        out += [(e1 + e2) / (2 * a * ma) - D0 / a, ((a - p) * e1 - (a + p) * e2) / (2 * a * ma) - D1 / a]
    if centre:
        Dc = F[2 * n] + 1
        out.append([2 * S2P * np.exp(-0.5 * a * a) / (2 * a * ma) - Dc / a])
    return np.concatenate(out)


def newton_afree(z, n, centre, a, fix, maxit=60, tol=1e-15):
    """Newton for F(z, a) = 0 with z[fix] held fixed and a free (square system)"""
    free = [i for i in range(len(z)) if i != fix]

    def ev(z, a):
        o = system(z, n, centre, Quad(a))
        return None if o is None else (o[0], o[1])

    o = ev(z, a)
    if o is None:
        return z, a, np.inf
    F, J = o
    s2 = F @ F
    for it in range(maxit):
        if np.abs(F).max() < tol:
            break
        JJ = np.hstack([J[:, free], dF_da(z, n, centre, a)[:, None]])
        try:
            d = np.linalg.solve(JJ, F)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(JJ, F, rcond=None)[0]
        lam, ok = 1.0, False
        for _ in range(40):
            zn = z.copy()
            zn[free] = z[free] - lam * d[:-1]
            an = a - lam * d[-1]
            if valid(zn, n, centre) and an > 0.5:
                o = ev(zn, an)
                if o is not None and o[0] @ o[0] < s2:
                    z, a, (F, J), s2, ok = zn, an, o, o[0] @ o[0], True
                    break
            lam *= 0.5
        if not ok:
            break
    return z, a, float(np.abs(F).max())


def split_centre_path(zold, n, centre, a_star, a_target, out=say):
    """young-pair branch of a centre split, parametrised by the half-separation s (held fixed, a free), from
    s = 0.01 until a(s) >= a_target; then s is released at a = a_target. Returns (z, res) of the new structure.
    """
    p, v, c = unpack(zold, n, centre)
    s = 0.01
    z = np.concatenate([[s], p, [c / 2], v])
    nn = n + 1
    a = a_star
    z, a, r = newton_afree(z, nn, False, a, 0)
    if r >= RES_OK:
        return None, np.inf, []
    path = [(s, a, z.copy())]
    ds = 0.01
    while a < a_target and len(path) < 400:
        s_new = s + ds
        if len(path) >= 2:  # secant predictor in s
            (s1, a1, z1), (s2_, a2, z2) = path[-2], path[-1]
            t = (s_new - s2_) / (s2_ - s1)
            zp = z2 + t * (z2 - z1)
            ap = a2 + t * (a2 - a1)
        else:
            zp, ap = z.copy(), a
        zp[0] = s_new
        zt, at_, rt = newton_afree(zp, nn, False, ap, 0)
        if rt < RES_OK and at_ >= a - 0.05:
            s, z, a = s_new, zt, at_
            path.append((s, a, z.copy()))
            ds = min(ds * 1.5, 0.05)
        else:
            ds /= 2
            if ds < 1e-5:
                return None, np.inf, path
    # every path point solves the full system at its own a(s) (all equations imposed, only s fixed and a free), so the
    # split is applied at the path end a(s_end) >= a_target (v3; releasing s at a_target by Newton stalled at 1e-9 at 24.06)
    if a < a_target:
        return None, None, [(float(s_), float(a_)) for s_, a_, _ in path]
    return z, a, [(float(s_), float(a_)) for s_, a_, _ in path]


def birth_centre_path(zold, n, a_star, a_target):
    """young centre atom of a centre birth, parametrised by its weight c (held fixed, a free), from c = 1e-7
    (x1.5 per step) until a(c) >= a_target; each path point solves the full system at its own a(c)."""
    p, v, _ = unpack(zold, n, False)
    c = 1e-7
    z = np.concatenate([p, v, [c]])
    fix = 2 * n
    z, a, r = newton_afree(z, n, True, a_star, fix)
    if r >= RES_OK:
        return None, None, []
    path = [(c, a, z.copy())]
    fac = 1.5
    while a < a_target and len(path) < 400:
        c_new = c * fac
        if len(path) >= 2:
            (c1, a1, z1), (c2, a2, z2) = path[-2], path[-1]
            t = (c_new - c2) / (c2 - c1)
            zp = z2 + t * (z2 - z1)
            ap = a2 + t * (a2 - a1)
        else:
            zp, ap = z.copy(), a
        zp[fix] = c_new
        zt, at_, rt = newton_afree(zp, n, True, ap, fix)
        if rt < RES_OK and at_ >= a - 0.05:
            c, z, a = c_new, zt, at_
            path.append((c, a, z.copy()))
            fac = min(fac * 1.2, 2.0)
        else:
            fac = 1 + (fac - 1) / 2
            if fac < 1 + 1e-4:
                return None, None, [(float(c_), float(a_)) for c_, a_, _ in path]
    if a < a_target:
        return None, None, [(float(c_), float(a_)) for c_, a_, _ in path]
    return z, a, [(float(c_), float(a_)) for c_, a_, _ in path]


SPLIT_MEMO = {}


def split_seeds(z, n, centre, Q, ev, da_after):
    """multi-seed Newton for a split (the Newton basin of the young pair is narrow when the centre is flat).
    Seeds: half-separations s on a grid, tried in order of distance from the pitchfork prediction
    s_prev sqrt(da_after/da_prev) of the previous split of the same kind. Accept a converged solution with a
    non-degenerate pair and no KKT violation."""
    p, v, c = unpack(z, n, centre)
    kind = ev["type"]
    grid = np.concatenate([np.linspace(0.01, 0.1, 10), np.linspace(0.12, 1.0, 45)])
    if kind in SPLIT_MEMO:
        s0, d0 = SPLIT_MEMO[kind]
        pred = s0 * np.sqrt(max(da_after, 1e-6) / max(d0, 1e-6))
        grid = grid[np.argsort(np.abs(grid - pred))]
    for s in grid:
        if kind == "split-centre":
            if n and s >= 0.9 * p[0]:
                continue
            zn = np.concatenate([[s], p, [c / 2], v])
            nn, cc = n + 1, False
        else:
            k = int(np.argmin(np.abs(p - ev["at"])))
            pk = p[k]
            left = p[k - 1] if k > 0 else (0.0 if centre else -pk)
            right = p[k + 1] if k + 1 < n else pk + 6.0
            if pk - s <= max(left, 0.0) + 1e-3 or pk + s >= right - 1e-3:
                continue
            pn = np.concatenate([p[:k], [pk - s, pk + s], p[k + 1 :]])
            vn = np.concatenate([v[:k], [v[k] / 2, v[k] / 2], v[k + 1 :]])
            zn = np.concatenate([pn, vn] + ([[c]] if centre else []))
            nn, cc = n + 1, centre
        zz, rr = newton(zn, nn, cc, Q, maxit=60)
        if rr >= RES_OK:
            continue
        pp, vv, c2 = unpack(zz, nn, cc)
        if (
            np.min(np.diff(np.concatenate([[0.0], pp]))) < 1e-3
            or np.min(list(vv) + ([c2] if cc else [])) <= 0
        ):
            continue
        if diagnose(zz, nn, cc, Q)[0]:
            continue
        if kind == "split-centre":
            SPLIT_MEMO[kind] = (float(pp[0]), da_after)
        return zz, rr, s
    return None, np.inf, None


# ------------------------------------------------------------------------------------------------ continuation
def solve_same(zseed, n, centre, a, q=16):
    Q = Quad(a, q)
    z, res = newton(zseed.copy(), n, centre, Q)
    return z, res, Q


def rescale(z, n, a_old, a_new):
    z = z.copy()
    z[:n] *= a_new / a_old
    return z


def full_report(z, n, centre, Q, a, res):
    p, v, c = unpack(z, n, centre)
    reasons, info = diagnose(z, n, centre, Q)
    sc = info["sc"]
    # noise floor: q doubling and reversed summation
    Q2 = Quad(a, 2 * Q.q)
    z2, res2 = newton(z.copy(), n, centre, Q2)
    sc2 = scan(z2, n, centre, Q2)
    scr = scan(z, n, centre, Q, reverse=True)
    dq_atoms = float(np.max(np.abs(z2 - z))) if len(z) else 0.0
    dS_q = abs(sc2["S"] - sc["S"]) if np.isfinite(sc["S"]) else np.nan
    dS_r = abs(scr["S"] - sc["S"]) if np.isfinite(sc["S"]) else np.nan
    nu = float(np.nanmax([res, dS_q, dS_r]))
    wall = list(v) + ([c] if centre else [])
    row = dict(
        a=a,
        K=K_of(n, centre),
        n=n,
        centre=bool(centre),
        res=res,
        viol=sc["viol"],
        viol_at=sc["viol_at"],
        off_grid=sc["off_grid"],
        S=sc["S"],
        S_at=sc["S_at"],
        dips=sc["dips"],
        minw=float(min(wall)),
        d2c=info["d2c"],
        maxd2p=float(np.max(info["d2p"])) if n else None,
        loglik=loglik(z, n, centre, Q),
        nu=nu,
        dS_q=dS_q,
        dS_r=dS_r,
        dq_z=dq_atoms,
        res_q2=res2,
        reasons=reasons,
        resolved=bool(np.isfinite(sc["S"]) and sc["S"] >= 1e3 * nu),
        p=list(map(float, p)),
        v=list(map(float, v)),
        c=(float(c) if centre else None),
        gaps=list(map(float, np.diff(np.concatenate([[0.0], p])))) if n else [],
    )
    return row


def sweep(AMAX, DA=0.25, A0=1.0, log=None, resume=False):
    T0 = time.time()
    el = lambda: f"[{time.time() - T0:.0f}s]"
    out = lambda s: (say(s), log.write(s + "\n"), log.flush()) if log else say(s)
    rows, events = [], []
    n, centre, z = 0, True, np.array([1.0])  # exact solution for a <= sqrt(3)
    a = A0
    if resume:
        d = json.load(open(RESULTS))
        rows, events = d["rows"], [e for e in d["events"] if e.get("K_after") is not None]
        r = rows[-1]
        a, n, centre = r["a"], r["n"], r["centre"]
        z = np.concatenate([r["p"], r["v"]] + ([[r["c"]]] if centre else []))
        out(f"   RESUME from a = {a}, K = {r['K']}")
    Q = Quad(a)
    z, res = newton(z, n, centre, Q)
    hist = [(a, z.copy())]
    targets = [round(A0 + k * DA, 6) for k in range(int(round((AMAX - A0) / DA)) + 1)]
    targets = [t for t in targets if t > a + 1e-9] if resume else targets
    for at in targets:
        # ---- advance from a to at with the current structure, handling changes by bisection ----
        guard = 0
        while a < at - 1e-12 and guard < 200:
            guard += 1
            step = at - a
            ok = False
            while step > 1e-6:
                an = a + step
                if len(hist) >= 2 and hist[-2][1].shape == hist[-1][1].shape:
                    (a1, z1), (a2, z2) = hist[-2], hist[-1]
                    seed = z2 + (z2 - z1) * (an - a2) / (a2 - a1)
                else:
                    seed = rescale(hist[-1][1], n, hist[-1][0], an)
                zn, resn, Qn = solve_same(seed, n, centre, an)
                if resn < RES_OK:
                    ok = True
                    break
                step /= 2
            if not ok:
                out(f"   continuation failed at a = {a:.6f} (K = {K_of(n, centre)}) {el()}")
                return rows, events
            reasons, info = diagnose(zn, n, centre, Qn)
            if not reasons:
                a, z = an, zn
                hist.append((a, z.copy()))
                continue
            # ---- bisection for the change point with the old structure ----
            lo, zlo, hi, zhi = a, z.copy(), an, zn.copy()
            while hi - lo > 1e-4:
                mid = 0.5 * (lo + hi)
                seed = zlo + (zhi - zlo) * (mid - lo) / (hi - lo)
                zm, resm, Qm = solve_same(seed, n, centre, mid)
                if resm >= RES_OK:
                    break
                rm, _ = diagnose(zm, n, centre, Qm)
                if rm:
                    hi, zhi = mid, zm
                else:
                    lo, zlo = mid, zm
            Qh = Quad(hi)
            reasons, info = diagnose(zhi, n, centre, Qh)
            Kb = K_of(n, centre)
            a_app, z_app = hi, zhi
            if ("split" in reasons or "split0" in reasons) and "death" not in reasons:
                # a split is a pitchfork: the pair separation grows like sqrt(a - a*), and at a* + 1e-4 Newton falls
                # into the merged (degenerate) solution. March the old branch 0.02 further and split there.
                for dap in (0.02, 0.05):
                    a_try = hi + dap
                    zt, rt, Qt = solve_same(zhi + (zhi - zlo) * (a_try - hi) / (hi - lo), n, centre, a_try)
                    if rt < RES_OK:
                        a_app, z_app = a_try, zt
                        break
                Qh = Quad(a_app)
                reasons, info = diagnose(z_app, n, centre, Qh)
            zc, nc, cc = z_app, n, centre
            evs = []
            for _ in range(6):
                zprev, nprev, cprev = zc, nc, cc
                zc0, nc, cc, ev = apply_change(zc, nc, cc, Qh, reasons, info)
                resc = np.inf
                if ev["type"] == "split-centre" and nprev == n and cprev == centre:
                    zs, as_, path = split_centre_path(zhi, n, centre, hi, a_app)
                    if zs is not None:
                        Qs = Quad(as_)
                        rs = float(np.abs(system(zs, nc, cc, Qs, jac=False)[0]).max())
                        if rs < RES_OK and not diagnose(zs, nc, cc, Qs)[0]:
                            zc, resc, a_app, Qh = zs, rs, as_, Qs
                            ev["path_end"] = path[-1]
                            ev["path_len"] = len(path)
                            ev["a_star_path"] = path[0][1]
                if resc >= RES_OK:
                    zc, resc = newton(zc0, nc, cc, Qh)
                if (
                    resc >= RES_OK
                    and ev["type"] == "birth-centre"
                    and nprev == n
                    and cprev == centre
                    and not centre
                ):
                    zs, as_, path = birth_centre_path(zhi, n, hi, hi + 0.02)
                    if zs is not None:
                        Qs = Quad(as_)
                        rs = float(np.abs(system(zs, nc, cc, Qs, jac=False)[0]).max())
                        if rs < RES_OK and not diagnose(zs, nc, cc, Qs)[0]:
                            zc, resc, a_app, Qh = zs, rs, as_, Qs
                            ev["path_end"] = path[-1]
                            ev["path_len"] = len(path)
                            ev["a_star_path"] = path[0][1]
                if resc < RES_OK and ev["type"].startswith("split") and diagnose(zc, nc, cc, Qh)[0]:
                    resc = np.inf  # converged to a wrong (KKT-violating) configuration
                if resc >= RES_OK and ev["type"].startswith("split"):
                    zs, rs, s_used = split_seeds(zprev, nprev, cprev, Qh, ev, a_app - 0.5 * (lo + hi))
                    if zs is not None:
                        zc, resc = zs, rs
                        ev["seeded"] = float(s_used)
                if resc >= RES_OK:
                    zc, resc = relax_newton(zc0, nc, cc, Qh)
                    ev["relaxed"] = True
                evs.append(ev)
                if resc >= RES_OK:
                    out(
                        f"   after {ev} at a = {a_app:.5f}: Newton residual {resc:.1e} even after EM relaxation; stopping"
                    )
                    for e in evs:
                        e.update(
                            a_star=0.5 * (lo + hi),
                            a_lo=lo,
                            a_hi=hi,
                            a_applied=a_app,
                            K_before=Kb,
                            K_after=None,
                        )
                        events.append(e)
                    json.dump(dict(rows=rows, events=events, AMAX=AMAX, DA=DA), open(RESULTS, "w"))
                    return rows, events
                reasons, info = diagnose(zc, nc, cc, Qh)
                if not reasons:
                    break
            for ev in evs:
                ev.update(
                    a_star=0.5 * (lo + hi),
                    a_lo=lo,
                    a_hi=hi,
                    a_applied=a_app,
                    K_before=Kb,
                    K_after=K_of(nc, cc),
                )
                ev = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in ev.items()}
                events.append(ev)
                out(
                    f"   EVENT {ev['type']:13s} at a* = {ev['a_star']:.4f} (theta = {float(ev['at']):.4f}): K {Kb} -> {K_of(nc, cc)}"
                    f"{'' if a_app == hi else f' (applied at a = {a_app:.4f})'}  {el()}"
                )
            if reasons:
                out(f"   unresolved structure after changes at a = {a_app:.5f}: {reasons}")
            a, z, n, centre = a_app, zc, nc, cc
            hist = [(a, z.copy())]
            if a > at + 1e-12:  # the split was applied beyond the grid point: come back
                zb, ab, okb = z.copy(), a, True
                for k in range(1, 9):
                    ak = a + (at - a) * k / 8
                    zk, rk, Qk = solve_same(zb, n, centre, ak)
                    if rk >= RES_OK:
                        okb = False
                        break
                    zb, ab = zk, ak
                if okb:
                    a, z = at, zb
                    hist = [(a, z.copy())]
                else:
                    out(
                        f"   backward continuation to the grid point {at} failed; row reported at a = {a:.4f}"
                    )
        Qa = Quad(a)
        res = float(np.abs(system(z, n, centre, Qa, jac=False)[0]).max())
        row = full_report(z, n, centre, Qa, a, res)
        rows.append(row)
        gl = row["gaps"]
        out(
            f"a = {a:6.2f}: K = {row['K']:3d}, res {res:.1e}, off-atom max(D-1) {row['viol']:+.2e} (grid {row['off_grid']:+.2e}), "
            f"S = {row['S']:.2e} at {row['S_at']:.2f}, min w {row['minw']:.2e}, nu {row['nu']:.1e}, dq {row['dq_z']:.1e}, "
            f"{'ok' if row['resolved'] else 'UNRESOLVED'}  gaps(centre..edge) {np.round(gl, 3).tolist() if len(gl) <= 8 else np.round(gl[:3], 3).tolist() + ['...'] + np.round(gl[-3:], 3).tolist()}  {el()}"
        )
        json.dump(dict(rows=rows, events=events, AMAX=AMAX, DA=DA), open(RESULTS, "w"))
    return rows, events


# ------------------------------------------------------------------------------------------------ controls
def em_grid_control(a, h=0.002, M=800, cycles=20000, tol=1e-9):
    x = -a + (np.arange(M) + 0.5) * (2 * a / M)
    wx = np.full(M, 1.0 / M)
    th = np.arange(-(a + 1.5), a + 1.5 + h / 2, h)
    Phi = S2P * np.exp(-0.5 * (x[:, None] - th[None, :]) ** 2)
    w = np.full(len(th), 1.0 / len(th))
    ll = lambda w: float(wx @ np.log(Phi @ w))
    em = lambda w: w * (Phi.T @ (wx / (Phi @ w)))
    for cyc in range(cycles):
        w1 = em(w)
        w2 = em(w1)
        r = w1 - w
        vv = w2 - w1 - r
        nv = np.sqrt(vv @ vv)
        if nv == 0:
            w = w2
            break
        alpha = min(-1.0, -np.sqrt(r @ r) / nv)
        l2 = ll(w2)
        while True:
            wn = w - 2 * alpha * r + alpha * alpha * vv
            if np.all(wn >= 0) or alpha >= -1.0:
                break
            alpha = (alpha - 1) / 2
        wn = np.maximum(wn, 0)
        wn /= wn.sum()
        wn = em(wn)
        w = wn if ll(wn) >= l2 else w2
        if cyc % 200 == 0:
            gap = float((Phi.T @ (wx / (Phi @ w))).max() - 1)
            if gap < tol:
                break
    Dg = Phi.T @ (wx / (Phi @ w))
    gap = float(Dg.max() - 1)
    # clusters
    big = np.where(w > 1e-7)[0]
    cl = []
    cur = [big[0]]
    for i in big[1:]:
        if th[i] - th[cur[-1]] > 0.05:
            cl.append(cur)
            cur = [i]
        else:
            cur.append(i)
    cl.append(cur)
    out = []
    for cidx in cl:
        lo, hi = th[cidx[0]] - 0.05, th[cidx[-1]] + 0.05
        sel = (th >= lo) & (th <= hi)
        mass = w[sel].sum()
        cen = (w[sel] @ th[sel]) / mass
        out.append((float(cen), float(mass), float(np.sqrt(max((w[sel] @ (th[sel] - cen) ** 2) / mass, 0)))))
    return out, gap, cyc, ll(w)


def control():
    T0 = time.time()
    el = lambda: f"[{time.time() - T0:.0f}s]"
    say("q496 controls (P2a, P2b)")
    # Newton solutions at a = 1, 2, 6 by the same continuation as the sweep
    rows, events = sweep(6.0, DA=0.25, A0=1.0)
    byA = {r["a"]: r for r in rows}
    allok = True
    for a in (1.0, 2.0):
        r = byA[a]
        th_n = sorted([-t for t in r["p"]] + ([0.0] if r["centre"] else []) + r["p"])
        w_n = [
            w
            for _, w in sorted(
                zip(
                    [-t for t in r["p"]] + ([0.0] if r["centre"] else []) + r["p"],
                    r["v"] + ([r["c"]] if r["centre"] else []) + r["v"],
                )
            )
        ]
        cl, gap, cyc, ll = em_grid_control(a)
        say(
            f"(P2a) a = {a}: EM grid: {len(cl)} clusters after {cyc + 1} SQUAREM cycles, grid KKT gap {gap:.1e}, loglik {ll:.10f}; "
            f"Newton: K = {r['K']}, loglik {r['loglik']:.10f}   {el()}"
        )
        for (cen, mass, sd), t, w in zip(cl, th_n, w_n):
            say(
                f"        EM centroid {cen:+.7f} mass {mass:.7f} (sd {sd:.1e})   Newton atom {t:+.7f} weight {w:.7f}"
            )
        ok = len(cl) == r["K"] and all(
            abs(c[0] - t) < 1e-4 and abs(c[1] - w) < 1e-4 for c, t, w in zip(cl, th_n, w_n)
        )
        say(f"      -> {'PASS' if ok else 'FAIL'}")
        allok &= ok
    # P2b: Jacobian vs central differences at a = 6
    r = byA[6.0]
    n, centre = r["n"], r["centre"]
    z = np.concatenate([r["p"], r["v"]] + ([[r["c"]]] if centre else []))
    Q = Quad(6.0)
    F, J, _ = system(z, n, centre, Q)
    Jn = np.zeros_like(J)
    h = 1e-6
    for j in range(len(z)):
        zp = z.copy()
        zp[j] += h
        zm = z.copy()
        zm[j] -= h
        Jn[:, j] = (system(zp, n, centre, Q, jac=False)[0] - system(zm, n, centre, Q, jac=False)[0]) / (2 * h)
    rel = float(np.abs(J - Jn).max() / np.abs(J).max())
    ok = rel < 1e-6
    say(
        f"(P2b) a = 6 (K = {r['K']}): analytic vs central-difference Jacobian, max |diff|/max |J| = {rel:.1e}: {'PASS' if ok else 'FAIL'}"
    )
    allok &= ok
    say(f"CONTROLS: {'PASS' if allok else 'FAIL'}  {el()}")


# ------------------------------------------------------------------------------------------------ fits (P1)
def fit():
    from scipy.optimize import curve_fit

    d = json.load(open(RESULTS))
    rows, events = d["rows"], d["events"]
    A_res = None
    for r in rows:
        if not r["resolved"]:
            break
        A_res = r["a"]
    say(
        f"P1: resolved range [1, {A_res}] (first unresolved grid point after it: "
        f"{next((r['a'] for r in rows if not r['resolved']), None)})"
    )
    lo = (1 + A_res) / 2

    def fits(A, K, label):
        say(f"  {label}: {len(A)} points, a in [{A.min():.3f}, {A.max():.3f}]")
        res = {}
        for pfix in (1.0, 4 / 3, 2.0):
            X = np.vstack([A**pfix, np.ones(len(A))]).T
            cb, *_ = np.linalg.lstsq(X, K, rcond=None)
            rms = float(np.sqrt(np.mean((X @ cb - K) ** 2)))
            res[pfix] = rms
            say(f"     p = {pfix:.4f} fixed: c = {cb[0]:.5f}, b = {cb[1]:+.4f}, rms {rms:.4f}")
        try:
            f = lambda a, c, p, b: c * a**p + b
            (c, p, b), cov = curve_fit(f, A, K, p0=(0.5, 1.3, 0.0), maxfev=20000)
            rms = float(np.sqrt(np.mean((f(A, c, p, b) - K) ** 2)))
            say(
                f"     p free       : c = {c:.5f}, p = {p:.4f} (+-{np.sqrt(cov[1, 1]):.4f} nominal), b = {b:+.4f}, rms {rms:.4f}"
            )
        except Exception as e:
            say(f"     p free: fit failed ({e})")
        best = min(res, key=res.get)
        srt = sorted(res.values())
        undecided = [p for p in res if p != best and res[p] <= 1.5 * res[best]]
        say(
            f"     reading rule: smallest fixed-p rms at p = {best:.4f}"
            + (f"; undecided vs p = {[round(u, 4) for u in undecided]}" if undecided else "")
        )

    A = np.array([r["a"] for r in rows if lo - 1e-9 <= r["a"] <= A_res + 1e-9])
    K = np.array([r["K"] for r in rows if lo - 1e-9 <= r["a"] <= A_res + 1e-9], float)
    fits(A, K, "grid points (primary)")
    E = [(e["a_star"], e["K_after"]) for e in events if lo <= e["a_star"] <= A_res]
    if len(E) >= 4:
        fits(np.array([e[0] for e in E]), np.array([e[1] for e in E], float), "event points (secondary)")


def mp_check(avals, prec=160, q=16, per_gap=12):
    """MULTIPRECISION check (python-flint arb) of stored double solutions: same symmetric system, composite
    Gauss-Legendre with q nodes per panel of width <= 0.5 (nodes and weights from arb.legendre_p_root), Newton polish
    with the analytic Jacobian, then in arb: every gap's dip (refined by Newton on D'), D'' at every atom, and D - 1
    on per_gap interior points of every gap. The count K is confirmed at a if max |F| is tiny, all weights > 0, all
    D''(atoms) < 0 and D - 1 < 0 at every off-atom point examined."""
    from flint import arb, arb_mat, ctx

    ctx.prec = prec
    d = json.load(open(RESULTS))
    rows = {r["a"]: r for r in d["rows"]}
    S2Pm = 1 / (2 * arb.pi()).sqrt()
    phi = lambda u: S2Pm * (-(u * u) / 2).exp()
    tw = [arb.legendre_p_root(q, k, weight=True) for k in range(q)]
    out = []
    for a0 in avals:
        T0 = time.time()
        r = min(rows.values(), key=lambda rr: abs(rr["a"] - a0))
        a = r["a"]
        n, centre = r["n"], r["centre"]
        A = arb(a)
        npan = max(1, int(np.ceil(a / 0.5 - 1e-9)))
        xs, Ws = [], []
        for j in range(npan):
            lo, hi = A * j / npan, A * (j + 1) / npan
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            for t, w in tw:
                xs.append(mid + half * t)
                Ws.append(half * w / (2 * A))
        N = len(xs)
        z = (
            [arb(float(t)) for t in r["p"]]
            + [arb(float(t)) for t in r["v"]]
            + ([arb(float(r["c"]))] if centre else [])
        )

        def build(z, jac=True):
            p = z[:n]
            v = z[n : 2 * n]
            c = z[2 * n] if centre else None
            ph0 = [phi(x) for x in xs]
            E1 = [[phi(x - pk) for x in xs] for pk in p]
            E2 = [[phi(x + pk) for x in xs] for pk in p]
            m = [
                sum((v[k] * (E1[k][i] + E2[k][i]) for k in range(n)), arb(0)) + (c * ph0[i] if centre else 0)
                for i in range(N)
            ]
            rr_ = [Ws[i] / m[i] for i in range(N)]
            G = [[E1[k][i] + E2[k][i] for i in range(N)] for k in range(n)]
            H = [[(xs[i] - p[k]) * E1[k][i] - (xs[i] + p[k]) * E2[k][i] for i in range(N)] for k in range(n)]
            D2 = [
                sum(
                    (((xs[i] - p[k]) ** 2 - 1) * E1[k][i] + ((xs[i] + p[k]) ** 2 - 1) * E2[k][i]) * rr_[i]
                    for i in range(N)
                )
                for k in range(n)
            ]
            rows_ = G + H + ([[2 * y for y in ph0]] if centre else [])
            Rm = arb_mat(rows_)
            rv = arb_mat([[y] for y in rr_])
            Fv = Rm * rv
            F = (
                [Fv[k, 0] - 1 for k in range(n)]
                + [Fv[n + k, 0] for k in range(n)]
                + ([Fv[2 * n, 0] - 1] if centre else [])
            )
            if not jac:
                return F, None, D2
            cols = [[v[k] * y for y in H[k]] for k in range(n)] + G + ([ph0] if centre else [])
            Rw = arb_mat([[row[i] * rr_[i] / m[i] for i in range(N)] for row in rows_])
            CT = arb_mat([[cols[j][i] for j in range(len(cols))] for i in range(N)])
            J = -(Rw * CT)
            for k in range(n):
                J[k, k] = J[k, k] + Fv[n + k, 0]
                J[n + k, k] = J[n + k, k] + D2[k]
            return F, J, D2

        for it in range(12):
            F, J, D2 = build(z)
            nF = max(abs(f.mid()) for f in F)
            if nF < arb(10) ** (-(prec * 0.3010 - 12)):
                break
            dz = J.solve(arb_mat([[f.mid()] for f in F]))
            z = [(z[i] - dz[i, 0]).mid() for i in range(len(z))]
        F, _, D2 = build(z, jac=False)
        nF = float(max(abs(f.mid()) for f in F))
        p = z[:n]
        v = z[n : 2 * n]
        c = z[2 * n] if centre else None
        m_at = None

        def Dfun(ts):
            # D, D', D'' at the arb points ts
            pv = [pp for pp in p]
            m = [
                sum((v[k] * (phi(x - pv[k]) + phi(x + pv[k])) for k in range(n)), arb(0))
                + (c * phi(x) if centre else 0)
                for x in xs
            ]
            rr_ = [Ws[i] / m[i] for i in range(N)]
            res = []
            for t in ts:
                e1 = [phi(x - t) for x in xs]
                e2 = [phi(x + t) for x in xs]
                d0 = sum(((e1[i] + e2[i]) * rr_[i] for i in range(N)), arb(0))
                d1 = sum((((xs[i] - t) * e1[i] - (xs[i] + t) * e2[i]) * rr_[i] for i in range(N)), arb(0))
                d2 = sum(
                    (
                        (((xs[i] - t) ** 2 - 1) * e1[i] + ((xs[i] + t) ** 2 - 1) * e2[i]) * rr_[i]
                        for i in range(N)
                    ),
                    arb(0),
                )
                res.append((d0, d1, d2))
            return res

        # dips: start from the double dip locations, Newton on D' in arb
        dips = []
        for dep, t0 in r["dips"]:
            t = arb(float(t0))
            for _ in range(6):
                ((d0, d1, d2),) = Dfun([t])
                if float(t0) == 0.0:
                    break
                t = (t - d1 / d2).mid()
            ((d0, d1, d2),) = Dfun([t])
            dips.append((float((d0 - 1).mid()), float(t.mid()), dep))
        # interior grid points of every gap
        at = ([0.0] if centre else []) + [float(t.mid()) for t in p]
        edges = at if centre else [-at[0]] + at
        pts = []
        for g in range(len(edges) - 1):
            lo, hi = max(edges[g], 0.0), edges[g + 1]
            pts += [lo + (hi - lo) * (k + 0.5) / per_gap for k in range(per_gap)]
        vals = Dfun([arb(t) for t in pts])
        gmax = max(float((d0 - 1).mid()) for d0, _, _ in vals)
        wmin = min([float(x.mid()) for x in v] + ([float(c.mid())] if centre else []))
        d2max = max(float(x.mid()) for x in D2) if n else None
        S_mp = min(-d for d, _, _ in dips) if dips else None
        ok = (
            nF < 1e-30
            and wmin > 0
            and (d2max is None or d2max < 0)
            and gmax < 0
            and all(d < 0 for d, _, _ in dips)
        )
        dz = max(
            abs(float(z[i].mid()) - float(([*r["p"], *r["v"]] + ([r["c"]] if centre else []))[i]))
            for i in range(len(z))
        )
        say(
            f"MP a = {a}: K = {r['K']}, prec {prec}, max|F| {nF:.1e}, |z_mp - z_double| {dz:.1e}, min w {wmin:.3e}, max D''(atoms) {d2max:+.2e}, "
            f"S_mp = {S_mp:.3e} (double S = {r['S']:.3e}), max(D - 1) on {len(pts)} gap points {gmax:+.3e}: {'K CONFIRMED' if ok else 'NOT CONFIRMED'}  [{time.time() - T0:.0f}s]"
        )
        out.append(
            dict(
                a=a,
                K=r["K"],
                prec=prec,
                maxF=nF,
                dz=dz,
                minw=wmin,
                maxd2=d2max,
                S_mp=S_mp,
                S_double=r["S"],
                gmax=gmax,
                dips=dips,
                ok=bool(ok),
            )
        )
    return out


class MPC:
    """arb (python-flint) version of the symmetric system for the continuation beyond the double-precision range"""

    def __init__(self, prec=128, q=16):
        from flint import arb, arb_mat, ctx

        ctx.prec = prec
        self.arb, self.arb_mat, self.prec = arb, arb_mat, prec
        self.S2P = 1 / (2 * arb.pi()).sqrt()
        self.tw = [arb.legendre_p_root(q, k, weight=True) for k in range(q)]
        self.tol = 10.0 ** (
            -min(25.0, prec * 0.30103 - 12)
        )  # signals are >= 1e-15; the residual floor is ~cond*eps

    def phi(self, u):
        return self.S2P * (-(u * u) / 2).exp()

    def quad(self, A):
        a = float(A.mid())
        npan = max(1, int(np.ceil(a / 0.5 - 1e-9)))
        xs, Ws = [], []
        for j in range(npan):
            lo, hi = A * j / npan, A * (j + 1) / npan
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            for t, w in self.tw:
                xs.append(mid + half * t)
                Ws.append(half * w / (2 * A))
        return xs, Ws

    def state(self, z, n, centre, A):
        arb, phi = self.arb, self.phi
        xs, Ws = self.quad(A)
        N = len(xs)
        p = z[:n]
        v = z[n : 2 * n]
        c = z[2 * n] if centre else None
        E1 = [[phi(x - pk) for x in xs] for pk in p]
        E2 = [[phi(x + pk) for x in xs] for pk in p]
        ph0 = [phi(x) for x in xs]
        m = [
            sum((v[k] * (E1[k][i] + E2[k][i]) for k in range(n)), arb(0)) + (c * ph0[i] if centre else 0)
            for i in range(N)
        ]
        if any(float(mi.mid()) <= 0 for mi in m):
            return None
        r = [Ws[i] / m[i] for i in range(N)]
        return dict(
            xs=xs, Ws=Ws, N=N, p=p, v=v, c=c, E1=E1, E2=E2, ph0=ph0, m=m, r=r, A=A, n=n, centre=centre
        )

    def FJ(self, st, jac=True, fix=None):
        arb, arb_mat, phi = self.arb, self.arb_mat, self.phi
        xs, N, p, v, c, E1, E2, ph0, m, r, A, n, centre = (
            st[k] for k in ("xs", "N", "p", "v", "c", "E1", "E2", "ph0", "m", "r", "A", "n", "centre")
        )
        G = [[E1[k][i] + E2[k][i] for i in range(N)] for k in range(n)]
        H = [[(xs[i] - p[k]) * E1[k][i] - (xs[i] + p[k]) * E2[k][i] for i in range(N)] for k in range(n)]
        D2 = [
            sum(
                (
                    (((xs[i] - p[k]) ** 2 - 1) * E1[k][i] + ((xs[i] + p[k]) ** 2 - 1) * E2[k][i]) * r[i]
                    for i in range(N)
                ),
                arb(0),
            )
            for k in range(n)
        ]
        rows_ = G + H + ([[2 * y for y in ph0]] if centre else [])
        Fv = arb_mat(rows_) * arb_mat([[y] for y in r])
        F = (
            [Fv[k, 0] - 1 for k in range(n)]
            + [Fv[n + k, 0] for k in range(n)]
            + ([Fv[2 * n, 0] - 1] if centre else [])
        )
        D0c = sum((2 * ph0[i] * r[i] for i in range(N)), arb(0))
        D2c = sum((2 * (xs[i] * xs[i] - 1) * ph0[i] * r[i] for i in range(N)), arb(0))
        out = dict(F=F, D2=D2, D0c=D0c, D2c=D2c)
        if not jac:
            return out
        cols = [[v[k] * y for y in H[k]] for k in range(n)] + G + ([ph0] if centre else [])
        Rw = arb_mat([[row[i] * r[i] / m[i] for i in range(N)] for row in rows_])
        CT = arb_mat([[cols[j][i] for j in range(len(cols))] for i in range(N)])
        J = -(Rw * CT)
        for k in range(n):
            J[k, k] = J[k, k] + Fv[n + k, 0]
            J[n + k, k] = J[n + k, k] + D2[k]
        if fix is not None:  # replace column fix by dF/da
            mA = sum((v[k] * (phi(A - p[k]) + phi(A + p[k])) for k in range(n)), arb(0)) + (
                c * phi(A) if centre else 0
            )
            col = [(phi(A - p[k]) + phi(A + p[k])) / (2 * A * mA) - Fv[k, 0] / A for k in range(n)]
            col += [
                ((A - p[k]) * phi(A - p[k]) - (A + p[k]) * phi(A + p[k])) / (2 * A * mA) - Fv[n + k, 0] / A
                for k in range(n)
            ]
            if centre:
                col.append(2 * phi(A) / (2 * A * mA) - Fv[2 * n, 0] / A)
            for i in range(len(col)):
                J[i, fix] = col[i]
        out["J"] = J
        return out

    def newton(self, z, n, centre, A, fix=None, maxit=14):
        arb, arb_mat = self.arb, self.arb_mat
        st = self.state(z, n, centre, A)
        if st is None:
            return z, A, np.inf, None
        o = self.FJ(st, jac=True, fix=fix)
        nF = max(abs(float(f.mid())) for f in o["F"])
        slow = 0
        for it in range(maxit):
            if nF < self.tol:
                break
            if slow >= 4 and fix is None:  # v6c: plain steps only; stagnation (4 iterations below 1.5x)
                break
            nF_prev = nF
            try:
                d = o["J"].solve(arb_mat([[f.mid()] for f in o["F"]]))
            except (ZeroDivisionError, ValueError):
                break
            lam, ok = 1.0, False
            for _ in range(6):
                zn = [(z[i] - lam * d[i, 0]).mid() if i != fix else z[i] for i in range(len(z))]
                An = (A - lam * d[fix, 0]).mid() if fix is not None else A
                pn = [float(t.mid()) for t in zn[:n]]
                if (not n or (pn[0] > 0 and all(pn[i] < pn[i + 1] for i in range(n - 1)))) and float(
                    An.mid()
                ) > 0.5:
                    stn = self.state(zn, n, centre, An)
                    if stn is not None:
                        on = self.FJ(stn, jac=True, fix=fix)
                        nFn = max(abs(float(f.mid())) for f in on["F"])
                        if nFn < nF:
                            z, A, st, o, nF, ok = zn, An, stn, on, nFn, True
                            break
                lam /= 2
            if not ok:
                break
            slow = slow + 1 if nF > nF_prev / 1.5 else 0
        o["st"] = st
        return z, A, nF, o

    def D_at(self, st, ts):
        arb = self.arb
        xs, r, N = st["xs"], st["r"], st["N"]
        res = []
        for t in ts:
            e1 = [self.phi(x - t) for x in xs]
            e2 = [self.phi(x + t) for x in xs]
            d0 = sum(((e1[i] + e2[i]) * r[i] for i in range(N)), arb(0))
            d1 = sum((((xs[i] - t) * e1[i] - (xs[i] + t) * e2[i]) * r[i] for i in range(N)), arb(0))
            d2 = sum(
                ((((xs[i] - t) ** 2 - 1) * e1[i] + ((xs[i] + t) ** 2 - 1) * e2[i]) * r[i] for i in range(N)),
                arb(0),
            )
            res.append((d0, d1, d2))
        return res

    def flags(self, z, n, centre, o):
        v = [float(t.mid()) for t in z[n : 2 * n]] + ([float(z[2 * n].mid())] if centre else [])
        d2 = [float(t.mid()) for t in o["D2"]]
        f = {}
        if v and min(v) < 0:
            f["death"] = int(np.argmin(v))
        if d2 and max(d2) > 0:
            f["split"] = int(np.argmax(d2))
        if centre and float(o["D2c"].mid()) > 0:
            f["split0"] = True
        if not centre and float((o["D0c"] - 1).mid()) > 0:
            f["birth0"] = True
        return f

    def kkt(self, z, n, centre, o, per_gap=16):
        """full arb check: per-gap sampled D - 1 (min refined by Newton on D'), outer region, D'' at atoms, weights"""
        arb = self.arb
        st = o["st"]
        A = st["A"]
        at = ([0.0] if centre else []) + [float(t.mid()) for t in z[:n]]
        edges = at if centre else ([-at[0]] + at if n else [])
        gmax, dips = -np.inf, []
        for g in range(len(edges) - 1):
            lo, hi = max(edges[g], 0.0), edges[g + 1]
            ts = [lo + (hi - lo) * (k + 0.5) / per_gap for k in range(per_gap)]
            if not centre and g == 0:
                ts = [0.0] + ts
            vals = self.D_at(st, [arb(t) for t in ts])
            dv = [float((d0 - 1).mid()) for d0, _, _ in vals]
            gmax = max(gmax, max(dv))
            k = int(np.argmin(dv))
            t = arb(ts[k])
            if ts[k] != 0.0:
                for _ in range(5):
                    ((d0, d1, d2),) = self.D_at(st, [t])
                    t = (t - d1 / d2).mid()
            ((d0, d1, d2),) = self.D_at(st, [t])
            dips.append((float((d0 - 1).mid()), float(t.mid())))
        a = float(A.mid())
        pn = at[-1] if at else 0.0
        outer = [pn + 0.05 + (a + 4 - pn - 0.05) * k / 23 for k in range(24)]
        ov = max(float((d0 - 1).mid()) for d0, _, _ in self.D_at(st, [arb(t) for t in outer]))
        v = [float(t.mid()) for t in z[n : 2 * n]] + ([float(z[2 * n].mid())] if centre else [])
        d2 = [float(t.mid()) for t in o["D2"]] + ([float(o["D2c"].mid())] if centre else [])
        S = min(-d for d, _ in dips) if dips else np.nan
        ok = gmax < 0 and ov < 0 and min(v) > 0 and max(d2) < 0 and all(d < 0 for d, _ in dips)
        S_at = max(dips)[1] if dips else np.nan  # the dip closest to 0 is the flattest gap
        return dict(ok=bool(ok), S=S, S_at=S_at, gmax=gmax, outer=ov, minw=min(v), maxd2=max(d2), dips=dips)


def mpsweep(a_start, a_end, da=0.05, prec=128, log=None, resume=False):
    """arb continuation from the stored double row at a_start: steps of da, reporting rows (full arb KKT check) on the
    0.25 grid; events from sign changes of the arb flags (death, D''(atom) > 0, D''(0) > 0 with a centre atom, D(0) > 1
    without), located by arb bisection to 1e-4 and applied along arb paths (young parameter fixed, a free) to a* + 0.02.
    """
    T0 = time.time()
    el = lambda: f"[{time.time() - T0:.0f}s]"
    out = lambda s: (say(s), log.write(s + "\n"), log.flush()) if log else say(s)
    M = MPC(prec)
    arb = M.arb
    fn = os.path.join(HERE, "results_q496_npmle_uniform_mpsweep.json")
    if resume:  # v6: continue from the last arb row
        dd = json.load(open(fn))
        r = dd["rows"][-1]
        z = [arb(t) for t in r["p_str"]] + [arb(t) for t in r["v"]] + ([arb(r["c"])] if r["centre"] else [])
    else:
        d = json.load(open(RESULTS))
        r = min(d["rows"], key=lambda rr: abs(rr["a"] - a_start))
        z = [arb(t) for t in r["p"]] + [arb(t) for t in r["v"]] + ([arb(r["c"])] if r["centre"] else [])
    n, centre = r["n"], r["centre"]
    A = arb(r["a"])
    a = r["a"]
    z, A, nF, o = M.newton(z, n, centre, A, maxit=30)
    rows, events = (
        ([], []) if not resume else (dd["rows"][:-1], [e for e in dd["events"] if e["a_star"] < r["a"]])
    )

    def report(z, n, centre, o, a):
        k = M.kkt(z, n, centre, o)
        nF = max(abs(float(f.mid())) for f in o["F"])
        row = dict(
            a=a,
            K=K_of(n, centre),
            n=n,
            centre=centre,
            maxF=nF,
            S=k["S"],
            S_at=k["S_at"],
            gmax=k["gmax"],
            outer=k["outer"],
            minw=k["minw"],
            maxd2=k["maxd2"],
            ok=k["ok"],
            p=[float(t.mid()) for t in z[:n]],
            v=[float(t.mid()) for t in z[n : 2 * n]],
            c=(float(z[2 * n].mid()) if centre else None),
            p_str=[t.mid().str(30, radius=False) for t in z[:n]],
        )
        rows.append(row)
        out(
            f"MP a = {a:6.2f}: K = {row['K']:3d}, max|F| {nF:.1e}, S = {k['S']:.3e} at {k['S_at']:.2f}, off-atom max(D-1) {k['gmax']:+.2e}, "
            f"outer {k['outer']:+.1e}, min w {k['minw']:.2e}, max D'' {k['maxd2']:+.2e}: {'KKT ok' if k['ok'] else 'KKT FAILS'}  {el()}"
        )
        json.dump(dict(rows=rows, events=events, prec=prec), open(fn, "w"))
        return k["ok"]

    out(
        f"=== arb continuation (prec {prec}) from a = {a} (K = {K_of(n, centre)}, polished max|F| {nF:.1e}) to {a_end} ==="
    )
    report(z, n, centre, o, a)
    hist = [(a, z)]
    grid = [round(a + 0.25 * k, 6) for k in range(1, int(round((a_end - a) / 0.25)) + 1)]
    gi = 0
    while gi < len(grid):
        tgt = grid[gi]
        an = min(round(a + da, 10), tgt)
        if len(hist) >= 3:  # v6: quadratic (Lagrange) predictor
            (a0_, z0_), (a1, z1), (a2, z2) = hist[-3], hist[-2], hist[-1]
            L0 = (an - a1) * (an - a2) / ((a0_ - a1) * (a0_ - a2))
            L1 = (an - a0_) * (an - a2) / ((a1 - a0_) * (a1 - a2))
            L2 = (an - a0_) * (an - a1) / ((a2 - a0_) * (a2 - a1))
            seed = [(z0_[i] * L0 + z1[i] * L1 + z2[i] * L2).mid() for i in range(len(z2))]
        elif len(hist) >= 2:
            (a1, z1), (a2, z2) = hist[-2], hist[-1]
            seed = [(z2[i] + (z2[i] - z1[i]) * (an - a2) / (a2 - a1)).mid() for i in range(len(z2))]
        else:
            seed = [(t * an / a).mid() if i < n else t for i, t in enumerate(z)]
        zn, An, nFn, on = M.newton(seed, n, centre, arb(an))
        if nFn > M.tol:
            da /= 2
            if da < 1e-6:
                out(f"   arb continuation failed at a = {a} {el()}")
                break
            continue
        fl = M.flags(zn, n, centre, on)
        if not fl:
            a, z, o = an, zn, on
            hist.append((a, z))
            da = min(da * 1.5, 0.05)
            if abs(a - tgt) < 1e-9:
                if not report(z, n, centre, o, a):
                    out("   full KKT check fails without a flagged event: stopping")
                    break
                gi += 1
            continue
        # bisection with the old structure
        lo, zlo, hi, zhi, ohi = a, z, an, zn, on
        while hi - lo > 1e-4:
            mid = 0.5 * (lo + hi)
            seed = [(zlo[i] + (zhi[i] - zlo[i]) * (mid - lo) / (hi - lo)).mid() for i in range(len(z))]
            zm, Am, nFm, om = M.newton(seed, n, centre, arb(mid))
            if nFm > M.tol:  # v6d: retry from the lower-end solution before giving up
                zm, Am, nFm, om = M.newton(list(zlo), n, centre, arb(mid), maxit=30)
            if nFm > M.tol:
                out(f"      bisection stopped at [{lo:.5f}, {hi:.5f}]: Newton failed at the midpoint")
                break
            if M.flags(zm, n, centre, om):
                hi, zhi, ohi = mid, zm, om
            else:
                lo, zlo = mid, zm
        fl = M.flags(zhi, n, centre, ohi)
        Kb = K_of(n, centre)
        p = zhi[:n]
        v = zhi[n : 2 * n]
        c = zhi[2 * n] if centre else None
        if "death" in fl:
            k = fl["death"]
            if k == n:
                typ = "death-centre"
                zc = list(p) + list(v)
                nc, cc = n, False
            else:
                typ = "death-pair"
                zc = (
                    [t for i, t in enumerate(p) if i != k]
                    + [t for i, t in enumerate(v) if i != k]
                    + ([c] if centre else [])
                )
                nc, cc = n - 1, centre
            zc, Ac, nFc, oc = M.newton(zc, nc, cc, arb(hi))
            a_app, path, pz = hi, [], []
        else:
            if "birth0" in fl:  # priority: centre birth (see the v4 note)
                typ = "birth-centre"
                nc, cc, fix = n, True, 2 * n
                z0 = list(p) + list(v) + [arb("1e-12")]
                par = [1e-12]
            elif "split0" in fl:
                typ = "split-centre"
                nc, cc, fix = n + 1, False, 0
                z0 = [arb("0.01")] + list(p) + [c / 2] + list(v)
                par = [0.01]
            else:
                k = fl["split"]
                typ = "split-pair"
                nc, cc, fix = n + 1, centre, k
                pk = p[k]
                z0 = (
                    list(p[:k])
                    + [pk - arb("0.01"), pk + arb("0.01")]
                    + list(p[k + 1 :])
                    + list(v[:k])
                    + [v[k] / 2, v[k] / 2]
                    + list(v[k + 1 :])
                    + ([c] if centre else [])
                )
                par = [float((pk - arb("0.01")).mid())]
            zc, Ac, nFc, oc = M.newton(z0, nc, cc, arb(hi), fix=fix)
            path = [(par[0], float(Ac.mid()))]
            pz = [(arb(par[0]), Ac, zc)]
            step = 0.01
            fac = 3.0
            while nFc < M.tol and float(Ac.mid()) < hi + 0.02 and len(path) < 300:
                if typ == "birth-centre":
                    newpar = path[-1][0] * fac
                elif typ == "split-centre":
                    newpar = path[-1][0] + step
                else:
                    newpar = path[-1][0] - step
                if len(pz) >= 2:  # secant predictor in the path parameter (v5)
                    (q1, A1, z1), (q2, A2, z2) = pz[-2], pz[-1]
                    t = (arb(newpar) - q2) / (q2 - q1)
                    zt = [(z2[i] + t * (z2[i] - z1[i])).mid() for i in range(len(z2))]
                    At0 = (A2 + t * (A2 - A1)).mid()
                else:
                    zt, At0 = list(zc), Ac
                zt[fix] = arb(newpar)
                zt2, At, nFt, ot = M.newton(zt, nc, cc, At0, fix=fix, maxit=10)
                if nFt < M.tol:
                    zc, Ac, nFc, oc = zt2, At, nFt, ot
                    path.append((newpar, float(Ac.mid())))
                    pz.append((arb(newpar), Ac, zc))
                    step = min(step * 1.5, 0.05)
                    fac = min(1 + (fac - 1) * 1.5, 3.0)
                else:
                    step /= 2
                    fac = 1 + (fac - 1) / 2
                    if step < 1e-5 or fac < 1.001:
                        break
            a_app = float(Ac.mid())
            if nFc < M.tol:  # release: plain Newton at the path end (arb is well conditioned)
                zc, Ac, nFc, oc = M.newton(zc, nc, cc, arb(a_app))
        ev = dict(
            type=typ,
            a_star=0.5 * (lo + hi),
            a_lo=lo,
            a_hi=hi,
            a_star_path=(path[0][1] if path else None),
            a_applied=a_app,
            K_before=Kb,
            K_after=K_of(nc, cc),
            path_len=len(path),
            flags=list(fl.keys()),
        )
        events.append(ev)
        out(
            f"   MP EVENT {typ:13s} at a* = {ev['a_star']:.5f} (path start {ev['a_star_path']}), K {Kb} -> {K_of(nc, cc)}, "
            f"applied at {a_app:.5f}, max|F| {nFc:.1e}, flags {list(fl.keys())}  {el()}"
        )
        if nFc > M.tol:
            out("   event application failed: stopping")
            break
        k2 = M.kkt(zc, nc, cc, oc)
        fl2 = M.flags(zc, nc, cc, oc)
        out(
            f"      after the event: KKT {'ok' if k2['ok'] else 'FAILS'}, flags {list(fl2.keys())}, S {k2['S']:.2e}, max D'' {k2['maxd2']:+.2e}"
        )
        a, z, n, centre, o = a_app, zc, nc, cc, oc
        hist = [(a, z)]
        if (
            typ != "death-pair"
            and typ != "death-centre"
            and len(pz) >= 2
            and float(pz[-2][1].mid()) < a - 1e-6
        ):
            hist = [(float(pz[-2][1].mid()), pz[-2][2]), (a, z)]  # v5: tangent from the last two path points
        while gi < len(grid) and grid[gi] <= a + 1e-9:  # grid point(s) passed during the path: come back
            g = grid[gi]
            zb, Ab, nFb, ob = M.newton(z, n, centre, arb(g))
            if nFb < M.tol and not M.flags(zb, n, centre, ob):
                report(zb, n, centre, ob, g)
            else:
                out(f"   grid point {g} lies inside the event path; not reported")
            gi += 1
        json.dump(dict(rows=rows, events=events, prec=prec), open(fn, "w"))
    return rows, events


def analyse():
    """POST-HOC (not pre-registered) summaries of the stored sweep: event spacing and local exponent, the signal S per
    phase, and the gap law against the distance from the edge of the data."""
    d = json.load(open(RESULTS))
    rows, events = d["rows"], d["events"]
    ev = [(e["a_star"], e["K_after"], e["type"]) for e in events if e.get("K_after") is not None]
    say("events (a*, K after, type), spacing to the next:")
    for i, (a_, k_, t_) in enumerate(ev):
        say(
            f"   {a_:8.4f}  K -> {k_:3d}  {t_:13s}"
            + (f"  da = {ev[i + 1][0] - a_:.4f}" if i + 1 < len(ev) else "")
        )
    A = np.array([e[0] for e in ev])
    K = np.array([e[1] for e in ev], float)
    mid = 0.5 * (A[1:] + A[:-1])
    da = np.diff(A)
    for lo_ in (5.0, 10.0, 15.0, 20.0):
        sel = mid >= lo_
        if sel.sum() >= 4:
            sl = np.polyfit(np.log(mid[sel]), np.log(da[sel]), 1)[0]
            say(
                f"   local exponent from event spacing, midpoints >= {lo_}: 1 - dlog(da)/dlog(a) = {1 - sl:.4f} ({sel.sum()} spacings)"
            )
    # two-parameter check with a subleading term, all events a* >= 5: K = c a^(4/3) + d a^(2/3) + b and K = c a + d log a + b
    sel = A >= 5
    for name, cols in (
        ("c a^(4/3) + d a^(2/3) + b", [A ** (4 / 3), A ** (2 / 3), np.ones_like(A)]),
        ("c a + d a^(1/2) + b", [A, np.sqrt(A), np.ones_like(A)]),
        ("c a^2 + d a + b", [A**2, A, np.ones_like(A)]),
        ("c a log a + d a + b", [A * np.log(A), A, np.ones_like(A)]),
    ):
        X = np.vstack([c_[sel] for c_ in cols]).T
        cf, *_ = np.linalg.lstsq(X, K[sel], rcond=None)
        say(
            f"   events a* >= 5, K = {name}: coef {np.round(cf, 5).tolist()}, rms {np.sqrt(np.mean((X @ cf - K[sel])**2)):.4f}"
        )
    # extension with the arb continuation (post-hoc): double events below the arb start, arb events above it
    fnm = os.path.join(HERE, "results_q496_npmle_uniform_mpsweep.json")
    if os.path.exists(fnm):
        dm = json.load(open(fnm))
        a0 = dm["rows"][0]["a"] if dm["rows"] else np.inf
        evm = [(e["a_star"], e["K_after"]) for e in dm["events"] if e.get("K_after") is not None]
        comb = [(a_, k_) for a_, k_, _ in ev if a_ < a0] + evm
        Ac = np.array([c_[0] for c_ in comb])
        Kc = np.array([c_[1] for c_ in comb], float)
        say(
            f"events extended with the arb continuation (from a = {a0}): {len(evm)} arb events, last a* = {Ac.max():.4f}, K = {int(Kc.max())}"
        )
        for lo_ in (14.875, 5.0):
            sel = Ac >= lo_
            for pfix in (1.0, 4 / 3, 2.0):
                X = np.vstack([Ac[sel] ** pfix, np.ones(sel.sum())]).T
                cf, *_ = np.linalg.lstsq(X, Kc[sel], rcond=None)
                say(
                    f"   a* >= {lo_}: p = {pfix:.4f} fixed: c = {cf[0]:.5f}, b = {cf[1]:+.4f}, rms {np.sqrt(np.mean((X @ cf - Kc[sel])**2)):.4f}"
                )
            try:
                from scipy.optimize import curve_fit

                f_ = lambda a, c, p, b: c * a**p + b
                (c_, p_, b_), cov = curve_fit(f_, Ac[sel], Kc[sel], p0=(0.4, 1.33, 1.0), maxfev=20000)
                say(
                    f"   a* >= {lo_}: p free: c = {c_:.5f}, p = {p_:.4f}, b = {b_:+.4f}, rms {np.sqrt(np.mean((f_(Ac[sel], c_, p_, b_) - Kc[sel])**2)):.4f}"
                )
            except Exception as e_:
                say(f"   p free fit failed: {e_}")
            X = np.vstack([Ac[sel] ** (4 / 3), Ac[sel] ** (2 / 3), np.ones(sel.sum())]).T
            cf, *_ = np.linalg.lstsq(X, Kc[sel], rcond=None)
            say(
                f"   a* >= {lo_}: K = c a^(4/3) + d a^(2/3) + b: {np.round(cf, 5).tolist()}, rms {np.sqrt(np.mean((X @ cf - Kc[sel])**2)):.4f}"
            )
    # signal per phase
    say("signal S (shallowest dip of D - 1 between atoms) per K-phase: max over the grid points of the phase")
    ph = {}
    for r in rows:
        ph.setdefault(r["K"], []).append(r)
    pts = []
    for k_ in sorted(ph):
        rr = max(ph[k_], key=lambda r: r["S"] if np.isfinite(r["S"]) else -1)
        say(
            f"   K = {k_:3d}: a in [{ph[k_][0]['a']:.2f}, {ph[k_][-1]['a']:.2f}], max S = {rr['S']:.2e} at a = {rr['a']:.2f}, nu there {rr['nu']:.1e}"
        )
        if k_ >= 4 and np.isfinite(rr["S"]) and rr["S"] > 0:
            pts.append((rr["a"], rr["S"]))
    if len(pts) >= 6:
        P = np.array(pts)
        for nm, xv in (("a", P[:, 0]), ("a^(2/3)", P[:, 0] ** (2 / 3))):
            cf = np.polyfit(xv, np.log(P[:, 1]), 1)
            say(
                f"   log(max S per phase) = {cf[1]:+.3f} {cf[0]:+.4f} {nm}; rms {np.sqrt(np.mean((np.polyval(cf, xv) - np.log(P[:, 1]))**2)):.3f}"
            )
    # gap law on the last row
    r = rows[-1]
    p = np.array(r["p"])
    a = r["a"]
    at = np.concatenate([[0.0] if r["centre"] else [], p])
    g = np.diff(at)
    dist = a - 0.5 * (at[1:] + at[:-1])
    say(f"gap law at a = {a}: (distance of the gap midpoint from the edge a, gap)")
    for dd, gg in zip(dist[::-1], g[::-1]):
        say(f"   d = {dd:8.3f}  gap = {gg:.4f}  gap^-3 = {gg**-3:.4f}  gap^-3/d = {gg**-3/dd:.4f}")
    sel = dist > 3
    if sel.sum() >= 4:
        sl = np.polyfit(np.log(dist[sel]), np.log(g[sel]), 1)[0]
        say(f"   slope d log(gap)/d log(d) for d > 3: {sl:.4f}")
    say(f"   outermost atom at {p[-1]:.4f} = a - {a - p[-1]:.4f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "control":
        control()
    elif mode == "sweep":
        AMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        resume = len(sys.argv) > 3 and sys.argv[3] == "resume"
        with open(os.path.join(HERE, "q496_npmle_uniform.log"), "a", encoding="utf-8") as log:
            log.write(
                f"\n=== sweep a = 1 .. {AMAX} by 0.25{' (resume)' if resume else ''}, started {time.ctime()} ===\n"
            )
            sweep(AMAX, log=log, resume=resume)
    elif mode == "fit":
        fit()
    elif mode == "analyse":
        analyse()
    elif mode == "mpsweep":
        prec = int(os.environ.get("MPPREC", "128"))
        with open(os.path.join(HERE, "q496_npmle_uniform_mp.log"), "a", encoding="utf-8") as log:
            mpsweep(
                float(sys.argv[2]),
                float(sys.argv[3]),
                prec=prec,
                log=log,
                resume=(len(sys.argv) > 4 and sys.argv[4] == "resume"),
            )
    elif mode == "mp":
        prec = int(os.environ.get("MPPREC", "160"))
        res = mp_check([float(t) for t in sys.argv[2:]], prec=prec)
        fn = os.path.join(HERE, "results_q496_npmle_uniform_mp.json")
        old = json.load(open(fn)) if os.path.exists(fn) else []
        json.dump(old + res, open(fn, "w"))
