"""q935 (7 Oct 2026): generic text copy of q930 (m derived from the state, the two A_s states and the grid given on the command line or in the environment: Q935_GRID, Q935_ODIR); used for the tower of splits after T64. Text of q930 follows: q930 (7 Oct 2026): young-pair instrument for the split K 61 -> K 62 of the capacity (T64, registered in amendment 1707 before any K 62 state existed).
Modes (all in 200-bit ball arithmetic with q920 Cap; K 61 states have grid_A 42.0):
  py q930_split_cap.py As OUT.json                 zero A_s of D''(0) along the K 61 branch by secant steps in A (states A = 41.7325 and 41.7375 of capacity_odd)
  py q930_split_cap.py scan STATE.json OUT.json    scan of g(eps) = D'(u_1)/eps at fixed A over eps = 0.01 * 1.3^k, k = 0..16 (K 62 seed from the K 61 state); for the case of a sign change from + to - also the refined root, the full Newton, the signals and the branch in eps (bordered system with A unknown)
Definitions (1707): seed u = (eps, u_1..u_29), w = (wc/2, w_1..w_29), we; fixed-eps unknowns u_2..u_30, w_1..w_30, we from the 60 rows other than D'(u_1) = 0; g = residual row D'(u_1) divided by eps.
"""

import copy, json, os, sys, time
from flint import arb, arb_mat
import q920_cap_flint as F
import q921_cap_tools as T
from q920_cap_flint import Cap, maxabs, norm2

t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
GRID = os.environ.get("Q935_GRID", "43.5")
ODIR = os.environ.get("Q935_ODIR", "capacity_split")
TOL = arb("1e-38")
VERBOSE = [False]


def d2_centre(cap, v):
    m = cap.m
    rows, _, _, cen, Q, lQ = cap.tables(v[:m], v[m : 2 * m], v[2 * m], v[2 * m + 1])
    return cap.Dpp(cen[0], cen[1], cen[2], lQ)


def mode_As(out, f_lo, f_hi):
    pts = []
    for A_str, fpath in (("lo", f_lo), ("hi", f_hi)):
        A, u, w, we, wc, d = F.load_state(fpath)
        cap = Cap(A, len(u), True, grid_A=d.get("grid_A"))
        v = [t.mid() for t in (u + w + [we, wc])]
        v, res, ok = T.newton_full(cap, v, say, tol=TOL)
        if not ok:
            sys.exit("VOID q930: base state does not polish")
        pts.append((float(A), cap, v, d2_centre(cap, v)))
        say(
            f"{fpath} A {A}: D''(0) = {pts[-1][3].str(8, radius=False)} (residual {maxabs(res).str(2, radius=False)})"
        )
    (A0, c0, v0, g0), (A1, c1, v1, g1) = pts
    hist = [(A0, g0), (A1, g1)]
    for it in range(12):
        An = A1 - float(g1.mid()) * (A1 - A0) / float((g1 - g0).mid())  # amendment 1731: no rounding of A
        cn, vn, rn, ok = T.step_in_A(c1, v1, f"{An:.15f}", say)
        if not ok:
            sys.exit("VOID q930: corrector failed in the secant for A_s")
        gn = d2_centre(cn, vn)
        hist.append((An, gn))
        say(f"secant {it}: A {An:.9f}, D''(0) = {gn.str(8, radius=False)}")
        A0, g0, A1, c1, v1, g1 = A1, g1, An, cn, vn, gn
        if abs(g1) < arb("1e-22"):
            break
    else:
        sys.exit("VOID q930: secant for A_s did not reach 1e-22")
    json.dump(
        {"A_s": A1, "d2": float(g1.mid()), "history": [(a, float(g.mid())) for a, g in hist]},
        open(out, "w"),
        indent=1,
    )
    print(f"RESULT q930 As: A_s = {A1:.12f}, D''(0) = {g1.str(5, radius=False)}", flush=True)


def seed62(state, eps):
    A, u, w, we, wc, d = state
    if wc is None:
        sys.exit("VOID q930: seed state without centre atom")
    return [eps] + [t for t in u] + [wc / 2] + [t for t in w] + [we]


def rowscols(m):
    return [i for i in range(2 * m + 1) if i != m], list(range(1, 2 * m + 1))


def fixed_eps_solve(cap, v, eps, tol=TOL, maxit=40):
    """Newton on the 60 unknowns (all but u_1) from the 60 rows other than D'(u_1); returns (v, res (all 61 rows), ok)."""
    m = cap.m
    n = cap.n
    rows, cols = rowscols(m)
    v = list(v)
    v[0] = eps
    rn0 = None
    for it in range(maxit):
        res, J, _ = cap.res_jac(v, True, False)
        r = [res[i] for i in rows]
        if VERBOSE[0]:
            say(
                f"      fixed-eps iteration {it}: max|r| {maxabs(r).str(4, radius=False)}, eps radius {v[0].rad()}"
            )
        if maxabs(r) < tol:
            return v, res, True
        rn = norm2(r)
        rn0 = rn if rn0 is None else rn0
        if not (rn < 1e4 * rn0):
            fixed_eps_solve.why = f"residual norm {rn.str(3, radius=False)} above 1e4 times the first {rn0.str(3, radius=False)} at iteration {it}"
            return v, res, False
        Js = arb_mat(n - 1, n - 1, [J[i, k] for i in rows for k in cols])
        dx = T.solve_lin(Js, [-t for t in r])
        lam = arb(1)
        acc = False
        for _ in range(14):
            vn = [v[0]] + [(v[1 + i] + lam * dx[i]).mid() for i in range(n - 1)]
            if cap.feasible(vn):
                v = vn
                acc = True
                break
            lam = lam / 2
        if not acc:
            fixed_eps_solve.why = f"no feasible step at iteration {it} (residual {maxabs(r).str(3, radius=False)}, smallest mass {min(v[cap.m:2 * cap.m]).str(3, radius=False)})"
            return v, res, False
    res, _, _ = cap.res_jac(v, False, False)
    ok = maxabs([res[i] for i in rows]) < tol
    if not ok:
        fixed_eps_solve.why = (
            f"{maxit} iterations, residual {maxabs([res[i] for i in rows]).str(3, radius=False)}"
        )
    return v, res, ok


def predict_eps(cap, v, eps_new):
    """predictor dx/deps at the current point (fixed A)"""
    m = cap.m
    n = cap.n
    rows, cols = rowscols(m)
    res, J, _ = cap.res_jac(v, True, False)
    Js = arb_mat(n - 1, n - 1, [J[i, k] for i in rows for k in cols])
    dxde = T.solve_lin(Js, [-J[i, 0] for i in rows])
    de = eps_new - v[0]
    return [eps_new] + [(v[1 + i] + de * dxde[i]).mid() for i in range(n - 1)]


def bordered_solve(cap0, v, eps, A, tol=TOL, maxit=40):
    """eps fixed; unknowns u_2..u_30, w_1..w_30, we and A; all 61 rows."""
    n = cap0.n
    v = list(v)
    v[0] = eps
    rn0 = None
    for it in range(maxit):
        cap = copy.copy(cap0)
        cap.A = A
        res, J, b = cap.res_jac(v, True, True)
        if maxabs(res) < tol:
            return cap, v, A, res, True
        rn = norm2(res)
        rn0 = rn if rn0 is None else rn0
        if not (rn < 1e4 * rn0):
            return cap, v, A, res, False
        M = arb_mat(n, n, [(J[i, k + 1] if k < n - 1 else b[i]) for i in range(n) for k in range(n)])
        dz = T.solve_lin(M, [-t for t in res])
        lam = arb(1)
        acc = False
        for _ in range(14):
            vn = [v[0]] + [(v[1 + i] + lam * dz[i]).mid() for i in range(n - 1)]
            An = (A + lam * dz[n - 1]).mid()
            capn = copy.copy(cap0)
            capn.A = An
            if capn.feasible(vn):
                v = vn
                A = An
                acc = True
                break
            lam = lam / 2
        if not acc:
            return cap, v, A, res, False
    cap = copy.copy(cap0)
    cap.A = A
    res, _, _ = cap.res_jac(v, False, False)
    return cap, v, A, res, maxabs(res) < tol


def predict_bordered(cap, v, A, eps_new):
    n = cap.n
    res, J, b = cap.res_jac(v, True, True)
    M = arb_mat(n, n, [(J[i, k + 1] if k < n - 1 else b[i]) for i in range(n) for k in range(n)])
    dz = T.solve_lin(M, [-J[i, 0] for i in range(n)])
    de = eps_new - v[0]
    return [eps_new] + [(v[1 + i] + de * dz[i]).mid() for i in range(n - 1)], (A + de * dz[n - 1]).mid()


def verdict_state(cap, v, res):
    m = cap.m
    cen, off = cap.signals(v)
    wmin = min(v[m : 2 * m] + [v[2 * m]])
    ordered = all(v[i] < v[i + 1] for i in range(m - 1))
    opt = ordered and wmin > 0 and cen <= 0 and off <= 0 and maxabs(res) < arb("1e-20")
    return (
        (
            "OPTIMAL"
            if opt
            else (
                "centre positive" if not (cen <= 0) else ("off-atom positive" if not (off <= 0) else "other")
            )
        ),
        cen,
        off,
        wmin,
        ordered,
    )


def save_state(path, cap, v, res, A, verdict, extra):
    F.dump_state(path, A.str(30, radius=False), cap, v, res, verdict, extra)


def walk_eps(cap, v, eps_to, min_step=1e-5):
    """continuation in eps at fixed A from the converged point v (v[0] is its eps) to eps_to with adaptive substeps (halved on failure, doubled after a success); repair of the instrument of amendment 1708 (the full steps of the scan did not converge from the predictor over a factor 1.3)"""
    cur = list(v)
    e = float(cur[0].mid())
    tgt = float(eps_to.mid())
    step = tgt - e
    nsub = 0
    while True:
        last = abs(tgt - e) <= abs(step) * 1.000001
        e_try = eps_to.mid() if last else (cur[0] + arb(step)).mid()
        vp = predict_eps(cap, cur, e_try)
        vn, res_n, ok = fixed_eps_solve(cap, vp, e_try)
        if ok:
            cur = vn
            e = float(e_try.mid())
            nsub += 1
            if last:
                return cur, res_n, True, nsub
            step = step * 2 if abs(step * 2) < abs(tgt - e) else (tgt - e)
        else:
            step = step / 2
            if abs(step) < min_step:
                return cur, res_n, False, nsub


def walk_bordered(cap0, v, A, eps_to, min_step=1e-5):
    """the same adaptive continuation for the bordered system (A unknown)"""
    cur = list(v)
    Ac = A
    e = float(cur[0].mid())
    tgt = float(eps_to.mid())
    step = tgt - e
    nsub = 0
    capc = copy.copy(cap0)
    capc.A = Ac
    while True:
        last = abs(tgt - e) <= abs(step) * 1.000001
        e_try = eps_to.mid() if last else (cur[0] + arb(step)).mid()
        vp, Ap = predict_bordered(capc, cur, Ac, e_try)
        capn, vn, An, resn, ok = bordered_solve(cap0, vp, e_try, Ap)
        if ok:
            cur, Ac, capc, e = vn, An, capn, float(e_try.mid())
            nsub += 1
            if last:
                return capc, cur, Ac, resn, True, nsub
            step = step * 2 if abs(step * 2) < abs(tgt - e) else (tgt - e)
        else:
            step = step / 2
            if abs(step) < min_step:
                return capc, cur, Ac, resn, False, nsub


def run_targets(cap0, vroot, A0, targets, grid, m):
    """branch points of the bordered system (eps fixed, A unknown) for the eps values in `targets`, one after the other from the root. Continuation in t = eps^2: the predictor is the tangent at the first step and the linear extrapolation of (u_2.., w, we, A) from the last two converged points afterwards; the step in t is halved on failure (minimum 1e-7) and doubled after a success."""
    n = cap0.n
    out = []
    hist = [(float((vroot[0] * vroot[0]).mid()), list(vroot), A0, cap0)]
    for tgt in targets:
        t_tgt = tgt * tgt
        frac = 1.0
        ok_target = False
        while True:
            t_last, v_last, A_last, cap_last = hist[-1]
            dt = (t_tgt - t_last) * frac
            if abs(dt) < 1e-7:
                break
            reach = frac >= 1.0
            t_try = t_tgt if reach else t_last + dt
            e_try = arb(tgt).mid() if reach else arb(t_try).sqrt().mid()
            if len(hist) >= 2:
                t_p, v_p, A_p, _ = hist[-2]
                s = (t_try - t_last) / (t_last - t_p)
                vp = [e_try] + [(v_last[i] + s * (v_last[i] - v_p[i])).mid() for i in range(1, n)]
                Ap = (A_last + s * (A_last - A_p)).mid()
            else:
                vp, Ap = predict_bordered(cap_last, v_last, A_last, e_try)
            capn, vn, An, resn, ok = bordered_solve(cap0, vp, e_try, Ap, maxit=15)
            if ok:
                hist.append((t_try, vn, An, capn))
                frac = min(1.0, frac * 2)
                if reach:
                    ok_target = True
                    break
            else:
                frac = frac / 2
        if not ok_target:
            out.append({"eps": tgt, "failed": True})
            say(f"branch eps {tgt:.6f}: NOT CONVERGED")
            break
        t_last, vn, An, capn = hist[-1]
        verd, cen, off, wmin, ordered = verdict_state(capn, vn, resn)
        out.append(
            {
                "eps": float(vn[0].mid()),
                "A": An.str(25, radius=False),
                "A_f": float(An.mid()),
                "verdict": verd,
                "centre": float(cen.mid()),
                "off": float(off.mid()),
                "w1": float(vn[m].mid()),
            }
        )
        say(
            f"branch eps {vn[0].str(6, radius=False)}: A = {An.str(15, radius=False)}, {verd}, centre {cen.str(4, radius=False)}, off {off.str(4, radius=False)} ({len(hist) - 1} points so far)"
        )
        save_state(
            os.path.join(
                ODIR,
                f"results_q220_gauss_K{2 * cap0.m + 2}_A{float(An.mid()):.6f}_eps{float(vn[0].mid()):.5f}.json",
            ),
            capn,
            vn,
            resn,
            An,
            verd,
            {"grid_A": grid, "eps": float(vn[0].mid())},
        )
    return out


NPTS = 13
UP_LIMIT = arb(
    "0.234"
)  # amendment 1709: the scan has 13 points (eps up to 0.23298) and the branch goes up to 0.234


def mode_scan(state_file, out):
    state = F.load_state(state_file)
    A_str, u61, w61, we, wc, d = state
    grid = d.get("grid_A") or GRID
    cap61 = Cap(A_str, len(u61), True, grid_A=grid)
    v61 = [t.mid() for t in (u61 + w61 + [we, wc])]
    v61, r61, ok61 = T.newton_full(cap61, v61, say, tol=TOL)
    if not ok61:
        sys.exit("VOID q930: K 61 base does not polish")
    cen61, _ = cap61.signals(v61)
    say(
        f"K 61 base {state_file}: A {A_str}, wc {wc.str(8, radius=False)}, centre signal {cen61.str(6, radius=False)}, D''(0) {d2_centre(cap61, v61).str(6, radius=False)}"
    )
    m61 = len(u61)
    m = m61 + 1
    cap = Cap(A_str, m, False, grid_A=grid)
    eps_list = [(arb("0.01") * arb("1.3") ** k).mid() for k in range(NPTS)]
    v = seed62((A_str, v61[:m61], v61[m61 : 2 * m61], v61[2 * m61], v61[2 * m61 + 1], d), eps_list[0])
    scan = []
    pts = []
    for k, eps in enumerate(eps_list):
        if k == 0:
            vn, res, ok = fixed_eps_solve(cap, v, eps)
            nsub = 1
        else:
            vn, res, ok, nsub = walk_eps(cap, v, eps)
        if not ok:
            say(
                f"eps {eps.str(5, radius=False)}: NOT CONVERGED ({getattr(fixed_eps_solve, 'why', '')}; {nsub} substeps done)"
            )
            scan.append({"eps": float(eps.mid()), "ok": False})
            break
        g = (res[m] / eps).mid()
        v = vn
        pts.append((eps, [t for t in vn], g))
        scan.append(
            {
                "eps": float(eps.mid()),
                "ok": True,
                "g": float(g.mid()),
                "g_str": g.str(10, radius=False),
                "wc_half": float(vn[m].mid()),
                "substeps": nsub,
            }
        )
        say(
            f"eps {eps.str(5, radius=False)}: g = {g.str(10, radius=False)}  (w_1 {vn[m].str(6, radius=False)}, {nsub} substeps)"
        )
    out_d = {
        "state": state_file,
        "A": float(A_str),
        "centre_signal_K61": float(cen61.mid()),
        "d2_K61": float(d2_centre(cap61, v61).mid()),
        "scan": scan,
        "root": None,
        "branch": [],
    }
    sc = [(p[0], p[2]) for p in pts]
    k_sc = [i for i in range(len(sc) - 1) if sc[i][1] > 0 and sc[i + 1][1] < 0]
    if k_sc and len(pts) == NPTS:
        VERBOSE[0] = os.environ.get("Q930_VERBOSE") == "1"
        k = k_sc[0]
        (e0, g0), (e1, g1) = sc[k], sc[k + 1]
        v0 = pts[k][1]
        v1 = pts[k + 1][1]
        t0_, t1_ = (e0 * e0).mid(), (e1 * e1).mid()
        side = 0
        for it in range(40):
            tn = ((t0_ * g1 - t1_ * g0) / (g1 - g0)).mid()
            en = tn.sqrt().mid()
            vn, res, ok, nsub = walk_eps(cap, v0 if abs(en - e0) < abs(en - e1) else v1, en)
            if not ok:
                out_d["root"] = {
                    "failed": "fixed-eps solve at refinement: " + getattr(fixed_eps_solve, "why", "")
                }
                say(out_d["root"]["failed"])
                break
            gn = (res[m] / en).mid()
            say(
                f"root refinement {it}: eps {en.str(10, radius=False)}, g = {gn.str(6, radius=False)} ({nsub} substeps)"
            )
            if abs(gn) < arb("1e-30"):
                out_d["root"] = {"eps": float(en.mid()), "g": float(gn.mid())}
                vroot = vn
                break
            if gn > 0:
                t0_, g0, v0, e0 = tn, gn, vn, en
                if side == +1:
                    g1 = g1 / 2
                side = +1
            else:
                t1_, g1, v1, e1 = tn, gn, vn, en
                if side == -1:
                    g0 = g0 / 2
                side = -1
        else:
            out_d["root"] = {"failed": "40 refinements"}
    if out_d["root"] and "eps" in out_d["root"]:
        vf, resf, okf = T.newton_full(cap, vroot, say, tol=TOL)
        verd, cen, off, wmin, ordered = verdict_state(cap, vf, resf)
        out_d["root"].update(
            {
                "full_newton_ok": bool(okf),
                "full_residual": float(maxabs(resf).mid()),
                "verdict": verd,
                "centre": float(cen.mid()),
                "off": float(off.mid()),
                "wmin": float(wmin.mid()),
                "ordered": bool(ordered),
                "u1": float(vf[0].mid()),
            }
        )
        say(
            f"K 62 root state: u_1 {vf[0].str(10, radius=False)}, verdict {verd}, centre {cen.str(5, radius=False)}, off {off.str(5, radius=False)}, full residual {maxabs(resf).str(3, radius=False)}"
        )
        os.makedirs(ODIR, exist_ok=True)
        A0 = cap.A
        save_state(
            os.path.join(
                ODIR,
                f"results_q220_gauss_K{2 * cap.m + 2}_A{float(A0.mid()):.6f}_eps{float(vf[0].mid()):.5f}.json",
            ),
            cap,
            vf,
            resf,
            A0,
            verd,
            {"grid_A": grid, "eps": float(vf[0].mid())},
        )
        branch = [
            {
                "eps": float(vf[0].mid()),
                "A": A0.str(25, radius=False),
                "A_f": float(A0.mid()),
                "verdict": verd,
                "centre": float(cen.mid()),
                "off": float(off.mid()),
            }
        ]
        out_d["branch"] = list(branch)
        json.dump(out_d, open(out, "w"), indent=1)  # checkpoint before the branch
        e_root = float(vf[0].mid())
        down = [e_root * 0.9**k for k in range(1, 60) if e_root * 0.9**k >= 0.02]
        up = [e_root * 1.1**k for k in range(1, 60) if e_root * 1.1**k <= 0.234]
        for tg in (down, up):
            branch += run_targets(cap, vf, A0, tg, grid, m)
        out_d["branch"] = sorted(branch, key=lambda r: r["eps"])
    json.dump(out_d, open(out, "w"), indent=1)
    print(
        f"RESULT q930 scan: {state_file}: points {len(scan)}, sign change {bool(k_sc)}, root {out_d['root']}, branch points {len(out_d['branch'])}",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "As":
        mode_As(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) >= 4 and sys.argv[1] == "scan":
        mode_scan(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
