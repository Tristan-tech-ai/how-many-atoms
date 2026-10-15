"""q930 (7 Oct 2026): young-pair instrument for the split K 61 -> K 62 of the capacity (T64, registered in amendment 1707 before any K 62 state existed).
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
GRID = "42.0"
ODIR = "capacity_even62"
TOL = arb("1e-38")
VERBOSE = [False]


def d2_centre(cap, v):
    m = cap.m
    rows, _, _, cen, Q, lQ = cap.tables(v[:m], v[m : 2 * m], v[2 * m], v[2 * m + 1])
    return cap.Dpp(cen[0], cen[1], cen[2], lQ)


def mode_As(out):
    pts = []
    for A_str in ("41.7325", "41.7375"):
        A, u, w, we, wc, d = F.load_state(f"capacity_odd/results_q220_gauss_K61_A{A_str}.json")
        cap = Cap(A, len(u), True, grid_A=d.get("grid_A"))
        v = [t.mid() for t in (u + w + [we, wc])]
        v, res, ok = T.newton_full(cap, v, say, tol=TOL)
        if not ok:
            sys.exit("VOID q930: base state does not polish")
        pts.append((float(A), cap, v, d2_centre(cap, v)))
        say(
            f"A {A_str}: D''(0) = {pts[-1][3].str(8, radius=False)} (residual {maxabs(res).str(2, radius=False)})"
        )
    (A0, c0, v0, g0), (A1, c1, v1, g1) = pts
    hist = [(A0, g0), (A1, g1)]
    for it in range(12):
        An = A1 - float(g1.mid()) * (A1 - A0) / float((g1 - g0).mid())
        An = round(An, 9)
        cn, vn, rn, ok = T.step_in_A(c1, v1, f"{An:.9f}", say)
        if not ok:
            sys.exit("VOID q930: corrector failed in the secant for A_s")
        gn = d2_centre(cn, vn)
        hist.append((An, gn))
        say(f"secant {it}: A {An:.9f}, D''(0) = {gn.str(8, radius=False)}")
        A0, g0, A1, c1, v1, g1 = A1, g1, An, cn, vn, gn
        if abs(g1) < arb("1e-18"):
            break
    else:
        sys.exit("VOID q930: secant for A_s did not reach 1e-18")
    json.dump(
        {"A_s": A1, "d2": float(g1.mid()), "history": [(a, float(g.mid())) for a, g in hist]},
        open(out, "w"),
        indent=1,
    )
    print(f"RESULT q930 As: A_s = {A1:.9f}, D''(0) = {g1.str(5, radius=False)}", flush=True)


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
    m = 30
    cap = Cap(A_str, m, False, grid_A=grid)
    eps_list = [arb("0.01") * arb("1.3") ** k for k in range(17)]
    v = seed62((A_str, v61[:29], v61[29:58], v61[58], v61[59], d), eps_list[0])
    scan = []
    pts = []
    for k, eps in enumerate(eps_list):
        vp = v if k == 0 else predict_eps(cap, v, eps)
        vn, res, ok = fixed_eps_solve(cap, vp, eps)
        if not ok:
            say(f"eps {eps.str(5, radius=False)}: NOT CONVERGED ({getattr(fixed_eps_solve, 'why', '')})")
            scan.append({"eps": float(eps.mid()), "ok": False})
            break
        g = res[m] / eps
        v = vn
        pts.append((eps, [t for t in vn], g))
        scan.append(
            {
                "eps": float(eps.mid()),
                "ok": True,
                "g": float(g.mid()),
                "g_str": g.str(10, radius=False),
                "wc_half": float(vn[m].mid()),
            }
        )
        say(
            f"eps {eps.str(5, radius=False)}: g = {g.str(10, radius=False)}  (w_1 {vn[m].str(6, radius=False)}, residual of the 60 rows below tol)"
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
    if k_sc:
        VERBOSE[0] = os.environ.get("Q930_VERBOSE") == "1"
        k = k_sc[0]
        (e0, g0), (e1, g1) = sc[k], sc[k + 1]
        v0 = pts[k][1]
        v1 = pts[k + 1][1]
        t0_, t1_ = e0 * e0, e1 * e1
        side = 0
        for it in range(40):
            tn = (t0_ * g1 - t1_ * g0) / (g1 - g0)
            en = tn.sqrt()
            vstart = predict_eps(cap, v0 if abs(en - e0) < abs(en - e1) else v1, en)
            vn, res, ok = fixed_eps_solve(cap, vstart, en)
            if not ok:
                out_d["root"] = {
                    "failed": "fixed-eps solve at refinement: " + getattr(fixed_eps_solve, "why", "")
                }
                say(out_d["root"]["failed"])
                break
            gn = res[m] / en
            say(f"root refinement {it}: eps {en.str(10, radius=False)}, g = {gn.str(6, radius=False)}")
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
                ODIR, f"results_q220_gauss_K62_A{float(A0.mid()):.6f}_eps{float(vf[0].mid()):.5f}.json"
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
        for direction, fac, lim in ((-1, arb("0.8"), arb("0.02")), (+1, arb("1.25"), arb("0.6"))):
            vb, Ab, eb, capb = [t for t in vf], A0, vf[0], cap
            while True:
                en = eb * fac
                if (direction < 0 and en < lim) or (direction > 0 and en > lim):
                    break
                vp, Ap = predict_bordered(capb, vb, Ab, en)
                capn, vn, An, resn, okn = bordered_solve(cap, vp, en, Ap)
                if not okn:
                    branch.append({"eps": float(en.mid()), "failed": True})
                    say(f"branch eps {en.str(6, radius=False)}: NOT CONVERGED")
                    break
                verd, cen, off, wmin, ordered = verdict_state(capn, vn, resn)
                branch.append(
                    {
                        "eps": float(en.mid()),
                        "A": An.str(25, radius=False),
                        "A_f": float(An.mid()),
                        "verdict": verd,
                        "centre": float(cen.mid()),
                        "off": float(off.mid()),
                    }
                )
                say(
                    f"branch eps {en.str(6, radius=False)}: A = {An.str(15, radius=False)}, {verd}, centre {cen.str(4, radius=False)}, off {off.str(4, radius=False)}"
                )
                save_state(
                    os.path.join(
                        ODIR, f"results_q220_gauss_K62_A{float(An.mid()):.6f}_eps{float(en.mid()):.5f}.json"
                    ),
                    capn,
                    vn,
                    resn,
                    An,
                    verd,
                    {"grid_A": grid, "eps": float(en.mid())},
                )
                vb, Ab, eb, capb = vn, An, en, capn
        out_d["branch"] = sorted(branch, key=lambda r: r["eps"])
    json.dump(out_d, open(out, "w"), indent=1)
    print(
        f"RESULT q930 scan: {state_file}: points {len(scan)}, sign change {bool(k_sc)}, root {out_d['root']}, branch points {len(out_d['branch'])}",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "As":
        mode_As(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "scan":
        mode_scan(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
