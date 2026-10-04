"""H_arb (03 Oct 2026): seeds and reads for the mixed curve (LAM 0.02, RP 1.2) in ball arithmetic. Adapted from G_arb18.py
(a helper script; copied as text, not imported): split / birth seeds generalised to any vertex or interior place, and reads at BOTH vertices.
Uses the Solver (D_solver.py: harmonic-coordinate Newton; KKT functions of D_core2, FUNC from env, here capfix), imported (no
writes at import). Env as D_solver: FUNC=capfix RP=1.2 LAM=0.02 PREC FDH TOLR MAXIT NFINE FIX_N.
Modes:
  seed SRC.json RM "RECIPE1;RECIPE2;..." OUTDIR   - build a new ring from SRC by a recipe and solve it at RM by harmonic Newton.
        Recipes: splitY:<delta_deg>      Y atom -> G pair at 90 -+ delta (half the Y per-atom mass each atom of the G orbit)
                 splitX:<delta_deg>      X atom -> G pair at 0 +- delta
                 splitG<k>:<delta_deg>   G orbit k (0-based, in the file's order) -> two G orbits at a_k -+ delta, half mass each
                 birthY:<per-atom mass>  add a Y orbit, masses rescaled to total 1
                 birthX:<per-atom mass>  add an X orbit
                 birthG@<deg>:<mass>     add a G orbit at the given angle
                 merge:<k1>,<k2>         (not used yet)
  reads STATE.json [NFINE]   - the full read set of H_reads(): margins, gap maxima, fine reads 0-5 and 85-90 deg, D - C at 0 and 90 deg,
                               curvature at every atom; printed and written to STATE_reads.json.
States are written with D_solver.save (full precision strings), loadable by D_solver.load / D_track / H_track.
"""

import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb, arb_mat, ctx
import D_solver as DS
from D_solver import Solver, save, f2

T0 = time.time()
say = lambda *a: print(*a, f"[{time.time() - T0:.0f}s]", flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))


def atoms_deg(S, v):
    """representative atom angles in [0, 90] deg, with a kind label, sorted."""
    P = S.P
    out = []
    if P.X:
        out.append((0.0, "X", arb(0)))
    if P.Y:
        out.append((90.0, "Y", P.pi / 2))
    for i, a in enumerate(v[: S.nG]):
        out.append((float((a * 180 / P.pi).mid()), f"G{i}", a))
    return sorted(out, key=lambda x: x[0])


def H_reads(S, v, rm, nfine=360, gaprefine=True):
    """reads at a stored state (coordinator's rule): margins midway, D - C on [0, 90] (nfine + 1 points), maxima of D - C in every gap
    between neighbouring atoms (golden-section refined), fine reads on 0-5 and 85-90 deg at 0.05 deg, D - C at 0 and 90 deg, curvature
    of D - C at every atom (finite difference 1e-3 rad, one-sided-symmetric at a vertex)."""
    P = S.P
    pi = P.pi
    out = {}
    deg = lambda x: arb(repr(x)) * pi / 180
    marg, best = P.scan(v, rm, nfine)
    out["margins"] = [[a, f2(x)] for a, x in marg]
    out["scanmax"] = [best[0], f2(best[1])]
    ts0 = [round(0.05 * i, 2) for i in range(101)]
    ts9 = [round(85 + 0.05 * i, 2) for i in range(101)]
    vals = [f2(x) for x in P.kkt_many(v, rm, [deg(t) for t in ts0 + ts9])]
    f0, f9 = vals[:101], vals[101:]
    out["fine0_5"] = f0
    out["fine85_90"] = f9
    out["DC_0"] = f0[0]
    out["DC_90"] = f9[-1]
    at = atoms_deg(S, v)
    adeg = [a[0] for a in at]
    near = (
        lambda t: min(abs(t - a) for a in adeg) < 0.02
    )  # skip points on an atom (D - C = 0 there to the residual)
    m0 = max(((t, x) for t, x in zip(ts0, f0) if not near(t)), key=lambda p: p[1], default=(None, -1.0))
    m9 = max(((t, x) for t, x in zip(ts9, f9) if not near(t)), key=lambda p: p[1], default=(None, -1.0))
    out["fine0_5_max"] = list(m0)
    out["fine85_90_max"] = list(m9)
    dd = arb("1e-3")
    d2 = []
    for adg, kind, t in at:
        if kind == "X":
            x = 2 * (P.kkt(v, rm, dd) - P.kkt(v, rm, arb(0))) / (dd * dd)
        elif kind == "Y":
            x = 2 * (P.kkt(v, rm, pi / 2 - dd) - P.kkt(v, rm, pi / 2)) / (dd * dd)
        else:
            x = (P.kkt(v, rm, t - dd) + P.kkt(v, rm, t + dd) - 2 * P.kkt(v, rm, t)) / (dd * dd)
        d2.append([kind, adg, f2(x)])
    out["d2_atoms"] = d2
    # gap maxima: gaps between neighbouring atoms in [0, 90] (with the mirror images -a1 and 180 - a_last when no vertex atom)
    full = list(adeg)
    if not P.X:
        full = [-adeg[0]] + full
    if not P.Y:
        full = full + [180 - adeg[-1]]
    gaps = []
    if gaprefine:
        for lo, hi in zip(full[:-1], full[1:]):
            a, b = max(lo, 0.0) if lo < 0 else lo, min(
                hi, 90.0
            )  # search the half of a vertex gap inside [0, 90]
            if b - a < 1e-6:
                continue
            # coarse: 39 interior points; the largest INTERIOR local maximum (a bump) is refined by golden section; if the gap has no
            # interior local maximum, the entry is the largest coarse value with flag "edge" (the maximum then sits at an atom)
            n = 40
            tt = [a + (b - a) * k / n for k in range(1, n)]
            vv = [f2(x) for x in P.kkt_many(v, rm, [deg(t) for t in tt])]
            vb = [f2(P.kkt(v, rm, deg(a))), f2(P.kkt(v, rm, deg(b)))]
            ext = [vb[0]] + vv + [vb[1]]
            loc = [i for i in range(1, len(ext) - 1) if ext[i] >= ext[i - 1] and ext[i] >= ext[i + 1]]
            vertex_end = (b == 90.0 and not P.Y) or (
                a == 0.0 and not P.X
            )  # a vertex end without atom is a legitimate maximum place
            if b == 90.0 and not P.Y and ext[-1] >= ext[-2]:
                loc.append(len(ext) - 1)
            if a == 0.0 and not P.X and ext[0] >= ext[1]:
                loc.append(0)
            if not loc:
                k = max(range(len(tt)), key=lambda i: vv[i])
                gaps.append([round(lo, 5), round(hi, 5), tt[k], vv[k], "edge"])
                continue
            i = max(loc, key=lambda j: ext[j])
            tx = [a] + tt + [b]
            l, r = tx[max(i - 1, 0)], tx[min(i + 1, len(tx) - 1)]
            gr = 0.6180339887498949
            c1 = r - gr * (r - l)
            c2 = l + gr * (r - l)
            v1 = f2(P.kkt(v, rm, deg(c1)))
            v2 = f2(P.kkt(v, rm, deg(c2)))
            for _ in range(20):
                if v1 > v2:
                    r, c2, v2 = c2, c1, v1
                    c1 = r - gr * (r - l)
                    v1 = f2(P.kkt(v, rm, deg(c1)))
                else:
                    l, c1, v1 = c1, c2, v2
                    c2 = l + gr * (r - l)
                    v2 = f2(P.kkt(v, rm, deg(c2)))
            tm, vm = (c1, v1) if v1 > v2 else (c2, v2)
            if ext[i] > vm:
                tm, vm = tx[i], ext[i]
            gaps.append([round(lo, 5), round(hi, 5), tm, vm, "bump"])
    out["gapmax"] = gaps
    return out


def valid_check(S, v, rd, tolv):
    """full KKT test from the reads: every value read below tolv, every atom curvature negative, every mass positive. Returns list of
    failure reasons (empty = valid)."""
    bad = []
    if rd["scanmax"][1] > tolv:
        bad.append(f"scan {rd['scanmax'][1]:+.2e}@{rd['scanmax'][0]:.3f}")
    if rd["fine0_5_max"][1] > tolv:
        bad.append(f"fine0 {rd['fine0_5_max'][1]:+.2e}@{rd['fine0_5_max'][0]}")
    if rd["fine85_90_max"][1] > tolv:
        bad.append(f"fine90 {rd['fine85_90_max'][1]:+.2e}@{rd['fine85_90_max'][0]}")
    for a, x in rd["margins"]:
        if x > tolv:
            bad.append(f"margin {x:+.2e}@{a:.3f}")
    for g in rd["gapmax"]:
        if g[3] > tolv:
            bad.append(f"gap({g[0]:.2f},{g[1]:.2f}) {g[3]:+.2e}@{g[2]:.4f}")
    for kind, adg, x in rd["d2_atoms"]:
        if x >= 0:
            bad.append(f"d2 {kind}@{adg:.3f} {x:+.2e}")
    if min(f2(x) for x in v[S.nG : S.ns]) <= 0:
        bad.append("negative mass")
    return bad


def short(rd):
    return (
        f"margins {[f'{a:.2f}:{x:+.2e}' for a, x in rd['margins']]}; gapmax {[f'{g[2]:.3f}:{g[3]:+.2e}' + ('' if g[4] == 'bump' else 'e') for g in rd['gapmax']]}; "
        f"scan {rd['scanmax'][1]:+.2e}@{rd['scanmax'][0]:.2f}; DC0 {rd['DC_0']:+.2e} DC90 {rd['DC_90']:+.2e}; fine0max {rd['fine0_5_max'][1]:+.2e}@{rd['fine0_5_max'][0]}; "
        f"fine90max {rd['fine85_90_max'][1]:+.2e}@{rd['fine85_90_max'][0]}; d2 {[f'{k}:{x:+.3e}' for k, a, x in rd['d2_atoms']]}"
    )


def show(tag, S, v, rm, nr, rd):
    G = [round(float((g * 180 / S.P.pi).mid()), 5) for g in v[: S.nG]]
    m = [round(f2(x), 7) for x in v[S.nG : S.ns]]
    say(
        f"{tag}: rm {f2(rm):.12f} (eps {1.2 - f2(rm):.12f}) X{int(S.P.X)} Y{int(S.P.Y)} nG {S.nG} residual {nr:.2e}; G deg {G}; masses {m}; C {v[S.ns].str(30, radius=False)}"
    )
    say("   " + short(rd))


def build(src, recipe):
    """new (X, Y, G list, m list, C) from a source json and one recipe."""
    d = json.load(open(src))
    ctx.prec = DS.PREC
    pi = arb.pi()
    X, Y = d["X"], d["Y"]
    G = [arb(g) for g in d["G"]]
    m = [arb(x) for x in d["m"]]
    C = arb(d["C"])
    k = 0
    mX = mY = None
    if X:
        mX = m[k]
        k += 1
    if Y:
        mY = m[k]
        k += 1
    mG = m[k:]
    kind, arg = recipe.split(":", 1)
    if kind == "splitY":
        assert Y
        dl = pi * arb(arg) / 180
        G = G + [pi / 2 - dl]
        mG = mG + [mY / 2]
        Y = 0
        mY = None
    elif kind == "splitX":
        assert X
        dl = pi * arb(arg) / 180
        G = [dl] + G
        mG = [mX / 2] + mG
        X = 0
        mX = None
    elif kind.startswith("splitG"):
        j = int(kind[6:])
        dl = pi * arb(arg) / 180
        a = G[j]
        mm = mG[j]
        G = G[:j] + [a - dl, a + dl] + G[j + 1 :]
        mG = mG[:j] + [mm / 2, mm / 2] + mG[j + 1 :]
    elif kind in (
        "mergeGtoY",
        "mergeGtoX",
    ):  # 06:2x: a G orbit sitting at a vertex -> vertex orbit (2 atoms, 2x mass)
        j = int(arg)
        mm = mG[j]
        G = G[:j] + G[j + 1 :]
        mG = mG[:j] + mG[j + 1 :]
        if kind == "mergeGtoY":
            assert not Y
            Y = 1
            mY = 2 * mm
        else:
            assert not X
            X = 1
            mX = 2 * mm
    elif kind == "none":
        pass
    elif kind in ("birthY", "birthX") or kind.startswith("birthG"):
        mb = arb(arg)
        tot_new = 2 * mb if kind in ("birthY", "birthX") else 4 * mb
        sc = 1 - tot_new
        mX = mX * sc if mX is not None else None
        mY = mY * sc if mY is not None else None
        mG = [x * sc for x in mG]
        if kind == "birthY":
            assert not Y
            Y = 1
            mY = mb
        elif kind == "birthX":
            assert not X
            X = 1
            mX = mb
        else:
            a = pi * arb(kind.split("@")[1]) / 180
            G = G + [a]
            mG = mG + [mb]
    else:
        raise ValueError(recipe)
    mm = ([mX] if X else []) + ([mY] if Y else []) + mG
    return X, Y, G, mm, C


def cmd_seed(argv):
    ctx.prec = (
        DS.PREC
    )  # 06:2x FIX: rm was made at the default 53 bits before (a wide ball: noise floor 1e-12)
    src, rm_s, recipes, out = argv[0], argv[1], argv[2].split(";"), argv[3]
    os.makedirs(out, exist_ok=True)
    rm = arb(rm_s)
    for rc in recipes:
        X, Y, G, m, C = build(src, rc)
        S = Solver(X, Y, len(G))
        v0 = list(G) + list(m) + [C]
        v, nr, ok, hist = S.newton(
            v0,
            rm,
            mode=os.environ.get("NMODE", "harm"),
            maxit=int(os.environ.get("MAXIT", "25")),
            verbose=True,
        )
        say(
            f"seed {rc} from {os.path.basename(src)} at rm {rm_s}: converged {ok}, residual history {[f'{h:.1e}' for h in hist]}"
        )
        tag = rc.replace(":", "_").replace("@", "a").replace(",", "_")
        if not ok:
            save(S, v, rm_s, os.path.join(out, f"seedFAIL_rm{rm_s}_{tag}.json"), {"residual": nr})
            continue
        rd = H_reads(S, v, rm, int(os.environ.get("NFINE", "360")))
        show(f"  state {rc}", S, v, rm, nr, rd)
        bad = valid_check(S, v, rd, float(os.environ.get("TOLV", "1e-31")))
        say(
            f"   KKT test (TOLV {os.environ.get('TOLV', '1e-31')}): {'VALID' if not bad else 'INVALID ' + str(bad)}"
        )
        save(
            S,
            v,
            rm_s,
            os.path.join(out, f"seed_rm{rm_s}_{tag}.json"),
            {"residual": nr, "reads": rd, "bad": bad},
        )


def cmd_reads(argv):
    d, v, rm = DS.load(argv[0])
    S = Solver(d["X"], d["Y"], len(d["G"]))
    rma = arb(str(rm))
    r = S.F(v, rma)
    nr = max(abs(f2(x)) for x in r)
    rd = H_reads(S, v, rma, int(argv[1]) if len(argv) > 1 else 360)
    show(f"reads {os.path.basename(argv[0])}", S, v, rma, nr, rd)
    bad = valid_check(S, v, rd, float(os.environ.get("TOLV", "1e-31")))
    say(f"   KKT test: {'VALID' if not bad else 'INVALID ' + str(bad)}")
    json.dump(
        {"rm": str(rm), "residual": nr, "reads": rd, "bad": bad},
        open(argv[0].replace(".json", "_reads.json"), "w"),
    )


def cmd_solve(argv):
    """solve STATE.json RM OUTFILE: harmonic Newton for the same ring at RM (tangent predictor if RM differs), then reads."""
    d, v, rm0 = DS.load(argv[0])
    S = Solver(d["X"], d["Y"], len(d["G"]))
    rm = arb(argv[1])
    if abs(float(argv[1]) - float(rm0)) > 1e-15:
        J, Fr = S.jac(v, arb(rm0), with_rm=True)
        t = J.solve(-Fr)
        dr = rm - arb(rm0)
        v = [(v[i] + dr * t[i, 0]).mid() for i in range(S.n)]
    v, nr, ok, hist = S.newton(
        v, rm, mode=os.environ.get("NMODE", "harm"), maxit=int(os.environ.get("MAXIT", "25")), verbose=True
    )
    say(
        f"solve {os.path.basename(argv[0])} at rm {argv[1]}: converged {ok}, residual history {[f'{h:.1e}' for h in hist]}"
    )
    rd = H_reads(S, v, rm, int(os.environ.get("NFINE", "360")))
    show("  state", S, v, rm, nr, rd)
    bad = valid_check(S, v, rd, float(os.environ.get("TOLV", "1e-31")))
    say(f"   KKT test: {'VALID' if not bad else 'INVALID ' + str(bad)}")
    save(S, v, argv[1], argv[2], {"residual": nr, "converged": ok, "reads": rd, "bad": bad})


def cmd_held(argv):
    """held SRC.json RECIPE IDX "a1,a2,..." OUTDIR [RM0]: build a ring from SRC by RECIPE ("none" = as is), hold the angle of G orbit
    IDX at a1, a2, ... deg in turn and solve all KKT equations with rm free (square system; plain Newton, finite-difference Jacobian in
    ball arithmetic, line search). Adapted from G_arb18.cmd_held. A pitchfork (split) or fold is regular in this
    parametrisation, so this walks a new branch out of a degenerate point; each solution is read and saved with its rm.
    """
    src, recipe, k, angs, out = argv[0], argv[1], int(argv[2]), argv[3].split(","), argv[4]
    os.makedirs(out, exist_ok=True)
    d = json.load(open(src))
    ctx.prec = DS.PREC
    if recipe == "none":
        X, Y, G, m, C = d["X"], d["Y"], [arb(g) for g in d["G"]], [arb(x) for x in d["m"]], arb(d["C"])
    else:
        X, Y, G, m, C = build(src, recipe)
    S = Solver(X, Y, len(G))
    P = S.P
    pi = P.pi
    h = arb(DS.FDH)
    rm = arb(argv[5] if len(argv) > 5 else d["rm"])
    v = list(G) + list(m) + [C]
    u = v[:k] + v[k + 1 :] + [rm]

    def F(u, a):
        vv = u[:k] + [a] + u[k:-1]
        return S.F(vv, u[-1])

    for ad in angs:
        a = pi * arb(ad) / 180
        ok = False
        for it in range(int(os.environ.get("MAXIT", "30"))):
            r = F(u, a)
            nr = max(abs(f2(x)) for x in r)
            n2 = sum(f2(x) ** 2 for x in r)
            if nr < DS.TOLR:
                ok = True
                break
            n = len(u)
            J = arb_mat(n, n)
            for j in range(n):
                up = list(u)
                up[j] = up[j] + h
                um = list(u)
                um[j] = um[j] - h
                fp = F(up, a)
                fm = F(um, a)
                for i in range(n):
                    J[i, j] = ((fp[i] - fm[i]) / (2 * h)).mid()
            du = J.solve(arb_mat(n, 1, [-(x.mid()) for x in r]))
            lam = arb(1)
            moved = False
            for _ in range(30):
                un = [(u[i] + lam * du[i, 0]).mid() for i in range(n)]
                rn = F(un, a)
                if sum(f2(x) ** 2 for x in rn) < n2:
                    u = un
                    moved = True
                    break
                lam = lam / 2
            say(
                f"   held {ad} deg it {it}: residual {nr:.2e} -> step lam {f2(lam):.2e}, rm {u[-1].mid().str(20, radius=False)}"
            )
            if not moved:
                break
        vv = u[:k] + [a] + u[k:-1]
        rmv = u[-1]
        r = F(u, a)
        nr = max(abs(f2(x)) for x in r)
        say(
            f"held angle {ad} deg: converged {nr < DS.TOLR} (residual {nr:.2e}); rm {rmv.mid().str(25, radius=False)} (eps {1.2 - f2(rmv):.12f})"
        )
        rs = rmv.mid().str(30, radius=False)
        if nr >= DS.TOLR:
            save(
                S,
                vv,
                rs,
                os.path.join(out, f"heldFAIL_{recipe.replace(':', '_')}_{k}_{ad}.json"),
                {"residual": nr, "held_deg": ad},
            )
            break
        rd = H_reads(S, vv, rmv, int(os.environ.get("NFINE", "360")))
        show(f"  held {ad}", S, vv, rmv, nr, rd)
        bad = valid_check(S, vv, rd, DS.TOLV)
        say(f"   KKT test (TOLV {DS.TOLV:.0e}): {'VALID' if not bad else 'INVALID ' + str(bad)}")
        save(
            S,
            vv,
            rs,
            os.path.join(out, f"held_{recipe.replace(':', '_')}_{k}_{ad}.json"),
            {"residual": nr, "held_deg": ad, "reads": rd, "bad": bad},
        )


if __name__ == "__main__":
    {"seed": cmd_seed, "reads": cmd_reads, "solve": cmd_solve, "held": cmd_held}[sys.argv[1]](sys.argv[2:])
