"""G_arb18 (03 Oct 2026): the rp 3 eighteen-ring X + 4G in ball arithmetic, past the K 16 Y split (Amendments 1348/1349).
Uses the Solver (D_solver.py: harmonic-coordinate Newton; KKT functions of D_core2, FUNC from env, here capfix), imported (no
writes at import). Env as D_solver: FUNC=capfix RP=3 PREC FDH TOLR NSCAN FIX_N FIX_B.
Modes:
  seed RM "d1,d2,..." OUTDIR  - from G_work/arb_rp3_K16_rm1.994168_capfix.json (X + Y + 3G, 40 digits) split the Y atom into a pair at
                               90 -+ delta (half the Y mass each) and solve X + 4G at rm = RM by harmonic Newton for each delta (deg).
  held "d1,d2,..." OUTDIR     - held-angle sweep: the new pair's angle held at 90 - delta, rm free, all equations kept (square system);
                               plain Newton with a finite-difference Jacobian in ball arithmetic.
  birth STATE.json RM "mY1,..." OUTDIR - add a Y orbit of per-atom mass mY to a ring without one and solve at RM (03 Oct 01:06).
  reads STATE.json [NFINE]    - margins, D - C at 90 deg and at 89.9 .. 80 deg, curvature at each atom (finite difference), scan of
                               [0, 90] deg on NFINE + 1 points (default 900), for a stored state.
States are written with D_solver.save (full precision strings), loadable by D_solver.load / D_track."""

import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb, arb_mat, ctx
import D_solver as DS
from D_solver import Solver, save, f2

T0 = time.time()
say = lambda *a: print(*a, f"[{time.time() - T0:.0f}s]", flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get(
    "SRC", os.path.join(HERE, "G_work", "arb_rp3_K16_rm1.994168_capfix.json")
)  # 03 Oct 02:28: env SRC (any X + Y + nG ring)


def split_seed(delta_deg):
    d = json.load(open(SRC))
    ctx.prec = DS.PREC
    pi = arb.pi()
    G = [arb(g) for g in d["G"]]
    m = [arb(x) for x in d["m"]]  # m = [mX, mY, mG1, mG2, mG3]
    Gn = G + [pi / 2 - pi * arb(repr(delta_deg)) / 180]
    mn = [m[0]] + m[2:] + [m[1] / 2]
    return Gn + mn + [arb(d["C"])]


def reads(S, v, rm, nfine=900):
    P = S.P
    pi = P.pi
    out = {}
    marg, best = S.scan(v, rm, nfine)
    out["margins"] = [[a, f2(x)] for a, x in marg]
    out["scanmax"] = [best[0], f2(best[1])]
    for dg in ["90", "89.9", "89.75", "89.5", "89", "88", "86", "84", "80"]:
        out[f"DC_{dg}"] = f2(P.kkt(v, rm, pi * arb(dg) / 180))
    out["DC_0"] = f2(P.kkt(v, rm, arb(0)))
    dd = arb("1e-3")
    d2 = []
    for t in P.reps(v[: S.nG]):
        tt = t if f2(t) > 1e-12 else arb(0)
        a = P.kkt(v, rm, tt - dd) if f2(tt) > 0 else P.kkt(v, rm, tt + dd)
        b = P.kkt(v, rm, tt + dd)
        d2.append(f2((a + b - 2 * P.kkt(v, rm, tt)) / (dd * dd)))
    out["d2_fd"] = d2
    return out


def show(tag, S, v, rm, nr, rd):
    G = [round(float((g * 180 / S.P.pi).mid()), 5) for g in v[: S.nG]]
    m = [round(f2(x), 7) for x in v[S.nG : S.ns]]
    say(
        f"{tag}: rm {f2(rm):.10f} residual {nr:.2e}; G deg {G}; masses {m}; C {v[S.ns].str(25, radius=False)}"
    )
    say(
        f"   margins {[f'{a:.3f}:{x:+.3e}' for a, x in rd['margins']]}; scan max {rd['scanmax'][1]:+.3e} at {rd['scanmax'][0]:.3f} deg"
    )
    say(
        "   D - C at "
        + ", ".join(f"{k[3:]}: {rd[k]:+.3e}" for k in rd if k.startswith("DC_"))
        + f"; d2_fd {[f'{x:+.3e}' for x in rd['d2_fd']]}"
    )


def cmd_seed(argv):
    rm_s = argv[0]
    deltas = [float(x) for x in argv[1].split(",")]
    out = argv[2]
    os.makedirs(out, exist_ok=True)
    S = Solver(1, 0, len(json.load(open(SRC))["G"]) + 1)
    rm = arb(rm_s)
    for dl in deltas:
        v0 = split_seed(dl)
        v, nr, ok, hist = S.newton(
            v0, rm, mode="harm", maxit=int(os.environ.get("MAXIT", "25")), verbose=False
        )
        say(
            f"seed delta {dl} deg at rm {rm_s}: converged {ok}, residual history {[f'{h:.1e}' for h in hist]}"
        )
        if not ok:
            continue
        rd = reads(S, v, rm, int(os.environ.get("NFINE", "360")))
        show(f"  state d{dl}", S, v, rm, nr, rd)
        save(S, v, rm_s, os.path.join(out, f"seed_rm{rm_s}_d{dl}.json"), {"residual": nr, **rd})


def cmd_held(argv):
    deltas = [float(x) for x in argv[0].split(",")]
    out = argv[1]
    os.makedirs(out, exist_ok=True)
    S = Solver(1, 0, 4)
    P = S.P
    pi = P.pi
    h = arb(DS.FDH)
    k = S.nG - 1  # held orbit = the last G
    d = json.load(open(SRC))
    rm = arb(repr(float(d["rm"])))
    v = split_seed(deltas[0])
    u = v[:k] + v[k + 1 :] + [rm]  # u = [G1..G3, masses, C, rm]

    def F(u, a):
        vv = u[:k] + [a] + u[k:-1]
        return S.F(vv, u[-1])

    for dl in deltas:
        a = pi / 2 - pi * arb(repr(dl)) / 180
        for it in range(30):
            r = F(u, a)
            nr = max(abs(f2(x)) for x in r)
            n2 = sum(f2(x) ** 2 for x in r)
            if nr < DS.TOLR:
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
            for _ in range(20):
                un = [(u[i] + lam * du[i, 0]).mid() for i in range(n)]
                rn = F(un, a)
                if sum(f2(x) ** 2 for x in rn) < n2:
                    u = un
                    moved = True
                    break
                lam = lam / 2
            say(f"   held {dl} deg it {it}: residual {nr:.2e} -> step lam {f2(lam):.2e}")
            if not moved:
                break
        vv = u[:k] + [a] + u[k:-1]
        rmv = u[-1]
        r = F(u, a)
        nr = max(abs(f2(x)) for x in r)
        rd = reads(S, vv, rmv, int(os.environ.get("NFINE", "360")))
        show(f"held {dl} deg", S, vv, rmv, nr, rd)
        rs = rmv.mid().str(30, radius=False)
        save(S, vv, rs, os.path.join(out, f"held_{dl}.json"), {"residual": nr, "held_deg": dl, **rd})


def cmd_birth(argv):
    """birth STATE.json RM "mY1,mY2,..." OUTDIR: add a Y orbit (per-atom mass mY) to a ring without one, renormalise, harmonic Newton at RM"""
    d, v, rm0 = DS.load(argv[0])
    rm_s = argv[1]
    mys = argv[2].split(",")
    out = argv[3]
    os.makedirs(out, exist_ok=True)
    assert not d["Y"]
    nX = int(d["X"])
    nG = len(d["G"])
    S = Solver(nX, 1, nG)
    rm = arb(rm_s)  # 05:37: X optional
    G = v[:nG]
    m = v[nG : nG + nX + nG]
    C = v[-1]
    for ms in mys:
        mY = arb(ms)
        sc = 1 - 2 * mY
        v0 = list(G) + [x * sc for x in m[:nX]] + [mY] + [x * sc for x in m[nX:]] + [C]
        vv, nr, ok, hist = S.newton(
            v0, rm, mode="harm", maxit=int(os.environ.get("MAXIT", "25")), verbose=False
        )
        say(f"birth seed mY {ms} at rm {rm_s}: converged {ok}, residual history {[f'{h:.1e}' for h in hist]}")
        if not ok:
            continue
        rd = reads(S, vv, rm, int(os.environ.get("NFINE", "360")))
        show(f"  state mY{ms}", S, vv, rm, nr, rd)
        save(S, vv, rm_s, os.path.join(out, f"birth_rm{rm_s}_mY{ms}.json"), {"residual": nr, **rd})


def cmd_reads(argv):
    d, v, rm = DS.load(argv[0])
    S = Solver(d["X"], d["Y"], len(d["G"]))
    rma = arb(str(rm))
    r = S.F(v, rma)
    nr = max(abs(f2(x)) for x in r)
    rd = reads(S, v, rma, int(argv[1]) if len(argv) > 1 else 900)
    show(f"reads {os.path.basename(argv[0])}", S, v, rma, nr, rd)


if __name__ == "__main__":
    {"seed": cmd_seed, "held": cmd_held, "reads": cmd_reads, "birth": cmd_birth}[sys.argv[1]](sys.argv[2:])
