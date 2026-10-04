"""F_ring32_cert: full certificate attempt for D's flank 32-ring (Amendment 1329): polish -> Krawczyk -> curve.
Usage: py F_ring32_cert.py STATE.json OUTNAME
Env: HD, NN, PREC (fine grid for the KKT system), HDC, NNC, PRECC (coarse grid for the curve), A (strip), NT, MT, RHO, DZ,
     NEWTON (max harmonic Newton steps), THREADS.
Phase 1 (not part of the proof): harmonic Newton (the linear step carried through the moment map b_j = sum mult m cos(j t),
j = 2..2(ns-1), plus total mass; the method of D_solver.py, re-implemented here) on the fine-grid system.
Phase 2: Krawczyk on the box v^ +- r. Phase 3: curve certificate on [0, pi/2] with Lam over the box."""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb, arb_mat, ctx
import F_core as FC, F_kkt as FK
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
E = os.environ.get
HD = int(E("HD", "28")); NN = int(E("NN", "658")); PREC = int(E("PREC", "576"))
HDC = int(E("HDC", "16")); NNC = int(E("NNC", "288")); PRECC = int(E("PRECC", "256"))
a = arb(E("A", "1.45")); NT = int(E("NT", "32")); MT = int(E("MT", "36")); RHO = E("RHO", "1"); DZ = float(E("DZ", "0.004"))
NEWTON = int(E("NEWTON", "8")); THREADS = int(E("THREADS", "3"))
st = json.load(open(sys.argv[1])); OUT = sys.argv[2]
FC.setprec(PREC, threads=THREADS)
geo = FC.Geo(st.get("rp", E("RP", "1.0")), st["rm"], st["X"], st["Y"], len(st["G"]))
nG, nO = geo.nG, geo.nO; ns = nG + nO; n = ns + 1
grid = FC.Grid(arb(1)/HD, NN)
(AL, BL), _, _ = FC.strip_consts(geo, a)
eqF = FC.quad_err(AL, BL, geo.rp, geo.rm, 0, a, grid.h, grid.Y)
log(f"state {sys.argv[1]}; fine grid h 1/{HD}, Y {NN/HD}, M {grid.M}, prec {PREC}; a {float(a.mid())}; quadrature bound for D {float(eqF):.3e}")


def orbit_terms(v):
    G, m, C = geo.split(v); k = 0; t = []
    if geo.X: t.append((m[k], 2, "X", None)); k += 1
    if geo.Y: t.append((m[k], 2, "Y", None)); k += 1
    for i, ang in enumerate(G): t.append((m[k + i], 4, "G", ang))
    return t


js = [2*k for k in range(1, ns)]


def moments(v):
    out = []
    for j in js:
        s = arb(0)
        for mm, mult, kind, ang in orbit_terms(v):
            if kind == "X": s += mult*mm
            elif kind == "Y": s += mult*mm*(1 if (j//2) % 2 == 0 else -1)
            else: s += mult*mm*(j*ang).cos()
        out.append(s)
    out.append(sum((mult*mm for mm, mult, kind, ang in orbit_terms(v)), arb(0)))
    return out


def moments_jac(v):
    Mj = arb_mat(ns, ns); G, m, C = geo.split(v); off = int(geo.X) + int(geo.Y)
    for r, j in enumerate(js + [0]):
        for i in range(nG): Mj[r, i] = (-4*m[off + i]*j*(j*G[i]).sin()) if j else arb(0)
        for o in range(nO):
            kind = orbit_terms(v)[o][2]
            if j == 0: Mj[r, nG + o] = arb(2) if kind in "XY" else arb(4)
            elif kind == "X": Mj[r, nG + o] = arb(2)
            elif kind == "Y": Mj[r, nG + o] = arb(2*(1 if (j//2) % 2 == 0 else -1))
            else: Mj[r, nG + o] = 4*(j*G[o - off]).cos()
    return Mj


def moment_solve(target, v0):
    v = list(v0); tol = 2.0**(-PREC + 40)
    for it in range(60):
        mo = moments(v); r = [x - y for x, y in zip(mo, target)]; nr = max(abs(float(x.mid())) for x in r)
        if nr < tol: return v, nr
        d = moments_jac(v).solve(arb_mat(ns, 1, [-(x.mid()) for x in r]))
        v = [arb((v[i] + d[i, 0]).mid()) for i in range(ns)] + v[ns:]
    return v, nr


def harm_newton(v, maxit):
    best = None
    for it in range(maxit):
        F, J, _ = FK.system(geo, grid, v, a, jac=True)
        res = max(float(abs(x).upper()) for x in F)
        log(f"  harmonic Newton {it}: |F(v)| <= {res:.3e}")
        if best is None or res < best[0]: best = (res, v, F, J)
        if res < 1e3*float(eqF) or (it > 0 and res > 0.5*prev): break
        prev = res
        Jm = arb_mat(n, n, [arb(J[i, j].mid()) for i in range(n) for j in range(n)])
        dv = Jm.solve(arb_mat(n, 1, [-arb(x.mid()) for x in F]))
        Mj = moments_jac(v); mo = moments(v)
        db = Mj*arb_mat(ns, 1, [dv[i, 0] for i in range(ns)])
        target = [mo[i] + db[i, 0] for i in range(ns - 1)] + [arb(1)]
        guess = [arb((v[i] + dv[i, 0]).mid()) for i in range(n)]
        vn, mr = moment_solve(target, guess)
        v = [arb(x.mid()) for x in vn[:ns]] + [arb((v[ns] + dv[ns, 0]).mid())]
    return best


v = [arb(arb(x).mid()) for x in st["G"]] + [arb(arb(x).mid()) for x in st["m"]] + [arb(arb(st["C"]).mid())]
res, v, F0, J0 = harm_newton(v, NEWTON)
log(f"polished: |F(v^)| <= {res:.3e} (rigorous, exact F)")
dig = int(PREC*0.30103) + 5
json.dump({"rm": st["rm"], "rp": st.get("rp", "1.0"), "X": st["X"], "Y": st["Y"], "func": "cap",
           "G": [x.mid().str(dig, radius=False) for x in v[:nG]], "m": [x.mid().str(dig, radius=False) for x in v[nG:ns]],
           "C": v[ns].mid().str(dig, radius=False), "F_upper": res, "grid": f"h 1/{HD} N {NN} prec {PREC} a {E('A', '1.45')}"},
          open(f"F_work/{OUT}_polished.json", "w"), indent=1)
# ---- Krawczyk
Jm = arb_mat(n, n, [arb(J0[i, j].mid()) for i in range(n) for j in range(n)]); Yi = Jm.inv()
YF = [float(abs((Yi*arb_mat(n, 1, F0))[i, 0]).upper()) for i in range(n)]
log("|Y F(v^)| componentwise: " + " ".join(f"{x:.2e}" for x in YF))
ok = False; fac = 10.0
for attempt in range(3):
    r = [max(fac*x, fac*max(YF)*1e-8) for x in YF]
    okk, det = FK.krawczyk(geo, grid, v, r, a, F0=F0)
    log(f"Krawczyk attempt {attempt} (r = {fac:g}|YF|, max r {max(r):.2e}): ok {okk}; ||I - YJ(B)||_inf {det['normM']:.3e}; "
        f"|K - v^|/r max {max(u/rr for u, rr in det['rows']):.3f}")
    if okk: ok = True; break
    fac *= 10
if not ok:
    log("KRAWCZYK FAILED"); sys.exit(1)
G, m, C = geo.split(v)
log("exact KKT point in the box: G deg " + ", ".join(f"{float(x.mid())*180/math.pi:.10f}" for x in G) + f"; radii max {max(r):.2e}; "
    f"smallest mass lower bound {min(float(x.mid()) - rr for x, rr in zip(m, r[nG:ns])):.6e}; C {C.mid().str(30, radius=False)}")
json.dump({"box_r": r, "krawczyk_ok": ok, "normM": det["normM"]}, open(f"F_work/{OUT}_krawczyk.json", "w"), indent=1)
# ---- curve
FC.setprec(PRECC, threads=THREADS)
gridc = FC.Grid(arb(1)/HDC, NNC)
B = [arb(vh.mid()) + arb(ri)*arb(0, 1) for vh, ri in zip(v, r)]
Gb, mb, Cb = geo.split(B); atomsB = geo.atoms(Gb, mb)
mdev = abs(FK.sum_w(geo, mb) - 1).upper()
(ALb, BLb), _, _ = FC.strip_consts(geo, a, msum_dev=mdev)
LamB = FC.lam_grid(gridc, geo, atomsB)
log(f"curve: coarse grid h 1/{HDC}, Y {NNC/HDC}, M {gridc.M}, prec {PRECC}; quadrature bound "
    f"{float(FC.quad_err(ALb, BLb, geo.rp, geo.rm, 0, a, gridc.h, gridc.Y)):.3e}; Lam over the box done")
ats = []; ars = []; atsb = []
if geo.X: ats.append(0.0); ars.append(0.0); atsb.append(arb(0))
if geo.Y: ats.append(math.pi/2); ars.append(0.0); atsb.append(arb.pi()/2)
for g in range(nG): ats.append(float(v[g].mid())); ars.append(r[g]); atsb.append(arb(v[g].mid()))
okc, rep = FK.curve_certify(geo, gridc, LamB, Cb, a, ats, ars, [DZ]*len(ats), 0.0, math.pi/2, NT, MT, RHO, AL_BL=(ALb, BLb), log=log,
                            t_hi_exact=arb.pi()/2, atom_ts_arb=atsb)
log(f"curve certificate ok {okc} (on [0, pi/2] closed at the exact pi/2): max UB of D - C off the zones {rep['max_free']}; max UB of "
    f"(D - C)'' in the zones {rep['max_d2']}; intervals {rep['nfree']} free + {rep['natom']} atom zones; pieces evaluated {rep['pieces']}; "
    f"zone half-width {DZ} + atom radius; failures {rep['worst']}; certified violations {rep['violations']} {rep['viol']}")
# the vertex value, for the record
WB = [FC.weights(gridc, geo, arb(0), 0)]
c0 = FC.contract_multi(gridc, LamB, WB, 0)[0][0]
val = FC.half_norm2_series(geo, arb(0), 0)[0] - c0 - Cb
val = FK.ball(val, FC.quad_err(ALb, BLb, geo.rp, geo.rm, 0, a, gridc.h, gridc.Y))
log(f"D - C at the major vertex t = 0, enclosure valid for every v in the box (so for the exact KKT point): "
    f"[{val.lower().str(12, radius=False)}, {val.upper().str(12, radius=False)}]")
WB2 = [FC.weights(gridc, geo, arb.pi()/2, 0)]
c2 = FC.contract_multi(gridc, LamB, WB2, 0)[0][0]
val2 = FC.half_norm2_series(geo, arb.pi()/2, 0)[0] - c2 - Cb
val2 = FK.ball(val2, FC.quad_err(ALb, BLb, geo.rp, geo.rm, 0, a, gridc.h, gridc.Y))
log(f"D - C at the minor vertex t = pi/2, enclosure valid for every v in the box: "
    f"[{val2.lower().str(12, radius=False)}, {val2.upper().str(12, radius=False)}]")
json.dump({"box_r": r, "krawczyk_ok": ok, "normM": det["normM"], "curve_ok": okc, "max_free": rep["max_free"], "max_d2": rep["max_d2"],
           "vertex": [float(val.lower()), float(val.upper())]}, open(f"F_work/{OUT}_cert.json", "w"), indent=1)
log("done")
