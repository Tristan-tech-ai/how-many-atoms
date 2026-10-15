"""q924b (7 Oct 2026): text copy of q924 with ONE change registered in the amendment that registers it: for odd K the item K4 requires the positions to rise and dwc/dA > 0 (the innermost mass of a centre family is the centre mass); dw_1/dA is reported without a verdict for odd K (the bar dw_1/dA > 0 of q924 was an extension of the even-K picture and contradicts the measured centre families); the verdict line reads VERDICT q924b. Header of q924: (7 Oct 2026; K1 changed at 00:33 from forward differences with h = 1e-45 to central differences with h = 1e-20, because the 200-bit noise of about 1e-59 divided by h = 1e-45 gives an error of 1e-14 and the bar 1e-30 could not be met; recorded in the amendment that registers the script): the implicit-function flow, the reduced Hessian and its total positivity at one saved capacity state, even or odd K, in 200-bit ball arithmetic (python-flint, q920); the criteria are those of q901c (amendments 1677, 1680, 1681) adapted to the flint instrument, registered in an amendment before any odd state was analysed.
The state is read from a results_q220_gauss_K<K>_A<A>.json file and polished by Newton to residual below 1e-50. Unknowns (u_1..u_m, w_1..w_m, we[, wc]); rows value (0..m-1), stationarity (m..2m-1), mass (2m), centre (2m+1, odd K).
K1 Jacobian: central differences with h = 1e-20 in the columns 0, m/2, m-1, m, m + m/2, 2m-1, 2m (and 2m+1 for odd K) against the analytic J: max error <= 1e-30; control J[2,0] x 1.01 must give an error above 1e-30.
K2 dF/dA: central difference with h = 1e-30 in A on the frozen grid against the analytic b: max |b_an - b_fd| <= 1e-25 max |b_fd|; control b[0] x 1.1 must fail.
K3 flow: du = -J^-1 b, linear residual <= 1e-40 max(1, |b|); with delta in (1e-12, 1e-14, 1e-16) on the frozen grid ratio = ||F(A + delta, v + delta du)|| / ||F(A + delta, v)||: ratio(1e-16) <= 0.05, ratio(1e-14)/ratio(1e-12) <= 0.3, ratio(1e-16)/ratio(1e-14) <= 0.3 and ||F(A + 1e-16, v)|| >= 1e-25; controls D1 (du from 1.1 b) and D2 (stationarity diagonal J[m + j, j] x 1.01) must each fail one of the three ratio conditions.
K4 signs: du_j/dA > 0 for every position; dw_1/dA > 0 (innermost pair mass); for odd K also dwc/dA > 0 (the centre mass, item K4c); control -du must fail the position item.
K5 structure: S = A_s - B_s Dm^-1 C_s (stationarity rows, position columns; the masses eliminated through the value rows, the mass row and the centre row); C-sign (S_pq has the sign -(-1)^(p+q)), C-inv (S^-1 negative entrywise), TP of T = -DSD (all minors of orders 1, 2 and 20000 sampled of order 3), TPh (5000 sampled each of orders 4 and 5), in ball arithmetic at 400 bits, undecided minors recomputed at 1600 bits.
K6 positive control: S_pc = -D K D with K_pq = exp(-0.1 (p-q)^2) must pass C-sign, C-inv, TP and TPh. K7 negative controls: sign flip of T[3][7], |S|, identity must fail.
Verdict (last line): 'VERDICT q924 <file>: PASS' iff K1-K5 hold (K4 includes K4c for odd K) and the controls K6, K7 and those of K1-K4 behave; 'FAIL' otherwise with the items; 'VOID' (exit 2) if a control misbehaves or the state does not polish.
Usage: py q924_cap_structure.py STATE.json   (from temporal_within/)"""

import sys, json, os, copy, time
import numpy as np
from flint import arb, arb_mat, ctx
import q920_cap_flint as F
import q921_cap_tools as T
import q925_matrix_gates as MG

t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
MG.say = say
arbf = lambda t, n=5: t.str(n, radius=False)


def main():
    path = sys.argv[1]
    A_str, u, w, we, wc, d = F.load_state(path)
    m = len(u)
    odd = wc is not None
    K = 2 * m + 2 + (1 if odd else 0)
    cap = F.Cap(A_str, m, odd, grid_A=d.get("grid_A"))
    nn = cap.nn
    v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
    ctx.prec = 200
    v, res, ok = T.newton_full(cap, v, lambda *a: None, tol=arb("1e-50"), maxit=10)
    if not ok:
        print(f"VOID q924b {path}: the state does not polish (residual {F.maxabs(res)})", flush=True)
        sys.exit(2)
    say(f"state {path}: K {K}, m {m}, unknowns {nn}, A {A_str}, polished residual {arbf(F.maxabs(res), 3)}")
    res, J, b = cap.res_jac(v, True, True)
    items = {}
    ctrl = {}
    # K1
    cols = [0, m // 2, m - 1, m, m + m // 2, 2 * m - 1, 2 * m] + ([2 * m + 1] if odd else [])
    h = arb("1e-20")
    err1 = arb(0)
    fd0 = None
    for k in cols:
        v2 = list(v)
        v2[k] = v2[k] + h
        r2, _, _ = cap.res_jac(v2, False, False)
        v3 = list(v)
        v3[k] = v3[k] - h
        r3, _, _ = cap.res_jac(v3, False, False)
        fd = [(r2[j] - r3[j]) / (2 * h) for j in range(nn)]
        if k == 0:
            fd0 = fd
        e = max(abs(((fd[j] - J[j, k]) / max(arb(1), abs(J[j, k]))).mid()) for j in range(nn))
        err1 = max(err1, e)
    items["K1"] = bool(err1 <= arb("1e-30"))
    e1c = abs(((fd0[2] - arb("1.01") * J[2, 0]) / max(arb(1), abs(arb("1.01") * J[2, 0]))).mid())
    ctrl["K1c"] = bool(e1c > arb("1e-30"))
    say(
        f"K1: columns {len(cols)}, max error {arbf(err1, 3)} -> {items['K1']}; control J[2,0] x 1.01 gives {arbf(e1c, 3)} -> fails as it must {ctrl['K1c']}"
    )
    # K2
    hA = arb("1e-30")
    capP = copy.copy(cap)
    capP.A = cap.A + hA
    capM = copy.copy(cap)
    capM.A = cap.A - hA
    rp, _, _ = capP.res_jac(v, False, False)
    rm, _, _ = capM.res_jac(v, False, False)
    bfd = [((rp[j] - rm[j]) / (2 * hA)).mid() for j in range(nn)]
    bmax = max(abs(t) for t in bfd)
    err2 = max(abs((b[j] - bfd[j]).mid()) for j in range(nn))
    items["K2"] = bool(err2 <= arb("1e-25") * bmax)
    bc = list(b)
    bc[0] = bc[0] * arb("1.1")
    e2c = max(abs((bc[j] - bfd[j]).mid()) for j in range(nn))
    ctrl["K2c"] = bool(e2c > arb("1e-25") * bmax)
    say(
        f"K2: max |b_an - b_fd| {arbf(err2, 3)} against {arbf(arb('1e-25') * bmax, 3)} -> {items['K2']}; control b[0] x 1.1 gives {arbf(e2c, 3)} -> fails as it must {ctrl['K2c']}"
    )
    # K3
    du = T.solve_lin(J, [-t for t in b])
    lin = max(abs((sum((J[i, k] * du[k] for k in range(nn)), arb(0)) + b[i]).mid()) for i in range(nn))
    lin_ok = bool(lin <= arb("1e-40") * max(arb(1), max(abs(t.mid()) for t in b)))
    DEL = [arb("1e-12"), arb("1e-14"), arb("1e-16")]
    caps = []
    for dl in DEL:
        c_ = copy.copy(cap)
        c_.A = cap.A + dl
        caps.append(c_)
    frozen = [F.norm2(c_.res_jac(v, False, False)[0]) for c_ in caps]

    def ratios(duv):
        return [
            F.norm2(c_.res_jac([(v[i] + dl * duv[i]).mid() for i in range(nn)], False, False)[0]) / fr
            for c_, dl, fr in zip(caps, DEL, frozen)
        ]

    def flow_ok(rt):
        c = (
            bool(rt[2] <= arb("0.05")),
            bool(rt[1] / rt[0] <= arb("0.3")),
            bool(rt[2] / rt[1] <= arb("0.3")),
            bool(frozen[2] >= arb("1e-25")),
        )
        return all(c), c

    rt = ratios(du)
    okf, cf = flow_ok(rt)
    items["K3"] = bool(okf and lin_ok)
    say(
        f"K3: linear residual {arbf(lin, 3)} ({lin_ok}); ratios "
        + " ".join(arbf(t, 4) for t in rt)
        + f" conditions {cf} -> {items['K3']}"
    )
    d1 = T.solve_lin(J, [-arb("1.1") * t for t in b])
    ok1, c1 = flow_ok(ratios(d1))
    ctrl["K3c1"] = not ok1
    Jp = arb_mat(nn, nn, [J[i, k] for i in range(nn) for k in range(nn)])
    for j in range(m):
        Jp[m + j, j] = Jp[m + j, j] * arb("1.01")
    d2 = T.solve_lin(Jp, [-t for t in b])
    ok2, c2 = flow_ok(ratios(d2))
    ctrl["K3c2"] = not ok2
    say(
        f"K3 controls: D1 conditions {c1} fails as it must {ctrl['K3c1']}; D2 conditions {c2} fails as it must {ctrl['K3c2']}"
    )
    # K4
    pos_ok = all(du[j] > 0 for j in range(m))
    w1_ok = du[m] > 0
    wc_ok = (du[2 * m + 1] > 0) if odd else True
    items["K4"] = bool(pos_ok and wc_ok and (w1_ok or odd))
    ctrl["K4c"] = not all((-du[j]) > 0 for j in range(m))
    say(
        f"K4: du_j/dA > 0 for all {m} positions {pos_ok} (min {arbf(min(du[:m]), 4)}, max {arbf(max(du[:m]), 4)}); dw_1/dA = {arbf(du[m], 4)} ({w1_ok}); dwe/dA = {arbf(du[2 * m], 4)}"
        + (f"; dwc/dA = {arbf(du[2 * m + 1], 4)} ({wc_ok})" if odd else "")
        + f"; control -du fails {ctrl['K4c']}"
    )
    # K5
    R = list(range(m)) + list(range(2 * m, nn))
    Mc = list(range(m, nn))
    r_ = len(R)
    As = arb_mat(m, m, [J[m + p, q] for p in range(m) for q in range(m)])
    Bs = arb_mat(m, r_, [J[m + p, c] for p in range(m) for c in Mc])
    Cs = arb_mat(r_, m, [J[r, q] for r in R for q in range(m)])
    Dm = arb_mat(r_, r_, [J[r, c] for r in R for c in Mc])
    S = As - Bs * Dm.solve(Cs)
    Sm = [[S[p, q].mid() for q in range(m)] for p in range(m)]
    say(
        f"reduced Hessian {m} x {m}; diagonal from {arbf(min(Sm[p][p] for p in range(m)), 4)} to {arbf(max(Sm[p][p] for p in range(m)), 4)}"
    )
    res5 = MG.matrix_gates(Sm, m, "clean", early=False)
    ctx.prec = 200
    items["K5"] = res5["C_sign"] and res5["C_inv"] and res5["TP"] and res5["TPh"]
    say(
        f"K5: C-sign {res5['C_sign']}, C-inv {res5['C_inv']}, TP {res5['TP']}, TPh {res5['TPh']} -> {items['K5']}"
    )
    ctx.prec = 1600
    Kp = [[(arb(-1) / 10 * ((p - q) ** 2)).exp().mid() for q in range(m)] for p in range(m)]
    ctx.prec = 400
    Spc = [[-Kp[p][q] if (p + q) % 2 == 0 else Kp[p][q] for q in range(m)] for p in range(m)]
    rpc = MG.matrix_gates(Spc, m, "positive control", early=False)
    items["K6"] = rpc["C_sign"] and rpc["C_inv"] and rpc["TP"] and rpc["TPh"]
    Tn = [[Sm[p][q] * (-1 if (p + q) % 2 == 0 else 1) for q in range(m)] for p in range(m)]
    T1 = [row[:] for row in Tn]
    T1[3][7] = -T1[3][7]
    n1 = MG.tp_gate(T1, m, "N1", early=True)
    Sabs = [[abs(x) for x in row] for row in Sm]
    n2s = MG.sign_gate(Sabs, m)
    n2i = MG.inv_gate(Sabs, m)
    Tid = [[arb(1) if p == q else arb(0) for q in range(m)] for p in range(m)]
    n3 = MG.tp_gate(Tid, m, "N3", early=True)
    ctrl["N1"] = not n1[0]
    ctrl["N2"] = (not n2s[0]) and (not n2i)
    ctrl["N3"] = not n3[0]
    items["K7"] = ctrl["N1"] and ctrl["N2"] and ctrl["N3"]
    ctx.prec = 200
    say(
        f"K6 positive control passes all four gates: {items['K6']}; K7 negative controls fail as they must: {items['K7']}"
    )
    bad_ctrl = [k for k, t in ctrl.items() if not t]
    if bad_ctrl or not items["K6"]:
        print(f"VOID q924b {path}: control problem {bad_ctrl} K6 {items['K6']}; items {items}", flush=True)
        sys.exit(2)
    core = all(items[k] for k in ("K1", "K2", "K3", "K4", "K5"))
    print(
        f"VERDICT q924b {os.path.basename(path)}: {'PASS' if core else 'FAIL'}; K1 {items['K1']} K2 {items['K2']} K3 {items['K3']} K4 {items['K4']} K5 {items['K5']} (C-sign {res5['C_sign']} C-inv {res5['C_inv']} TP {res5['TP']} TPh {res5['TPh']}) K6 {items['K6']} K7 {items['K7']}",
        flush=True,
    )
    sys.exit(0 if core else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VOID q924b: exception", flush=True)
        sys.exit(2)
