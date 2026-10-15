"""figsrc (7 Oct 2026): readers of the primary files behind the figures of PAPER3_DRAFT.md. Nothing here is typed in: every value is parsed from an output or a json file of scratchpad_archive_729e03fd/ (and the sources are returned with the data).
Run from temporal_within/ (the paths are relative to that folder)."""
import glob, json, os, re

ARCH = "scratchpad_archive_729e03fd"


def births_law_residuals():
    """returns (list of (K_b, measured A, predicted A, residual), sources). Measured: A_s of tower_K<K_b+1>_As.json (split, K_b odd) or A_cross of tower_K<K_b+1>_birth_out.json (birth, K_b even)."""
    pred = {}; srcs = []
    for fn in ("q635_capacity_K44_69_predictions.json", "q635_capacity_K67_72_predictions.json", "q635_capacity_K72_74_predictions.json", "q635_capacity_K74_79_predictions.json"):
        p = f"{ARCH}/{fn}"
        d = json.load(open(p))["capacity"]["pred"]; srcs.append(p)
        for K, v in d.items(): pred.setdefault(int(K), v["a_pred"])
    out = []
    for Kb in range(61, 80):   # 7 Oct 19:10: extended from 74; a K_b without an event file or a prediction is skipped
        if Kb not in pred: continue
        if Kb % 2 == 1:
            f = f"{ARCH}/tower_K{Kb + 1}_As.json"; key = "A_s"
        else:
            f = f"{ARCH}/tower_K{Kb + 1}_birth_out.json"; key = "A_cross"
        if Kb == 69: f = f"{ARCH}/tower_K70_As_T83.json"   # the copy of T83: the file tower_K70_As.json was written again by the driver of T98
        if not os.path.exists(f): continue
        d = json.load(open(f)); m = float(d[key]); srcs.append(f)
        out.append((Kb, m, pred[Kb], m - pred[Kb]))
    return out, sorted(set(srcs))


def births_law_residuals_deep(D=106.0):
    """8 Oct 2026: residuals measured minus predicted for K_b 61 to 79 with the half-line stage D of centre_scripts/c50_ext_D110.json (stages up to 106, steps of 2) and the pipeline of q1183 (cut 20, window 4 to 30, density of the fitted law); returns (list of (K_b, residual), sources)."""
    import sys
    if "." not in sys.path: sys.path.insert(0, ".")
    import q1183_births_residual_halfline_depth_reading as Q
    ext = json.load(open("centre_scripts/c50_ext_D110.json"))["stages"]
    Q.STAGES = [t for t in Q.STAGES if t["D"] <= 92.0] + [t for t in ext if t["D"] > 92.0 and t["conv"]]
    Ks = list(range(61, 80)); p, _ = Q.predict(D, 20.0, (4.0, 30.0), Ks)
    return [(K, Q.measured(K) - p[K]) for K in Ks], ["centre_scripts/c50_cap_D100_norefit.json", "centre_scripts/c50_ext_D110.json"] + [f"{ARCH}/tower_K{K + 1}_As.json" if K % 2 else f"{ARCH}/tower_K{K + 1}_birth_out.json" for K in Ks]


def births_law_residuals_c50c(D=110.0):
    """8 Oct 2026: residuals measured minus predicted for K_b 61 to 79, and for K_b 80 when the birth file of the event exists, with the c50c half-line stage D of centre_scripts/c50_capc_D110.json (buffer on the derived bulk law, amendment 960; stages up to 110) and the pipeline of q1183 (cut 20, window 4 to 30, fitted-law density). Returns (list of (K_b, residual), sources)."""
    import sys
    if "." not in sys.path: sys.path.insert(0, ".")
    import q1183_births_residual_halfline_depth_reading as Q
    saved = Q.STAGES
    try:
        Q.STAGES = [t for t in json.load(open("centre_scripts/c50_capc_D110.json"))["stages"] if t["conv"]]
        Ks = list(range(61, 80)) + ([80] if os.path.exists(f"{ARCH}/tower_K81_birth_out.json") else []); p, _ = Q.predict(D, 20.0, (4.0, 30.0), Ks)
        out = [(K, Q.measured(K) - p[K]) for K in Ks]
    finally:
        Q.STAGES = saved
    return out, ["centre_scripts/c50_capc_D110.json"] + [f"{ARCH}/tower_K{K + 1}_As.json" if K % 2 else f"{ARCH}/tower_K{K + 1}_birth_out.json" for K in Ks]


def births_law_shared_centre():
    """8 Oct 2026: deviation A_meas - a_pred of the births law with the shared centre function (series to e5 times the immune ratio curve, density of the derived bulk law, c50c capacity stage D = 110; script centre_scripts/q1200_immune_curve_capacity_k61_80.py, amendment 1956) for K_b 61 to 80, with the linear interpolation of the ratio curve (registered) and with a cubic spline of ln(1 - r) (post hoc, amendment 1958). Returns (list of (K_b, linear, spline), sources)."""
    import sys, numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    if "." not in sys.path: sys.path.insert(0, ".")
    import q1183_births_residual_halfline_depth_reading as Q
    cwd = os.getcwd(); sys.path.insert(0, os.path.abspath("centre_scripts")); sys.argv = ["x"]; os.chdir("centre_scripts")
    try:
        import q1200_immune_curve_capacity_k61_80 as M
    finally:
        os.chdir(cwd)
    cs = CubicSpline(M.rr, np.log(1 - M.qq)); rcs = lambda r: 1 - float(np.exp(cs(r)))
    def pred_cs(K): return brentq(lambda a: 2 * M.Nc(a) - K - 1 / 6 - M.dder(float(M.rinv(a + M.oc[0]))) * rcs(float(M.rinv(a + M.oc[0]))), 30.0, 60.0, xtol=1e-14)
    Ks = list(range(61, 81)); out = [(K, Q.measured(K) - M.pred(K), Q.measured(K) - pred_cs(K)) for K in Ks]
    return out, ["centre_scripts/c50_capc_D110.json", "centre_scripts/q645_predictions.json", "centre_scripts/c33_immc_D146.json", "centre_scripts/c19_*.json (immune events of the ratio curve)"] + [f"{ARCH}/tower_K{K + 1}_As.json" if K % 2 else f"{ARCH}/tower_K{K + 1}_birth_out.json" for K in Ks]


def births_law_derived(orders=(5, 6, 7, 8, 10, 12, 14, 16)):
    """8 Oct 2026: deviations A_meas - a_pred of the births law with the derived centre series (coefficients e_2 to e_16 of centre_scripts/d22_out_from_log.json, density dN_half/da; script centre_scripts/q1209_births_law_derived_series.py, amendments 1968 and 1970). Returns (dict with 'orders', 'cap_max', 'npm_max', 'cap_dev', 'npm_dev' at the highest order, 'cap_dev12'), sources."""
    import sys
    if "." not in sys.path: sys.path.insert(0, ".")
    cwd = os.getcwd(); os.environ["E_FILE"] = "d22_out_from_log.json"; sys.path.insert(0, os.path.abspath("centre_scripts")); sys.argv = ["x"]; os.chdir("centre_scripts")
    try:
        import q1209_births_law_derived_series as G
    finally:
        os.chdir(cwd)
    import q1183_births_residual_halfline_depth_reading as Q
    Kc = list(range(61, 81)); Kn = list(range(54, 67)); meas = {K: (json.load(open(f"{ARCH}/tower_K{K + 1}_As2.json"))["A_s"] if K % 2 == 1 else Q.measured(K)) for K in Kc}   # 8 Oct 2026: odd K_b from the rerun to the float floor (amendments 2003 to 2006)
    os.chdir("centre_scripts")
    try:
        npm = {K: json.load(open(f"q607_u1205_K{K}.json" if K != 62 else "q607_u1205r_K62.json"))["a_t"] for K in Kn}
    finally:
        os.chdir(cwd)
    out = {"orders": list(orders), "cap_max": {}, "npm_max": {}}; dev = {}
    for n in orders:
        dc = {K: meas[K] - G.pred_series(G.Ncap, K, n) for K in Kc}; dn = {K: npm[K] - G.pred_series(G.Nnp, K, n, 20.0, 80.0) for K in Kn}
        out["cap_max"][n] = max(abs(v) for v in dc.values()); out["npm_max"][n] = max(abs(v) for v in dn.values()); dev[n] = (dc, dn)
    top = max(orders); out["cap_dev"] = dev[top][0]; out["npm_dev"] = dev[top][1]; out["cap_dev12"] = dev[12][0] if 12 in dev else None
    return out, ["centre_scripts/d22_out_from_log.json", "centre_scripts/c50_capc_D110.json", "centre_scripts/c33_npmc_D110.json", "centre_scripts/q607_u1205_K*.json", f"{ARCH}/tower_K*_As2.json", "centre_scripts/q607_u1205r_K62.json"]


def smallest_flows():
    """returns {K: smallest flow over positions and states} from the readings files (q953 for odd K, q954 for even K) and their sources."""
    out = {}; srcs = []
    pat = re.compile(r"smallest over all positions and states ([0-9.]+)")
    for K in range(63, 80):
        for f in (f"{ARCH}/tower_K{K}_readings.txt", f"{ARCH}/q953_K{K}_out.txt", f"{ARCH}/q954_K{K}_out.txt"):
            if os.path.exists(f):
                m = pat.search(open(f, encoding="utf-8").read())
                if m: out[K] = float(m.group(1)); srcs.append(f); break
    return out, srcs


def density_rho():
    rho = {}; srcs = []
    for fn in ("q635_capacity_K44_69_predictions.json", "q635_capacity_K67_72_predictions.json", "q635_capacity_K72_74_predictions.json", "q635_capacity_K74_79_predictions.json"):
        p = f"{ARCH}/{fn}"; d = json.load(open(p))["capacity"]["pred"]; srcs.append(p)
        for K, v in d.items(): rho.setdefault(int(K), v["rho"])
    return rho, srcs


def roots(path):
    """real positive roots t_k of N1 from an output of q1087; returns (list, header line)."""
    L = open(path, encoding="utf-8").read().splitlines()
    ts = [float(re.search(r"t_\d+\s*=\s*([0-9.]+)", l).group(1)) for l in L if re.match(r"\s+t_\d+", l)]
    return ts, L[0]


def flows_q1082(path):
    """returns [(atom, u', u' - 1)] from the reading q1082."""
    rows = []
    for l in open(path, encoding="utf-8"):
        m = re.match(r"\s+(\d+)\s+u' ([0-9.]+)\s+u'-1 ([+-][0-9.e+-]+)", l)
        if m: rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return rows


def event_profiles(nj=10):
    """returns ([(problem, K, [xi_j], [f_j])], sources): the flows f_j = dp_j/da of the innermost nj positions at the event that ends the family (four-node windows at the last two states, linear extrapolation to the event), xi_j = p_j h / (4 pi), h = p_2 - p_1 at the last state. Capacity: families K 63 to 73 (tower_family_list_K<K>.txt and the event files of q1114); chains: the completed families of results_q584_ext.json and results_q601_ext.json with steps of at most 0.0125 and a last row within 0.03 of the event (the selection of q1115)."""
    import math, sys
    if "." not in sys.path: sys.path.insert(0, ".")
    import q1119_event_flow_profile_reading as Q19
    from q1113_log_flow_reading import families
    from q1114_log_flow_along_family_reading import event
    Q19.NJ = nj; out = []; srcs = []
    for K in range(63, 80):   # 7 Oct 19:10: extended from 74; a family whose list or event file does not exist yet is skipped
        lp = f"{ARCH}/tower_family_list_K{K}.txt"
        if not os.path.exists(lp): continue
        try: Ae, fe = event(K)
        except Exception: continue
        if Ae is None or not os.path.exists(fe): continue
        st = []; srcs += [lp, fe]
        for line in open(lp):
            t = line.split()
            if len(t) < 2: continue
            d = json.load(open(t[1])); u = [float(x) for x in d["u"]]; st.append((float(d["A"]), u[:nj + 1]))
        f, p = Q19.flows_at_event(st, Ae); h = p[1] - p[0]
        out.append(("capacity", K, [p[k] * h / (4 * math.pi) for k in range(nj)], f[:nj]))
    for name, fn in (("immune", "results_q584_ext.json"), ("NPMLE", "results_q601_ext.json")):
        d = json.load(open(fn)); srcs.append(fn); ev = {}
        for e in d["events"]:
            if e["kind"] in ("centre appear", "centre split"): ev.setdefault(e["K_before"], e["a_lo"])
        for c in families(d["rows"])[:-1]:
            K = c[-1]["K"]
            if K not in ev or len(c) < 8: continue
            a = [r["a"] for r in c[-7:]]; s = [a[i + 1] - a[i] for i in range(6)]
            if max(s[1:]) > 0.0125 or ev[K] - a[-1] > 0.03: continue
            S = [(r["a"], [float(x) for x in r["p"][:nj + 1]]) for r in c[-8:]]
            f, p = Q19.flows_at_event(S, ev[K]); h = p[1] - p[0]
            out.append((name, K, [p[k] * h / (4 * math.pi) for k in range(nj)], f[:nj]))
    return out, sorted(set(srcs))


def event_log_flow_rows():
    """returns ([(problem, K, g_ev)], sources): the values of g_ev parsed from the outputs q1114_log_flow_family_out.txt (capacity) and q1115_chain_event_out.txt (immune, NPMLE)."""
    rows = []; p1 = f"{ARCH}/q1114_log_flow_family_out.txt"; p2 = f"{ARCH}/q1115_chain_event_out.txt"
    for line in open(p1, encoding="utf-8"):
        m = re.match(r"K (\d+): .*linear extrapolation to d = 0: ([0-9.]+)", line)
        if m: rows.append(("capacity", int(m.group(1)), float(m.group(2))))
    for line in open(p2, encoding="utf-8"):
        m = re.match(r"\s+(immune|NPMLE) K (\d+) centre (True|False):.*extrapolated to the event ([0-9.]+)", line)
        if m: rows.append((m.group(1), int(m.group(2)), float(m.group(4))))
    return rows, [p1, p2]


def correction_rows():
    """returns (capacity_rows, chain_rows, coef, sources) for Figure 6 (7 Oct 2026, readings q1169b, q1171, q1173).
    capacity_rows: (parity, K, xi_j, y_j) with y_j = (f_j - F(xi_j)) 4 pi^2 rho^2, rho = rho_pred(K), K = 63 up to the last family with an event file; chain_rows: (problem, parity, K, xi_j, y_j, rho_h) with rho_h from the spacing as in q1171.
    coef = {parity: (a, b)} of y = a xi F' + b xi^2 F'' fitted by least squares per parity on the capacity rows."""
    import math, sys
    import numpy as np
    import mpmath as mp
    if "." not in sys.path: sys.path.insert(0, ".")
    import q1119_event_flow_profile_reading as Q19
    import q1114_log_flow_along_family_reading as Q14
    import q1155_apriori_lattice_k76_79_verdict as T
    NJ = 10; Q19.NJ = NJ; rho = T.rho_pred(); F4 = 4 * math.pi ** 2
    Fm = lambda x: mp.coth(mp.pi * x) - mp.pi * x / mp.sinh(mp.pi * x) ** 2
    Fv = lambda x: float(Fm(x)); d1 = lambda x: float(mp.diff(Fm, x)); d2 = lambda x: float(mp.diff(Fm, x, 2))
    cap = []; srcs = []
    for K in range(63, 78):   # 8 Oct: calibration set K 63 to 77 of the tests T141 and T142; K 78 and 79 are the hold-out and stay out of the fit
        lp = f"{ARCH}/tower_family_list_K{K}.txt"
        if not os.path.exists(lp) or K not in rho: continue
        Ae, fe = Q14.event(K)
        if Ae is None or not os.path.exists(fe): continue
        st = []; srcs += [lp, fe]
        for line in open(lp):
            t = line.split()
            if len(t) < 2: continue
            d = json.load(open(t[1])); u = [float(x) for x in d["u"]]; st.append((float(d["A"]), u[:NJ + 1]))
        f, p = Q19.flows_at_event(st, Ae); h = p[1] - p[0]
        for j in range(NJ):
            xi = p[j] * h / (4 * math.pi); cap.append((K % 2, K, xi, (f[j] - Fv(xi)) * F4 * rho[K] ** 2))
    coef = {}
    for par in (0, 1):
        r = [x for x in cap if x[0] == par]
        X = np.array([[x[2] * d1(x[2]), x[2] ** 2 * d2(x[2])] for x in r]); y = np.array([x[3] for x in r]); coef[par] = tuple(np.linalg.lstsq(X, y, rcond=None)[0])
    rows, s2 = event_profiles(NJ); chains = []
    for name, K, xi, f in rows:
        if name == "capacity": continue
        par = K % 2; h = math.sqrt(4 * math.pi * xi[0] / (0.5 if par == 0 else 1.0)); r = (1 + math.sqrt(1 + h * h / math.pi ** 2)) / (2 * h)
        chains.append((name, par, K, xi, [(f[j] - Fv(xi[j])) * F4 * r * r for j in range(len(xi))], r))
    return cap, chains, coef, sorted(set(srcs + s2))
