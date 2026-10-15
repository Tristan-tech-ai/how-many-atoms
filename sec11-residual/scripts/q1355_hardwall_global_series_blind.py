"""q1355 (8 Oct 2026, registered in amendment 2122): blind test, on the CONVERGED instrument (tight_run.py with TIGHT_DEG=17: repaired density inversion, spline degree 17 for the prediction a_med, degree 19 for the convergence check), of the three-term hard-wall series R / Da = P0(k) + P1(k) / v^2 + P2(k) / v^4 with P0 a polynomial of degree 4 and P1, P2 polynomials of degree 1 in k, on three new exponents over K_b 14 to 45.
Frozen: the coefficients of the global fit (model (4, 1, 1) of q1355a) by linear least squares on the relative misfit over all events with R > 0 of q1352_events.json (hard_wall_deg17: exponents 0.02, 0.05, 0.052-0.058, 0.06, 0.075, 0.09, 0.1, 0.25, 0.35, 0.5, 0.75); the sha256 of that file is in the table. The test exponents k = 0.15, 0.4, 0.6 are not in it.
Test exponents k = 0.15, 0.4, 0.6 (stages c52_hw015c, hw04c, hw06c, D = 92); events K_b 14 to 45 (32 each, 96 in all). The events K_b 33 to 45 were located for q1345 (files c61_u1345_*; their amplitudes were seen only through the degree-5 readout of q1345, not as degree-17 R / Da); the events K_b 14 to 32 are new chains (files c61_u1355_*, start A0 = a_med - 0.0045, step 0.003, 160 bits).
Located rule: A_t returned, last Newton residual below 1e-18, |A_t - last A| < 1e-9; both interpolants (degree 17 and degree 19).
Items H1 (k = 0.15), H2 (k = 0.4), H3 (k = 0.6): at least 24 of the 32 events located and |R / Da - prediction| <= 0.006 at every located event, for both interpolants, where R = (A_t - a_med) 1e9, v^2 = Z.v2f(N17, A_t). Class PASS if H1, H2 and H3 hold; INCOMPLETE if for some exponent fewer than 24 are located; otherwise FAIL.
Bar derivation: leave-one-exponent-out cross-validation of the same model on k = 0.1, 0.25, 0.35, 0.5 (q1355a_out.txt) gives a largest absolute misfit 0.0021 in R / Da over K_b 14-40; the all-data rms of the relative misfit is 0.00087; the bar 0.006 is 2.9 times the cross-validation maximum. The old two-term form (C1 = 143.756 with the same P0) misses by more than the bar (control c7).
Priors: H1, H2, H3 0.65 each, class 0.35. Competing reading: the coefficients P1, P2 are not linear in k outside the fitted exponents, or the K_b 14-20 events (v^2 66-77) need a fourth term.
Controls (registration pass): c0 the instrument: TIGHT_DEG is 17 and the a_med of stage hw025 at K_b 30 equals an independent implementation (scipy make_interp_spline of degree 17 through the residuals at the atoms, own root finder) to 1e-13; c1 item function; c2 Da(115) = 0.21082; c3 the frozen model reproduces the training events (rms of the relative misfit 0.00087 within 5e-5; the largest in-sample |R / Da - prediction| is printed and at most the bar); c4 the chain controls; c5 the three stages converged to D = 92; c6 the cross-validation maximum recomputed here is 0.0021 within 2e-4 and the bar is at least 2.5 times it; c7 power: the two-term form misses the bar at one or more events of every test exponent; c8 |a_17 - a_19| 1e9 / Da <= 0.001 at every test event (no decision role).
Usage: TIGHT_DEG=17 py tight_run.py q1355_hardwall_global_series_blind.py --registration-run | --read   (from temporal_within/centre_scripts/; both calls must run through tight_run.py with TIGHT_DEG=17, otherwise the script ends with VOID, exit 2)
"""

import hashlib, io, contextlib, functools, json, os, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
if os.environ.get("TIGHT_DEG") != "17":
    print("q1355: VOID (instrument: TIGHT_DEG=17 and tight_run.py are required)")
    sys.exit(2)
import numpy as np

os.environ["E_FILE"] = "d22_out_from_log.json"
with contextlib.redirect_stdout(io.StringIO()):
    import q1321_hardwall_common_c1_blind as B
H, QQ, Z, nhw = B.H, B.QQ, B.Z, B.nhw
Da = functools.lru_cache(maxsize=None)(B.Da)
BAR = 0.006
KS = list(range(14, 46))
NEED = 24
DEGS = (4, 1, 1)
TAGS = {0.15: "hw015", 0.4: "hw04", 0.6: "hw06"}
EVENTS = "q1352_events.json"


def design(ev, degs):
    rows = []
    meta = []
    for k, evs in ev.items():
        for K, R, v in evs:
            y = R / Da(v)
            x = 1.0 / v
            row = []
            for n, d in enumerate(degs):
                for j in range(d + 1):
                    row.append(k**j * x**n / y)
            rows.append(row)
            meta.append((k, K, v, y))
    return np.array(rows), meta


def fit(ev, degs):
    A, _ = design(ev, degs)
    c, *_ = np.linalg.lstsq(A, np.ones(len(A)), rcond=None)
    return c


def predict(c, k, v, degs=DEGS):
    out = 0.0
    p = 0
    for n, d in enumerate(degs):
        out += sum(c[p + j] * k**j for j in range(d + 1)) / v**n
        p += d + 1
    return out


def training():
    hw = {float(k): v for k, v in json.load(open(EVENTS))["hard_wall_deg17"].items()}
    return {k: [e for e in evs if e[1] > 0] for k, evs in hw.items() if any(e[1] > 0 for e in evs)}


def inside(y):
    return abs(y) <= BAR


def evfile(tag, K):
    return f"c61_u1345_{tag}_K{K}" if K >= 33 else f"c61_u1355_{tag}_K{K}"


def independent_amed(tag, K):
    from scipy.interpolate import make_interp_spline
    from scipy.optimize import brentq, least_squares

    st = [s for s in json.load(open(f"c52_{tag}c.json"))["stages"] if s["D"] == 92.0 and s["conv"]][-1]
    d = np.array(st["d"][: 1 + st["nfree"]])
    j = np.arange(1, len(d) + 1) - 0.5
    mm = (d >= 4) & (d <= 30)
    law = tuple(
        least_squares(lambda q: np.array([H.NOF(x, q[0], q[1]) for x in d[mm]]) - j[mm], [0.0558, 0.5482]).x
    )
    X = d[d <= 72.0]
    sp = make_interp_spline(X, np.arange(1, len(X) + 1) - 0.5 - np.array([H.NOF(x, *law) for x in X]), k=17)
    N = lambda a: H.NOF(a, *law) + float(sp(a))
    z = lambda a: 2 * N(a) - K - 1 / 6
    a0 = brentq(z, 5.0, 55.0, xtol=1e-12)
    f = lambda a: z(a) - float(QQ.T3(1.0 / (4 * np.pi * (N(a + QQ.H) - N(a - QQ.H)) / (2 * QQ.H)) ** 2))
    return brentq(f, a0 - 2.5, a0 + 2.5, xtol=1e-14)


def regrun():
    tr = training()
    c = fit(tr, DEGS)
    A, meta = design(tr, DEGS)
    rms = float(np.sqrt(np.mean((A @ c - 1) ** 2)))
    insample = max(abs(predict(c, k, v) - y) for k, K, v, y in meta)
    cvmax = 0.0
    for kh in (0.1, 0.25, 0.35, 0.5):
        ch = fit({k: v for k, v in tr.items() if k != kh}, DEGS)
        for K, R, v in tr[kh]:
            cvmax = max(cvmax, abs(predict(ch, kh, v) - R / Da(v)))
    N17, _ = nhw("c52_hw025c.json", 92.0, 5)
    c0 = abs(QQ.root(N17, 30) - independent_amed("hw025", 30)) < 1e-13
    c1 = inside(0.0) and inside(0.006) and inside(-0.006) and not inside(0.0061) and not inside(-0.0061)
    c2 = abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082
    c3 = abs(rms - 0.00087) < 5e-5 and insample <= BAR
    a0 = json.load(open("c61_ctl30k0.json"))["A_t"]
    a1 = json.load(open("c61_ctl30k1em6.json"))["A_t"]
    ast = json.load(open("c60_small_K30.json"))["A_t"]
    c4 = abs(a0 - ast) < 1e-9 and 1e-8 < abs(a1 - ast) < 1e-5
    c5 = all(H.stage_ok(t) for t in TAGS.values())
    c6 = abs(cvmax - 0.0021) < 2e-4 and BAR >= 2.5 * cvmax
    out = {
        "events_sha256": hashlib.sha256(open(EVENTS, "rb").read()).hexdigest(),
        "degs": list(DEGS),
        "coef": c.tolist(),
        "bar": BAR,
    }
    w = 0.0
    powok = True
    for k, tag in TAGS.items():
        sj = f"c52_{tag}c.json"
        N17, law = nhw(sj, 92.0, 5)
        N19, _ = nhw(sj, 92.0, 7)
        stl = [s for s in json.load(open(sj))["stages"] if s["D"] == 92.0][-1]
        out[str(k)] = {"law": list(law), "wall_mass": stl["m"][0]}
        far = 0
        for K in KS:
            a17, a19 = QQ.root(N17, K), QQ.root(N19, K)
            v2 = Z.v2f(N17, a17)
            dd = Da(v2)
            w = max(w, abs(a17 - a19) * 1e9 / dd)
            old = sum(c[: DEGS[0] + 1][j] * k**j for j in range(DEGS[0] + 1)) - 143.756 / v2
            far += int(not inside(old - predict(c, k, v2)))
            out[str(k)][str(K)] = dict(
                a_med=a17, a_med19=a19, v2=float(v2), pred=predict(c, k, v2), a0=float(a17 - 0.0045)
            )
        powok = powok and far >= 1
    c7 = powok
    c8 = w <= 0.001
    print(
        f"model {DEGS}: rms {rms:.5f}, in-sample largest |dR/Da| {insample:.4f}, cross-validation maximum {cvmax:.4f}; control c0 {c0}; c1 {c1}; c2 {c2}; c3 {c3}; c4 {c4}; c5 {c5}; c6 {c6}; c7 (two-term form fails the bar) {c7}; c8 (max |a17 - a19| / Da {w:.5f}) {c8}"
    )
    if not (c0 and c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8):
        print("REGISTRATION-RUN q1355: VOID (control)")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1355_predictions.json", "w"), indent=1)
    for k, tag in TAGS.items():
        law = out[str(k)]["law"]
        for part, Kp in (("a", list(range(14, 24))), ("b", list(range(24, 33)))):
            L = [
                "#!/bin/bash",
                f"# run1355{tag}{part}.sh (8 Oct 2026): hard-wall chain c61 for k = {k}, K_b {Kp[0]}-{Kp[-1]}; start A0 = a_med - 0.0045, step 0.003 (q1355_predictions.json)",
                'cd "./centre_scripts"',
                f"export KEXP={k} HL_JSON=c52_{tag}c.json HL_STAGE=92 LAW0={law[0]:.5f},{law[1]:.5f} WM0={out[str(k)]['wall_mass']:.6f}",
            ]
            for K in Kp:
                L.append(
                    f"py c61_hw_fullchain.py {K} {out[str(k)][str(K)]['a0']:.6f} 0.003 160 u1355_{tag}_K{K} > c61_u1355_{tag}_K{K}.log 2>&1"
                )
            L.append(f"echo done $(date +%H:%M:%S) > run1355{tag}{part}.done")
            open(f"run1355{tag}{part}.sh", "w", newline="\n").write("\n".join(L) + "\n")
    print("sha256", sh)
    print("REGISTRATION-RUN q1355: controls True, table and run1355*.sh written")


def read():
    js = json.load(open("q1355_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1355: VOID (table hash)")
        sys.exit(2)
    if hashlib.sha256(open(EVENTS, "rb").read()).hexdigest() != js["events_sha256"]:
        print("READING q1355: VOID (training events changed)")
        sys.exit(2)
    res = {}
    incomplete = False
    for k, tag in TAGS.items():
        sj = f"c52_{tag}c.json"
        N17, _ = nhw(sj, 92.0, 5)
        N19, _ = nhw(sj, 92.0, 7)
        rows = []
        for K in KS:
            A = H.locr(evfile(tag, K))
            if A is None:
                print(f"k {k} K_b {K}: NOT LOCATED")
                continue
            e = js[str(k)][str(K)]
            v2 = Z.v2f(N17, A)
            d = Da(v2)
            pr = predict(np.array(js["coef"]), k, v2)
            y17 = (A - QQ.root(N17, K)) * 1e9 / d - pr
            y19 = (A - QQ.root(N19, K)) * 1e9 / d - pr
            rows.append((K, y17, y19))
            print(
                f"k {k} K_b {K}: v^2 {v2:.2f}, R/Da {(A - QQ.root(N17, K)) * 1e9 / d:+.4f}, predicted {pr:+.4f}, dR/Da {y17:+.4f} (degree 19 {y19:+.4f})"
            )
        if len(rows) < NEED:
            incomplete = True
            print(f"k {k}: located {len(rows)} of {len(KS)}")
            continue
        res[k] = (
            all(inside(a) and inside(b) for _, a, b in rows),
            len(rows),
            max(abs(a) for _, a, _ in rows),
        )
        print(
            f"k {k}: located {len(rows)}; largest |dR/Da| {res[k][2]:.4f}; item {'H1' if k == 0.15 else 'H2' if k == 0.4 else 'H3'}: {res[k][0]}"
        )
    if incomplete:
        print("VERDICT q1355 hard-wall global-series class: INCOMPLETE")
        return
    print(
        f"VERDICT q1355 hard-wall global-series class: {'PASS' if all(v[0] for v in res.values()) else 'FAIL'}; "
        + "; ".join(
            f"k {k}: {res[k][0]} ({res[k][1]} located, largest |dR/Da| {res[k][2]:.4f})" for k in TAGS
        )
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
