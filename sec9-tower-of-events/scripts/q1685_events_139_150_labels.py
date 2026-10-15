"""q1685 (10 Oct 2026, READING, no verdict word): labels for the future capacity events K_b 139..150 against the bars registered in q1684_prereg_139_150.txt BEFORE any such event exists, with a selftest. The wall pair of each cell is refitted inside the script on the 55 events K_b 83..138 (four linearisation rounds, bulk law through c28, stored series), exactly as in q1684. Differences a_pred - A only; the amplitudes are loaded inside the script (BLIND RULE).
Bars (copied from q1684_prereg_139_150.txt): cell (14, 14) |a_pred - A| < 5e-13 for K_b 139..145 and < 1e-12 for K_b 146..150; full series (14, 18) |a_pred - A| < 5e-12 and the sign of a_pred - A negative.
Selftest predictions (registration text of Q75, written before the first run):
 P1 on synthetic events A = a_pred(14, 14) for every K_b 139..150 the (14, 14) bar holds and the (14, 18) bar and sign hold;
 P2 on synthetic events A = a_pred(14, 14) - 2e-12 (difference +2e-12) the (14, 14) bar does not hold, and the (14, 18) sign does not hold;
 P3 with no event files the script prints NO EVENT and exits 0 (subprocess);
 P4 an event file without the key A_event makes it print VOID (flushed) and exit 2 (subprocess);
 P5 control: the predictions of the two cells equal those of q1684_prereg_139_150.txt to 5e-13.
Usage: py q1685_events_139_150_labels.py (real events from BL_FWPROD or fw_prod) | py q1685_events_139_150_labels.py --selftest (from small_gap_q206/).
"""

import json, os, re, subprocess, sys, tempfile
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
CELLS = ((14, 14), (14, 18))
KS = tuple(range(139, 151))
lab = lambda q: "holds" if q else "does not hold"


def bar14(K, d):
    return abs(d) < (5e-13 if K <= 145 else 1e-12)


def bar18(K, d):
    return abs(d) < 5e-12


_cache = {}


def preds():
    if "p" in _cache:
        return _cache["p"]
    S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
    E = {n: float(S20["e%d" % n]) for n in range(2, 21)}
    txt = open(os.path.join(H, "q1557_stdout_ORD28.txt"), encoding="utf-8").read()
    f = {}
    for n in (26, 28):
        m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
        f[n] = float(
            mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30))
        )
    FALL = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28]]
    save = os.environ.pop("BL_FWPROD", None)
    try:
        Kin = [K for K in range(83, 139) if B.measured(K) is not None]
        min_ = np.array([B.measured(K) for K in Kin])
    finally:
        if save is not None:
            os.environ["BL_FWPROD"] = save
    B.F = FALL[:15]
    B.NG = 14
    B.g = [1.0]
    for n in range(1, 15):
        B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))
    out = {}
    h = 1e-7
    for c in CELLS:
        ex = [E[n] for n in range(6, c[1] + 1)]
        nu, d0 = -0.0937736508347, 0.0011599160063
        for rnd in range(4):
            p0 = np.array([B.a_pred(K, nu, d0, ex) for K in Kin])
            M = np.stack(
                [
                    [(B.a_pred(K, nu + h, d0, ex) - B.a_pred(K, nu - h, d0, ex)) / (2 * h) for K in Kin],
                    [(B.a_pred(K, nu, d0 + h, ex) - B.a_pred(K, nu, d0 - h, ex)) / (2 * h) for K in Kin],
                ],
                1,
            )
            sol, *_ = np.linalg.lstsq(M, -(p0 - min_), rcond=None)
            nu, d0 = nu + sol[0], d0 + sol[1]
        out[c] = {K: B.a_pred(K, nu, d0, ex) for K in KS}
    _cache["p"] = out
    return out


def evaluate(verbose=True):
    base = os.environ.get("BL_FWPROD", os.path.join(H, "fw_prod"))
    found = {}
    for K in KS:
        fn = os.path.join(base, "event_Kb%d.json" % K)
        if os.path.exists(fn):
            try:
                found[K] = float(json.load(open(fn))["A_event"])
            except Exception as e:
                print("VOID: event file of K_b %d unreadable (%s)" % (K, type(e).__name__), flush=True)
                os._exit(2)
    if not found:
        return None
    P = preds()
    res = {}
    for K, A in sorted(found.items()):
        d14 = P[(14, 14)][K] - A
        d18 = P[(14, 18)][K] - A
        res[K] = (d14, d18, bar14(K, d14), bar18(K, d18), d18 < 0)
        if verbose:
            print(
                "K_b %d: cell (14, 14) diff %+.3e (bar %s); full series (14, 18) diff %+.3e (bar %s, sign negative %s)"
                % (K, d14, lab(res[K][2]), d18, lab(res[K][3]), lab(res[K][4]))
            )
    return res


if "--selftest" in sys.argv:
    P = preds()
    print(
        "predictions K_b 139: (14, 14) %.12f, (14, 18) %.12f; K_b 150: %.12f, %.12f"
        % (P[(14, 14)][139], P[(14, 18)][139], P[(14, 14)][150], P[(14, 18)][150])
    )

    def make(d, shift):
        os.makedirs(d, exist_ok=True)
        for K in KS:
            json.dump({"A_event": P[(14, 14)][K] + shift}, open(os.path.join(d, "event_Kb%d.json" % K), "w"))

    with tempfile.TemporaryDirectory() as t:
        d1 = os.path.join(t, "ok")
        make(d1, 0.0)
        os.environ["BL_FWPROD"] = d1
        r1 = evaluate(verbose=False)
        d2 = os.path.join(t, "bad")
        make(d2, -2e-12)
        os.environ["BL_FWPROD"] = d2
        r2 = evaluate(verbose=False)
        d3 = os.path.join(t, "none")
        os.makedirs(d3)
        d4 = os.path.join(t, "corrupt")
        os.makedirs(d4)
        json.dump({"x": 1}, open(os.path.join(d4, "event_Kb139.json"), "w"))
        s3 = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env=dict(os.environ, BL_FWPROD=d3),
            capture_output=True,
            text=True,
        )
        s4 = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env=dict(os.environ, BL_FWPROD=d4),
            capture_output=True,
            text=True,
        )
    p1 = all(v[2] and v[3] and v[4] for v in r1.values()) and len(r1) == 12
    p2 = all((not v[2]) and (not v[4]) for v in r2.values())
    p3 = s3.returncode == 0 and "NO EVENT" in s3.stdout
    p4 = s4.returncode == 2 and "VOID" in s4.stdout
    reg = open(os.path.join(H, "q1684_prereg_139_150.txt")).read()
    ok5 = True
    for K in KS:
        m = re.search(
            r"K_b %d: NG 14 nmax 14 a_pred = ([\d.]+); NG 13 nmax 14 a_pred = [\d.]+; NG 14 nmax 16 a_pred = [\d.]+; NG 14 nmax 18 a_pred = ([\d.]+)"
            % K,
            reg,
        )
        ok5 &= (
            bool(m)
            and abs(float(m.group(1)) - P[(14, 14)][K]) < 5e-13
            and abs(float(m.group(2)) - P[(14, 18)][K]) < 5e-13
        )
    print(
        "P1 labels hold on synthetic events at the (14, 14) prediction: %s; P2 the (14, 14) bar and the (14, 18) sign do not hold at +2e-12: %s; P3 no files gives NO EVENT, exit 0: %s; P4 file without A_event gives VOID, exit 2: %s; P5 control predictions equal the registered ones: %s"
        % (lab(p1), lab(p2), lab(p3), lab(p4), lab(ok5))
    )
    print("SELFTEST " + ("PASS" if p1 and p2 and p3 and p4 and ok5 else "FAIL"))
    sys.exit(0)
r = evaluate()
if r is None:
    print("NO EVENT K_b 139..150 in the event directory: no label")
    sys.exit(0)
print("READING q1685 done (labels by the script; no verdict word)")
