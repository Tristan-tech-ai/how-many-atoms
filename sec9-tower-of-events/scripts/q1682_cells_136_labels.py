"""q1682 (10 Oct 2026, READING, no verdict word): labels for the missing birth K_b 136 against the bars registered in q1681_prereg_136.txt BEFORE the event exists (listing: no event file in any directory), for the matched cells (NG, nmax) = (13, 14), (14, 14) and (14, 16) of q1680, with a selftest. The wall pair of each cell is refitted inside the script on K_b 83..130 (four linearisation rounds, as in q1681). Differences a_pred - A only; the amplitude is loaded inside the script (BLIND RULE).
Bars (copied from q1681_prereg_136.txt): cell (14, 14) |a_pred - A| < 5e-13; cell (13, 14) a_pred - A in [-6e-13, -2e-13]; cell (14, 16) a_pred - A in [-3.5e-13, -1.2e-13]. Registered predictions: 77.388989108976 (13, 14), 77.388989108977 (14, 14), 77.388989108976 (14, 16).
Selftest predictions (registration text of Q65, written before the first run; P1 is applied per cell because the three cells predict slightly different amplitudes):
 P1 on a synthetic event A = a_pred(cell) + 0 for (14, 14), + 4e-13 for (13, 14) and + 2e-13 for (14, 16) (differences 0, -4e-13, -2e-13) the bar of the corresponding cell holds;
 P2 on a synthetic event shifted by -1e-11 (difference +1e-11) the bars of all three cells do not hold;
 P3 with no event file the script prints NO EVENT and exits 0 (subprocess);
 P4 an event file without the key A_event makes it print VOID (flushed) and exit 2 (subprocess);
 P5 control: the cell predictions equal the registered printed values to 5e-13.
Usage: py q1682_cells_136_labels.py (real event from BL_FWPROD or fw_prod) | py q1682_cells_136_labels.py --selftest (from small_gap_q206/).
"""

import json, os, re, subprocess, sys, tempfile
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
CELLS = ((13, 14), (14, 14), (14, 16))
BARS = {
    (14, 14): ("abs", 5e-13),
    (13, 14): ("range", -6e-13, -2e-13),
    (14, 16): ("range", -3.5e-13, -1.2e-13),
}
REG = {(13, 14): 77.388989108976, (14, 14): 77.388989108977, (14, 16): 77.388989108976}
lab = lambda q: "holds" if q else "does not hold"


def held(c, d):
    b = BARS[c]
    return abs(d) < b[1] if b[0] == "abs" else b[1] <= d <= b[2]


def predictions():
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
        Kin = [K for K in range(83, 131) if B.measured(K) is not None]
        min_ = np.array([B.measured(K) for K in Kin])
        out = {}
    finally:
        if save is not None:
            os.environ["BL_FWPROD"] = save
    return Kin, min_, E, FALL


_cache = {}


def cell_pred(c, K=136):
    if c not in _cache:
        if "inputs" not in _cache:
            _cache["inputs"] = predictions()
        Kin, min_, E, FALL = _cache["inputs"]
        NG, nmax = c
        B.F = FALL[: NG + 1]
        B.NG = NG
        B.g = [1.0]
        for n in range(1, NG + 1):
            B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))
        ex = [E[n] for n in range(6, nmax + 1)]
        nu, d0 = -0.0937736508347, 0.0011599160063
        h = 1e-7
        for rnd in range(4):
            p0 = np.array([B.a_pred(k, nu, d0, ex) for k in Kin])
            M = np.stack(
                [
                    [(B.a_pred(k, nu + h, d0, ex) - B.a_pred(k, nu - h, d0, ex)) / (2 * h) for k in Kin],
                    [(B.a_pred(k, nu, d0 + h, ex) - B.a_pred(k, nu, d0 - h, ex)) / (2 * h) for k in Kin],
                ],
                1,
            )
            sol, *_ = np.linalg.lstsq(M, -(p0 - min_), rcond=None)
            nu, d0 = nu + sol[0], d0 + sol[1]
        _cache[c] = B.a_pred(K, nu, d0, ex)
    return _cache[c]


def evaluate(verbose=True, override=None):
    fn = os.path.join(os.environ.get("BL_FWPROD", os.path.join(H, "fw_prod")), "event_Kb136.json")
    if not os.path.exists(fn):
        return None
    try:
        A = float(json.load(open(fn))["A_event"])
    except Exception as e:
        print("VOID: event file of K_b 136 unreadable (%s)" % type(e).__name__, flush=True)
        os._exit(2)
    out = {}
    for c in CELLS:
        d = cell_pred(c) - A
        out[c] = (d, held(c, d))
        if verbose:
            print("K_b 136 cell NG %d nmax %d: a_pred - A = %+.3e (bar %s)" % (*c, d, lab(out[c][1])))
    return out


if "--selftest" in sys.argv:
    preds = {c: cell_pred(c) for c in CELLS}
    print("cell predictions: " + ", ".join("NG %d nmax %d %.12f" % (*c, preds[c]) for c in CELLS))

    def make(d, A):
        os.makedirs(d, exist_ok=True)
        json.dump({"A_event": A}, open(os.path.join(d, "event_Kb136.json"), "w"))

    res = {}
    with tempfile.TemporaryDirectory() as t:
        shifts = {(14, 14): 0.0, (13, 14): 4e-13, (14, 16): 2e-13}
        for c, s in shifts.items():
            d = os.path.join(t, "ok%d%d" % c)
            make(d, preds[c] + s)
            os.environ["BL_FWPROD"] = d
            res[("ok", c)] = evaluate(verbose=False)[c][1]
        d = os.path.join(t, "bad")
        make(d, preds[(14, 14)] - 1e-11)
        os.environ["BL_FWPROD"] = d
        bad = evaluate(verbose=False)
        d3 = os.path.join(t, "none")
        os.makedirs(d3)
        d4 = os.path.join(t, "corrupt")
        os.makedirs(d4)
        json.dump({"x": 1}, open(os.path.join(d4, "event_Kb136.json"), "w"))
        r3 = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env=dict(os.environ, BL_FWPROD=d3),
            capture_output=True,
            text=True,
        )
        r4 = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env=dict(os.environ, BL_FWPROD=d4),
            capture_output=True,
            text=True,
        )
    p1 = all(res[("ok", c)] for c in CELLS)
    p2 = not any(v[1] for v in bad.values())
    p3 = r3.returncode == 0 and "NO EVENT" in r3.stdout
    p4 = r4.returncode == 2 and "VOID" in r4.stdout
    p5 = all(abs(preds[c] - REG[c]) < 5e-13 for c in CELLS)
    print(
        "P1 bars hold on the synthetic events at the cell predictions: %s; P2 bars do not hold at +1e-11: %s; P3 no file gives NO EVENT, exit 0: %s; P4 file without A_event gives VOID, exit 2: %s; P5 control predictions: %s"
        % (lab(p1), lab(p2), lab(p3), lab(p4), lab(p5))
    )
    print("SELFTEST " + ("PASS" if p1 and p2 and p3 and p4 and p5 else "FAIL"))
    sys.exit(0)
r = evaluate()
if r is None:
    print("NO EVENT K_b 136 in the event directory: no label")
    sys.exit(0)
print("READING q1682 done (labels by the script; no verdict word)")
