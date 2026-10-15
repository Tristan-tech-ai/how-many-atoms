"""q1452 (9 Oct 2026): VERDICT gate for the continuation of the margin law of the innermost-atom flow to the rows K = 115 to 122 (track B of PROOF_TRACKS_2026-10-09.md), written after VERDICT q1446: M1 PASS; M2 PASS; M3 PASS (rows K = 109 to 114).
READING that led to it (after the verdict of q1446): the errors of the registered law L + a K^-(2/3) have the same sign within each parity and grow with K (P/N: -1.63, -2.37, -3.22 and +2.76, +3.55, +4.34 in units of 1e-6 for K = 109 to 114), so the exponent 2/3 is a local law. Two laws are therefore registered, both fitted on the rows K = 83 to 114 (training snapshot q1452_training_rows.json, 32 rows, parities apart, per quantity P/N and R_e):
  law A: a0 + a1 K^-(2/3) (fixed exponent, the law of q1446);
  law B: a0 + a1 K^-p with a free exponent p in [0.3, 1.2] per sequence (least squares, scipy; started at a0 = 1.04, a1 = 1, p = 0.7).
The null is the exponent 1. The bar of each law and quantity is twice its largest hold-out error: the fit on K <= 106 (law B refits p), tested on K = 107 to 114. Table q1452_margin_table.json (sha256 REG below) holds predictions of both laws and the null for K = 115 to 122 and the bars; it is recomputed from the snapshot (VOID if it differs by more than 1e-9).
Items, for the measured rows K = 115 to 122 of small_gap_q206/q1416_rows.json (a row counts only with its own control met):
  A: |PN - PN_A| < bar_A and |Re - Re_A| < bar_A of the parity, for every measured row.
  B: the same for law B with bar_B.
  C: PN > 1 and Re > 1 for every measured row.
Each item: FAIL if one measured row violates, INCOMPLETE if none violates and a row of 115 to 122 is missing, PASS otherwise. A FAIL of item A together with PASS of item B would show that the fixed exponent 2/3 does not continue. VOID (exit 2) if a hash differs, the table does not match its recomputation, a row control failed, a file is unreadable or a control of the gate fails. Last line: VERDICT q1452: A ...; B ...; C ... Usage: py q1452_margin_law_ext_verdict.py [--registration-run] (from temporal_within/centre_scripts/).
"""

import hashlib, json, os, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(HERE, "q1452_training_rows.json")
TABLE = os.path.join(HERE, "q1452_margin_table.json")
LIVE = os.path.join(ROOT, "small_gap_q206", "q1416_rows.json")
REG_TRAIN = "b5c18276f692109003d34caccead13d2d0111c6662d0bc815f453c3a947a8864"
REG_TABLE = "16ef0c20818c8209abef22de68378d489291a8cfecb150b96f575be59c3c17ba"
P_A = 2.0 / 3.0
P_NULL = 1.0
TARGET = list(range(115, 123))
QUANT = ("PN", "Re")
PARITY = ("even", "odd")


def sha_of(js):
    js = dict(js)
    js.pop("sha256", None)
    return hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()


def load_hashed(path, reg):
    js = json.load(open(path))
    h = sha_of(js)
    if not (h == reg and js.get("sha256") == reg):
        raise ValueError(f"hash of {os.path.basename(path)} is {h[:16]}..., registered {reg[:16]}...")
    return js


def fit_fixed(points, p):
    K = np.array([k for k, _ in points], float)
    y = np.array([v for _, v in points])
    M = np.vstack([np.ones_like(K), K ** (-p)]).T
    c = np.linalg.lstsq(M, y, rcond=None)[0]
    return float(c[0]), float(c[1]), p


def fit_free(points):
    K = np.array([k for k, _ in points], float)
    y = np.array([v for _, v in points])
    r = least_squares(
        lambda z: z[0] + z[1] * K ** (-z[2]) - y,
        [1.04, 1.0, 0.7],
        bounds=([0.5, -1e3, 0.3], [2.0, 1e3, 1.2]),
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
    )
    return float(r.x[0]), float(r.x[1]), float(r.x[2])


def ev(c, K):
    return c[0] + c[1] * K ** (-c[2])


def build(rows):
    out = {"A": {}, "B": {}, "null": {}, "bar_A": {}, "bar_B": {}, "p_B": {}}
    for par in PARITY:
        out["bar_A"][par] = {}
        out["bar_B"][par] = {}
        out["p_B"][par] = {}
        for q in QUANT:
            s = sorted((r["K"], r[q]) for r in rows if r["parity"] == par)
            tr = [(k, v) for k, v in s if k <= 106]
            te = [(k, v) for k, v in s if k >= 107]
            cA = fit_fixed(tr, P_A)
            cB = fit_free(tr)
            out["bar_A"][par][q] = 2 * max(abs(ev(cA, k) - v) for k, v in te)
            out["bar_B"][par][q] = 2 * max(abs(ev(cB, k) - v) for k, v in te)
            fA = fit_fixed(s, P_A)
            fB = fit_free(s)
            fN = fit_fixed(s, P_NULL)
            out["p_B"][par][q] = fB[2]
            for K in TARGET:
                if (K % 2 == 0) == (par == "even"):
                    out["A"].setdefault(str(K), {})[q] = float(ev(fA, K))
                    out["B"].setdefault(str(K), {})[q] = float(ev(fB, K))
                    out["null"].setdefault(str(K), {})[q] = float(ev(fN, K))
    return out


def judge(tab, meas):
    res = {}
    for item in ("A", "B", "C"):
        viol = []
        miss = []
        for K in TARGET:
            m = meas.get(K)
            if m is None:
                miss.append(K)
                continue
            par = "even" if K % 2 == 0 else "odd"
            for q in QUANT:
                v = m[q]
                if item in ("A", "B"):
                    pr = tab[item][str(K)][q]
                    bar = tab["bar_" + item][par][q]
                    if not abs(v - pr) < bar:
                        viol.append((K, q, v - pr))
                elif not v > 1.0:
                    viol.append((K, q, v))
        res[item] = ("FAIL" if viol else ("INCOMPLETE" if miss else "PASS"), viol, miss)
    return res


def live_rows():
    meas = {}
    for r in json.load(open(LIVE)):
        if r["K"] in TARGET:
            if r.get("control_ok") is not True:
                raise ValueError(f"row K = {r['K']} has a failed control")
            if (r["K"] % 2 == 0) != (r["parity"] == "even"):
                raise ValueError(f"row K = {r['K']} has parity {r['parity']}")
            meas[r["K"]] = {"PN": float(r["PN"]), "Re": float(r["Re"])}
    return meas


def main(loader=None, train=None, table=None):
    try:
        tj = load_hashed(train or TRAIN, REG_TRAIN)
        tab = load_hashed(table or TABLE, REG_TABLE)
        again = build(tj["rows"])
        for key in ("A", "B", "null"):
            for K in tab[key]:
                for q in QUANT:
                    if abs(tab[key][K][q] - again[key][K][q]) > 1e-9:
                        raise ValueError(f"table {key} K = {K} {q} differs from its recomputation")
        for key in ("bar_A", "bar_B"):
            for par in PARITY:
                for q in QUANT:
                    if abs(tab[key][par][q] - again[key][par][q]) > 1e-9:
                        raise ValueError(f"{key} {par} {q} differs from its recomputation")
        meas = (loader or live_rows)()
    except Exception as e:
        print("VOID: input unreadable, hash differs or table does not match:", repr(e))
        print("VERDICT q1452: VOID")
        sys.exit(2)
    base = {K: dict(tab["A"][str(K)]) for K in TARGET}
    baseB = {K: dict(tab["B"][str(K)]) for K in TARGET}
    null = {K: dict(tab["null"][str(K)]) for K in TARGET}

    def shifted(b, K, q, d):
        m = {k: dict(v) for k, v in b.items()}
        m[K][q] += d
        return m

    barA = lambda K, q: tab["bar_A"]["even" if K % 2 == 0 else "odd"][q]
    barB = lambda K, q: tab["bar_B"]["even" if K % 2 == 0 else "odd"][q]
    cA = judge(tab, base)
    cB = judge(tab, baseB)
    cN = judge(tab, null)
    cC = judge(tab, shifted(base, 117, "PN", 3 * barA(117, "PN")))
    cD = judge(tab, shifted(baseB, 118, "Re", -3 * barB(118, "Re")))
    cE = judge(tab, {K: v for K, v in base.items() if K != 120})
    cF = judge(tab, shifted(base, 116, "Re", 0.9 * barA(116, "Re")))
    cG = judge(tab, shifted(base, 119, "PN", 1.0 - base[119]["PN"] - 1e-3))
    ok = (
        cA["A"][0] == "PASS"
        and cA["C"][0] == "PASS"
        and cB["B"][0] == "PASS"
        and cB["C"][0] == "PASS"
        and cC["A"][0] == "FAIL"
        and cD["B"][0] == "FAIL"
        and cE["A"][0] == "INCOMPLETE"
        and cF["A"][0] == "PASS"
        and cG["C"][0] == "FAIL"
    )
    print(
        "controls: rows at law A -> A and C PASS; rows at law B -> B and C PASS; rows at the exponent-1 null -> A",
        cN["A"][0],
        "and B",
        cN["B"][0],
        "(informative); one PN row 3 bars off law A -> A",
        cC["A"][0],
        "; one Re row 3 bars off law B -> B",
        cD["B"][0],
        "; row 120 missing -> A",
        cE["A"][0],
        "; one Re row 0.9 bar off -> A",
        cF["A"][0],
        "; one PN row below one -> C",
        cG["C"][0],
        ":",
        ok,
    )
    if not ok:
        print("VERDICT q1452: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        try:
            n = len([r for r in json.load(open(LIVE)) if r["K"] >= 115])
        except Exception:
            n = -1
        print("controls only; rows K >= 115 in the live file:", n)
        sys.exit(0)
    r = judge(tab, meas)
    for name, desc in (
        ("A", "law 2/3 within its bar"),
        ("B", "free-exponent law within its bar"),
        ("C", "PN and Re above one"),
    ):
        v, viol, miss = r[name]
        print(
            f"{name} ({desc}): {v}; measured {len(TARGET) - len(miss)} of {len(TARGET)}; violations {[(K, q, float('%.3g' % d)) for K, q, d in viol]}"
        )
    print("VERDICT q1452: " + "; ".join(f"{n} {r[n][0]}" for n in ("A", "B", "C")))
    sys.exit(0)


if __name__ == "__main__":
    main()
