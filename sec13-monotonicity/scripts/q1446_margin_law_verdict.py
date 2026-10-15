"""q1446 (9 Oct 2026): VERDICT gate for the margin law of the innermost-atom flow (track B of PROOF_TRACKS_2026-10-09.md).
READING that led to it (post hoc, rows K = 83 to 108 of small_gap_q206/q1416_rows.json, parities apart): the margin sequences P/N and R_e follow L + a K^-p; a free exponent gives p = 0.64 to 0.72 in all four sequences, and the fixed exponent 2/3 predicts the held-out rows K = 97 to 108 (fit on K <= 96) better than the exponents 1/3, 1 and 4/3. The exponent 2/3 is the variable of the births-law corrections (1/v^2) and was chosen AFTER seeing the rows K <= 108; the test below is blind because no row K >= 109 existed when the table was registered.
Registered: the training snapshot q1446_training_rows.json (rows K = 83 to 108, keys K, parity, PN, Re) and the table q1446_margin_table.json (two-parameter least-squares fit a0 + a1 K^-(2/3) per parity and per quantity on all training rows; predictions for K = 109 to 114; null predictions with the exponent 1; bar = 2 x the largest hold-out error of the 2/3 fit trained on K <= 96 and tested on K = 97 to 108). Both files are hash checked, and the table is recomputed from the snapshot (VOID if it differs by more than 1e-12).
Items, for the measured rows K = 109 to 114 of the live file (a row counts only with control_ok true; K odd = odd family):
  M1 (law): |PN - PN_pred| < bar and |Re - Re_pred| < bar of its parity, for every measured row.
  M2 (margin persists): PN > 1 and Re > 1 for every measured row.
  M3 (against the null): |error of the 2/3 prediction| < |error of the exponent-1 prediction| for PN and Re, for every measured row.
Each item: FAIL if one measured row violates, INCOMPLETE if none violates and a row of 109 to 114 is missing, PASS otherwise. VOID (exit 2) if a hash differs, the table does not match its recomputation, a row's own control failed, a file is unreadable or a control of the gate fails. The last line is VERDICT q1446: M1 ...; M2 ...; M3 ... Usage: py q1446_margin_law_verdict.py [--registration-run] (from temporal_within/centre_scripts/).
"""

import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(HERE, "q1446_training_rows.json")
TABLE = os.path.join(HERE, "q1446_margin_table.json")
LIVE = os.path.join(ROOT, "small_gap_q206", "q1416_rows.json")
REG_TRAIN = "09598e00d779c636813e1d285b8088baccb1e149aa68c63737f7401ad8fda557"
REG_TABLE = "216406aca83b33634457c677619c25f0e6d315328c730a2374719776cfff5fd0"
P_LAW = 2.0 / 3.0
P_NULL = 1.0
TARGET = list(range(109, 115))
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


def fit(points, p):
    K = np.array([k for k, _ in points], float)
    y = np.array([v for _, v in points])
    M = np.vstack([np.ones_like(K), K ** (-p)]).T
    return np.linalg.lstsq(M, y, rcond=None)[0]


def build(rows):
    """predictions, null predictions and bars from the training rows (list of dicts K, parity, PN, Re)"""
    out = {"pred": {}, "null": {}, "bar": {}, "holdout": {}}
    for par in PARITY:
        out["bar"][par] = {}
        out["holdout"][par] = {}
        for q in QUANT:
            s = sorted((r["K"], r[q]) for r in rows if r["parity"] == par)
            tr = [(k, v) for k, v in s if k <= 96]
            te = [(k, v) for k, v in s if k >= 97]
            c = fit(tr, P_LAW)
            e = max(abs(c[0] + c[1] * k ** (-P_LAW) - v) for k, v in te)
            out["holdout"][par][q] = e
            out["bar"][par][q] = 2 * e
            cf = fit(s, P_LAW)
            cn = fit(s, P_NULL)
            for K in TARGET:
                if (K % 2 == 0) == (par == "even"):
                    out["pred"].setdefault(str(K), {})[q] = float(cf[0] + cf[1] * K ** (-P_LAW))
                    out["null"].setdefault(str(K), {})[q] = float(cn[0] + cn[1] * K ** (-P_NULL))
    return out


def judge(tab, meas):
    """meas: dict K -> {'PN':, 'Re':} or None; returns {item: (verdict, violations, missing)}"""
    res = {}
    for item in ("M1", "M2", "M3"):
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
                pr = tab["pred"][str(K)][q]
                nu = tab["null"][str(K)][q]
                if item == "M1" and not abs(v - pr) < tab["bar"][par][q]:
                    viol.append((K, q, v - pr))
                if item == "M2" and not v > 1.0:
                    viol.append((K, q, v))
                if item == "M3" and not abs(v - pr) < abs(v - nu):
                    viol.append((K, q, abs(v - pr) - abs(v - nu)))
        res[item] = ("FAIL" if viol else ("INCOMPLETE" if miss else "PASS"), viol, miss)
    return res


def live_rows():
    rows = json.load(open(LIVE))
    meas = {}
    for r in rows:
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
        for key in ("pred", "null"):
            for K in tab[key]:
                for q in QUANT:
                    if abs(tab[key][K][q] - again[key][K][q]) > 1e-12:
                        raise ValueError(f"table {key} K = {K} {q} differs from its recomputation")
        for par in PARITY:
            for q in QUANT:
                if abs(tab["bar"][par][q] - again["bar"][par][q]) > 1e-12:
                    raise ValueError(f"bar {par} {q} differs from its recomputation")
        meas = (loader or live_rows)()
    except Exception as e:
        print("VOID: input unreadable, hash differs or table does not match:", repr(e))
        print("VERDICT q1446: VOID")
        sys.exit(2)
    base = {K: dict(tab["pred"][str(K)]) for K in TARGET}

    def shifted(K, q, d):
        m = {k: dict(v) for k, v in base.items()}
        m[K][q] += d
        return m

    bar = lambda K, q: tab["bar"]["even" if K % 2 == 0 else "odd"][q]
    cA = judge(tab, base)
    cB = judge(tab, {K: dict(tab["null"][str(K)]) for K in TARGET})
    cC = judge(tab, shifted(111, "PN", 3 * bar(111, "PN")))
    cD = judge(tab, {K: v for K, v in base.items() if K != 112})
    cE = judge(tab, shifted(110, "Re", 0.9 * bar(110, "Re")))
    cF = judge(tab, shifted(113, "PN", 1.0 - base[113]["PN"] - 1e-3))
    ok = (
        all(v[0] == "PASS" for v in cA.values())
        and cB["M1"][0] == "FAIL"
        and cB["M3"][0] == "FAIL"
        and cC["M1"][0] == "FAIL"
        and cD["M1"][0] == "INCOMPLETE"
        and cE["M1"][0] == "PASS"
        and cF["M2"][0] == "FAIL"
    )
    print(
        "controls: rows at the table -> M1, M2, M3 PASS; rows at the exponent-1 null -> M1",
        cB["M1"][0],
        "and M3",
        cB["M3"][0],
        "; one PN row 3 bars off -> M1",
        cC["M1"][0],
        "; row 112 missing -> M1",
        cD["M1"][0],
        "; one Re row 0.9 bar off -> M1",
        cE["M1"][0],
        "; one PN row below 1 -> M2",
        cF["M2"][0],
        ":",
        ok,
    )
    if not ok:
        print("VERDICT q1446: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        try:
            n = len([r for r in json.load(open(LIVE)) if r["K"] >= 109])
        except Exception:
            n = -1
        print("controls only; rows K >= 109 in the live file:", n)
        sys.exit(0)
    r = judge(tab, meas)
    for name, desc in (
        ("M1", "law, bar 2 x hold-out error"),
        ("M2", "PN and Re above one"),
        ("M3", "2/3 prediction closer than the exponent-1 null"),
    ):
        v, viol, miss = r[name]
        print(
            f"{name} ({desc}): {v}; measured {len(TARGET) - len(miss)} of {len(TARGET)}; violations {[(K, q, float('%.3g' % d)) for K, q, d in viol]}"
        )
    print("VERDICT q1446: " + "; ".join(f"{n} {r[n][0]}" for n in ("M1", "M2", "M3")))
    sys.exit(0)


if __name__ == "__main__":
    main()
