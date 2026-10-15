"""q1462 (9 Oct 2026): VERDICT gate for the SERIES law of the margins in K^(-1/2), registered before any row K >= 115 exists (the rows are judged also by q1452; the two gates do not depend on each other).
READING that led to it (q1460, q1461, after the verdicts of q1446): 1/v_c^2 of the centre density scales like K^(-0.4987), so the natural correction variable of the births law is K^(-1/2), and the margins P/N and R_e of the rows K = 83 to 114 (32 rows, parities apart) are fitted to about 1e-9 (rms 4e-10 to 1e-10) by the three-term series L + c1 K^(-1/2) + c2 K^(-1) + c3 K^(-3/2), against 3e-7 to 1.6e-6 for the best one-exponent law. Its limits are L(P/N) = 1.0344 (even) and 1.0346 (odd), L(R_e) = 1.0541 and 1.0542, parities agreeing to 2e-4. An earlier statement of this project (amendments 2175 and 2177 and the docstring of q1446) that the exponent 2/3 is the variable of the births-law corrections was WRONG (erratum in amendment 2181).
Law C: per parity and per quantity P/N, R_e: least squares for L, c1, c2, c3 on all 16 training rows of the parity (snapshot q1452_training_rows.json, K = 83 to 114, sha256 REG_TRAIN, the same file as q1452). Predictions for K = 115 to 122. The bar of a sequence is max(1e-8, 2 x the largest hold-out error): fit on K <= 106, test on K = 107 to 114. The laws A (exponent 2/3) and B (free exponent) of the registered table q1452_margin_table.json (sha256 REG_Q1452) are the comparators. The table q1462_series_table.json (sha256 REG_TABLE) is recomputed from the snapshot (VOID if it differs by more than 1e-9).
Items for the measured rows K = 115 to 122 of small_gap_q206/q1416_rows.json (a row counts only with its own control met):
  C1: |PN - PN_C| < bar and |Re - Re_C| < bar for every measured row.
  C2: for every measured row and both quantities the error of law C is smaller than the errors of laws A and B.
Each item: FAIL if one measured row violates, INCOMPLETE if none violates and a row is missing, PASS otherwise. VOID (exit 2) if a hash differs, the table does not match, a row control failed, a file is unreadable or a control of the gate fails. Last line: VERDICT q1462: C1 ...; C2 ... Usage: py q1462_margin_series_verdict.py [--registration-run] (from temporal_within/centre_scripts/).
"""

import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(HERE, "q1452_training_rows.json")
Q1452 = os.path.join(HERE, "q1452_margin_table.json")
TABLE = os.path.join(HERE, "q1462_series_table.json")
LIVE = os.path.join(ROOT, "small_gap_q206", "q1416_rows.json")
REG_TRAIN = "b5c18276f692109003d34caccead13d2d0111c6662d0bc815f453c3a947a8864"
REG_Q1452 = "16ef0c20818c8209abef22de68378d489291a8cfecb150b96f575be59c3c17ba"
REG_TABLE = "e34e9a10e92fd86271b046781d6b9779c1c73e40a55cf36001c280d5167605e1"
EXPS = (0.5, 1.0, 1.5)
FLOOR = 1e-8
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


def fit(points):
    K = np.array([k for k, _ in points], float)
    y = np.array([v for _, v in points])
    M = np.vstack([np.ones_like(K)] + [K ** (-e) for e in EXPS]).T
    return np.linalg.lstsq(M, y, rcond=None)[0]


def ev(c, K):
    return float(c[0] + sum(ci * K ** (-e) for ci, e in zip(c[1:], EXPS)))


def build(rows):
    out = {"C": {}, "bar": {}, "limit": {}}
    for par in PARITY:
        out["bar"][par] = {}
        out["limit"][par] = {}
        for q in QUANT:
            s = sorted((r["K"], r[q]) for r in rows if r["parity"] == par)
            tr = [(k, v) for k, v in s if k <= 106]
            te = [(k, v) for k, v in s if k >= 107]
            c = fit(tr)
            hold = max(abs(ev(c, k) - v) for k, v in te)
            out["bar"][par][q] = max(FLOOR, 2 * hold)
            cf = fit(s)
            out["limit"][par][q] = float(cf[0])
            for K in TARGET:
                if (K % 2 == 0) == (par == "even"):
                    out["C"].setdefault(str(K), {})[q] = ev(cf, K)
    return out


def judge(tab, q1452, meas):
    res = {}
    for item in ("C1", "C2"):
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
                ec = abs(v - tab["C"][str(K)][q])
                if item == "C1" and not ec < tab["bar"][par][q]:
                    viol.append((K, q, v - tab["C"][str(K)][q]))
                if item == "C2" and not (
                    ec < abs(v - q1452["A"][str(K)][q]) and ec < abs(v - q1452["B"][str(K)][q])
                ):
                    viol.append((K, q, ec))
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


def main(loader=None, train=None, table=None, q1452=None):
    try:
        tj = load_hashed(train or TRAIN, REG_TRAIN)
        q = load_hashed(q1452 or Q1452, REG_Q1452)
        tab = load_hashed(table or TABLE, REG_TABLE)
        again = build(tj["rows"])
        for K in tab["C"]:
            for qq in QUANT:
                if abs(tab["C"][K][qq] - again["C"][K][qq]) > 1e-9:
                    raise ValueError(f"table C K = {K} {qq} differs from its recomputation")
        for par in PARITY:
            for qq in QUANT:
                if abs(tab["bar"][par][qq] - again["bar"][par][qq]) > 1e-9:
                    raise ValueError(f"bar {par} {qq} differs from its recomputation")
        meas = (loader or live_rows)()
    except Exception as e:
        print("VOID: input unreadable, hash differs or table does not match:", repr(e))
        print("VERDICT q1462: VOID")
        sys.exit(2)
    baseC = {K: dict(tab["C"][str(K)]) for K in TARGET}
    baseA = {K: dict(q["A"][str(K)]) for K in TARGET}
    baseB = {K: dict(q["B"][str(K)]) for K in TARGET}

    def shifted(b, K, qq, d):
        m = {k: dict(v) for k, v in b.items()}
        m[K][qq] += d
        return m

    bar = lambda K, qq: tab["bar"]["even" if K % 2 == 0 else "odd"][qq]
    cC = judge(tab, q, baseC)
    cA = judge(tab, q, baseA)
    cB = judge(tab, q, baseB)
    cX = judge(tab, q, shifted(baseC, 117, "PN", 3 * bar(117, "PN")))
    cM = judge(tab, q, {K: v for K, v in baseC.items() if K != 119})
    cH = judge(tab, q, shifted(baseC, 116, "Re", 0.9 * bar(116, "Re")))
    ok = (
        cC["C1"][0] == "PASS"
        and cC["C2"][0] == "PASS"
        and cA["C1"][0] == "FAIL"
        and cA["C2"][0] == "FAIL"
        and cB["C2"][0] == "FAIL"
        and cX["C1"][0] == "FAIL"
        and cM["C1"][0] == "INCOMPLETE"
        and cH["C1"][0] == "PASS"
    )
    print(
        "controls: rows at law C -> C1, C2 PASS; rows at law A -> C1",
        cA["C1"][0],
        "and C2",
        cA["C2"][0],
        "; rows at law B -> C2",
        cB["C2"][0],
        "; one PN row 3 bars off law C -> C1",
        cX["C1"][0],
        "; row 119 missing -> C1",
        cM["C1"][0],
        "; one Re row 0.9 bar off -> C1",
        cH["C1"][0],
        ":",
        ok,
    )
    if not ok:
        print("VERDICT q1462: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        try:
            n = len([r for r in json.load(open(LIVE)) if r["K"] >= 115])
        except Exception:
            n = -1
        print("controls only; rows K >= 115 in the live file:", n)
        sys.exit(0)
    r = judge(tab, q, meas)
    for name, desc in (("C1", "series law C within its bar"), ("C2", "law C closer than laws A and B")):
        v, viol, miss = r[name]
        print(
            f"{name} ({desc}): {v}; measured {len(TARGET) - len(miss)} of {len(TARGET)}; violations {[(K, qq, float('%.3g' % d)) for K, qq, d in viol]}"
        )
    print("VERDICT q1462: " + "; ".join(f"{n} {r[n][0]}" for n in ("C1", "C2")))
    sys.exit(0)


if __name__ == "__main__":
    main()
