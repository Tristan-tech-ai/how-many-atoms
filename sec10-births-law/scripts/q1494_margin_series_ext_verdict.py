"""q1494 (9 Oct 2026): VERDICT gate for the SERIES law of the margins in K^(-1/2) on the rows K = 123 to 130, registered before any row K >= 123 exists (extension of q1462, which judged the rows K = 115 to 122 and passed, amendment 2189).
Law C: per parity and per quantity P/N, R_e: least squares for L, c1, c2, c3 on all 20 training rows of the parity (snapshot q1494_training_rows.json, K = 83 to 122, sha256 REG_TRAIN; the rows 115 to 122 were blind rows of q1462 and are training rows here, which is the only change of the fit) with the terms 1, K^(-1/2), K^(-1), K^(-3/2). Predictions for K = 123 to 130.
Bar of a sequence: max(1e-8, 2 x the largest hold-out error of the law fitted on K <= 114 against the rows K >= 115 of the same parity) - the same rule as q1462, now with 8 blind rows as the hold-out set.
Items for the measured rows K = 123 to 130 of small_gap_q206/q1416_rows.json (a row counts only with its own control met):
  C1: |PN - PN_C| < bar and |Re - Re_C| < bar for every measured row.
  C2: for every measured row and both quantities the error of law C is smaller than the error of the PERSISTENCE null (the value of the row K - 2 of the same parity, measured when it is a row of this block, else the training row).
Each item: FAIL if one measured row violates, INCOMPLETE if none violates and a row (or a persistence row it needs) is missing, PASS otherwise. VOID (exit 2) if a hash differs, the table does not match, a row control failed, a file is unreadable or a control of the gate fails. Last line: VERDICT q1494: C1 ...; C2 ... Usage: py q1494_margin_series_ext_verdict.py [--registration-run] (from centre_scripts/).
"""

import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(HERE, "q1494_training_rows.json")
TABLE = os.path.join(HERE, "q1494_series_table.json")
LIVE = os.path.join(ROOT, "small_gap_q206", "q1416_rows.json")
REG_TRAIN = "d6a03bbc7353eaf91ddec8dfcaec4286e2122151508ded3dac4b5f4ed176b025"
REG_TABLE = "6b653d8c367861bd0f5fcb92d557d004fd776bf6e2f2f946fa14e7ebadf74760"
EXPS = (0.5, 1.0, 1.5)
FLOOR = 1e-8
TARGET = list(range(123, 131))
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
    out = {"C": {}, "bar": {}, "limit": {}, "last": {}}
    for par in PARITY:
        out["bar"][par] = {}
        out["limit"][par] = {}
        out["last"][par] = {}
        for q in QUANT:
            s = sorted((r["K"], r[q]) for r in rows if r["parity"] == par)
            tr = [(k, v) for k, v in s if k <= 114]
            te = [(k, v) for k, v in s if k >= 115]
            c = fit(tr)
            hold = max(abs(ev(c, k) - v) for k, v in te)
            out["bar"][par][q] = max(FLOOR, 2 * hold)
            cf = fit(s)
            out["limit"][par][q] = float(cf[0])
            out["last"][par][q] = {str(k): v for k, v in s if k >= 121}
            for K in TARGET:
                if (K % 2 == 0) == (par == "even"):
                    out["C"].setdefault(str(K), {})[q] = ev(cf, K)
    return out


def persistence(tab, meas, K, q):
    """value of the row K - 2 of the same parity: a measured row of this block, else the training row; None when the needed measured row is missing"""
    p = K - 2
    if p in TARGET:
        return None if meas.get(p) is None else meas[p][q]
    return tab["last"]["even" if K % 2 == 0 else "odd"][q][str(p)]


def judge(tab, meas):
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
                if item == "C2":
                    pv = persistence(tab, meas, K, q)
                    if pv is None:
                        miss.append(K)
                        continue
                    if not ec < abs(v - pv):
                        viol.append((K, q, ec))
        res[item] = ("FAIL" if viol else ("INCOMPLETE" if miss else "PASS"), viol, sorted(set(miss)))
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
        print("VERDICT q1494: VOID")
        sys.exit(2)
    baseC = {K: dict(tab["C"][str(K)]) for K in TARGET}
    basePers = {}
    for K in TARGET:
        basePers[K] = {
            qq: persistence(tab, basePers, K, qq) for qq in QUANT
        }  # the persistence null as a set of rows (rows K - 2 of this block are the null rows themselves)

    def shifted(b, K, qq, d):
        m = {k: dict(v) for k, v in b.items()}
        m[K][qq] += d
        return m

    bar = lambda K, qq: tab["bar"]["even" if K % 2 == 0 else "odd"][qq]
    cC = judge(tab, baseC)
    cP = judge(tab, basePers)
    cX = judge(tab, shifted(baseC, 125, "PN", 3 * bar(125, "PN")))
    cM = judge(tab, {K: v for K, v in baseC.items() if K != 127})
    cH = judge(tab, shifted(baseC, 124, "Re", 0.9 * bar(124, "Re")))
    cE = judge(tab, {})
    ok = (
        cC["C1"][0] == "PASS"
        and cC["C2"][0] == "PASS"
        and cP["C2"][0] == "FAIL"
        and cX["C1"][0] == "FAIL"
        and cM["C1"][0] == "INCOMPLETE"
        and cM["C2"][0] == "INCOMPLETE"
        and cH["C1"][0] == "PASS"
        and cH["C2"][0] == "PASS"
        and cE["C1"][0] == "INCOMPLETE"
    )
    print(
        "controls: rows at law C -> C1",
        cC["C1"][0],
        "C2",
        cC["C2"][0],
        "; rows at the persistence null -> C2",
        cP["C2"][0],
        "; one PN row 3 bars off -> C1",
        cX["C1"][0],
        "; row 127 missing -> C1",
        cM["C1"][0],
        "C2",
        cM["C2"][0],
        "; one Re row 0.9 bar off -> C1",
        cH["C1"][0],
        "C2",
        cH["C2"][0],
        "; no row -> C1",
        cE["C1"][0],
        ":",
        ok,
    )
    if not ok:
        print("VERDICT q1494: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        try:
            n = len([r for r in json.load(open(LIVE)) if r["K"] >= 123])
        except Exception:
            n = -1
        print("controls only; rows K >= 123 in the live file:", n)
        sys.exit(0)
    r = judge(tab, meas)
    for name, desc in (
        ("C1", "series law C within its bar"),
        ("C2", "law C closer than the persistence null"),
    ):
        v, viol, miss = r[name]
        print(
            f"{name} ({desc}): {v}; measured {len(TARGET) - len([k for k in miss if meas.get(k) is None])} of {len(TARGET)}; violations {[(K, qq, float('%.3g' % d)) for K, qq, d in viol]}"
        )
    print("VERDICT q1494: " + "; ".join(f"{n} {r[n][0]}" for n in ("C1", "C2")))
    sys.exit(0)


if __name__ == "__main__":
    main()
