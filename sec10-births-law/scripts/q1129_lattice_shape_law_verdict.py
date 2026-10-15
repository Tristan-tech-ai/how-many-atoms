"""q1129 (7 Oct 2026): verdict script of T128 (registered in the amendment that follows 1862): the shape of the last state of a capacity family, p_j / h for j = 1..6 with h = p_2 - p_1 (positions of the optimal input, u_1 < u_2 < ...), follows the law fitted on the immune and NPMLE chains only (reading q1128): for each parity y = alpha + beta ln K + gamma / K, fitted by least squares on the completed chain families whose last row lies within 0.03 of the event (the 27 families of q1128; the selection and the features are those of q1128).
Items: S1 for each capacity family K = 74, 75, 76, 77, 78, 79 and each j = 1..6, |p_j/h measured - model| / model <= TOL, with TOL = 0.0020 for even K and 0.0070 for odd K. The last state of a family is the last line of scratch/tower_family_list_K<K>.txt (the file is written after the family is complete; a missing file counts as a missing family).
Verdict (last line): PASS if all six families are present and S1 holds for all 36 items; FAIL if a measured item fails; INCOMPLETE if no measured item fails and a family is missing; VOID if a control misbehaves, the chain fit has fewer than 20 families or a list has fewer than 8 states.
Controls (no file of the families under test is read in the registration run): PC synthetic p_j/h = model for all six families passes; NC1 K 74 at j = 6 plus 0.004 fails; NC2 K 77 at j = 5 plus 0.010 fails; NC3 K 79 missing is INCOMPLETE; PC2 the chain model reproduces the measured capacity families K 72 (j = 1..6 within 0.0010) and K 73 (within 0.0050), which are existing data and not items.
Usage: py q1129_lattice_shape_law_verdict.py [--registration-run]"""

import json, os, sys
import numpy as np

ARCH = "scratch"
KS = (74, 75, 76, 77, 78, 79)
NJ = 6
TOLS = {0: 0.0020, 1: 0.0070}


def families(rows):
    fam = []
    cur = [rows[0]]
    for r in rows[1:]:
        if r["K"] == cur[-1]["K"] and r["centre"] == cur[-1]["centre"]:
            cur.append(r)
        else:
            fam.append(cur)
            cur = [r]
    fam.append(cur)
    return fam


def shape(p):
    h = p[1] - p[0]
    return [p[j] / h for j in range(NJ)]


def fit_model():
    ch = []
    for fn in ("results_q584_ext.json", "results_q601_ext.json"):
        d = json.load(open(fn))
        ev = {}
        for e in d["events"]:
            if e["kind"] in ("centre appear", "centre split"):
                ev.setdefault(e["K_before"], e["a_lo"])
        for c in families(d["rows"])[:-1]:
            K = c[-1]["K"]
            if K not in ev or ev[K] - c[-1]["a"] > 0.03:
                continue
            ch.append((K, shape(c[-1]["p"])))
    if len(ch) < 20:
        print("VERDICT q1129 T128: VOID (fewer than 20 chain families)")
        sys.exit(2)
    coef = {}
    for par in (0, 1):
        C = [r for r in ch if r[0] % 2 == par]
        K = np.array([r[0] for r in C], float)
        M = np.vstack([np.ones_like(K), np.log(K), 1 / K]).T
        coef[par] = [np.linalg.lstsq(M, np.array([r[1][j] for r in C]), rcond=None)[0] for j in range(NJ)]
    return coef, len(ch)


def predict(coef, K):
    return [float(coef[K % 2][j] @ np.array([1.0, np.log(K), 1.0 / K])) for j in range(NJ)]


def last_shape(K):
    lp = f"{ARCH}/tower_family_list_K{K}.txt"
    if not os.path.exists(lp):
        return None
    lines = [l.split() for l in open(lp) if l.strip()]
    if len(lines) < 8:
        print(f"VERDICT q1129 T128: VOID (K {K}: list has {len(lines)} states)")
        sys.exit(2)
    u = [float(x) for x in json.load(open(lines[-1][1]))["u"]]
    return shape(u)


def verdict(meas, model):
    res = {}
    for K in KS:
        if meas.get(K) is None:
            res[K] = "MISSING"
            continue
        res[K] = (
            "PASS"
            if all(abs(meas[K][j] - model[K][j]) / model[K][j] <= TOLS[K % 2] for j in range(NJ))
            else "FAIL"
        )
    s = list(res.values())
    return ("FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")), res


def main():
    reg = "--registration-run" in sys.argv
    coef, n = fit_model()
    model = {K: predict(coef, K) for K in KS}
    good = {K: list(model[K]) for K in KS}
    bad1 = {**good, 74: [*good[74][:5], good[74][5] * 1.004]}
    bad2 = {**good, 77: [*good[77][:4], good[77][4] * 1.010, good[77][5]]}
    c = {
        "PC": verdict(good, model)[0] == "PASS",
        "NC1": verdict(bad1, model)[0] == "FAIL",
        "NC2": verdict(bad2, model)[0] == "FAIL",
        "NC3": verdict({**good, 79: None}, model)[0] == "INCOMPLETE",
    }
    ok2 = True
    for K, tol in ((72, 0.0010), (73, 0.0050)):
        s = last_shape(K)
        if s is None:
            ok2 = False
            continue
        ok2 = ok2 and all(abs(s[j] - predict(coef, K)[j]) / predict(coef, K)[j] <= tol for j in range(NJ))
    c["PC2"] = ok2
    print(f"chain families in the fit: {n}; controls:", c)
    if not all(c.values()):
        print("VERDICT q1129 T128: VOID (a control does not behave)")
        sys.exit(2)
    if reg:
        print("VERDICT q1129 T128: INCOMPLETE (controls run only)")
        sys.exit(3)
    meas = {K: last_shape(K) for K in KS}
    v, res = verdict(meas, model)
    for K in KS:
        if meas[K] is None:
            print(f"   K {K}: missing -> MISSING")
            continue
        print(
            f"   K {K}: p_j/h measured "
            + " ".join(f"{x:.5f}" for x in meas[K])
            + "; relative differences "
            + " ".join(f"{(meas[K][j] - model[K][j]) / model[K][j]:+.5f}" for j in range(NJ))
            + f" (tolerance {TOLS[K % 2]}) -> {res[K]}"
        )
    print(f"VERDICT q1129 T128: {v}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
