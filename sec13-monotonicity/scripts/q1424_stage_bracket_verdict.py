"""q1424 (9 Oct 2026): VERDICT gate for two statements about the stage depth of the half-line stage behind the derived-series table (amendment 1968, D1). Written AFTER the exploratory readings q1423_explore_primitives.py and q1423_explore_stage.py had shown (reading, 9 Oct 10:2x) that the D1 deviations of K_b 81 to 99 move with the stage depth D; the table of five stages (centre_scripts/q1424_stage_table.json, sha256 below) was computed without any measured amplitude (q1424_stage_predict.py).
SL (stage limit, POST-HOC on events already measured, K_b 93 to 99): for every K_b of the range, max over D in {98, 104, 116, 122} of |pred_D - pred_110| >= 5 x |A_meas - pred_110|, i.e. the change of the prediction with the stage depth is at least five times the D1 deviation of the registered stage D = 110. PASS if all seven hold, FAIL if one does not, INCOMPLETE if a measured amplitude is missing.
BR1 (bracket, BLIND, K_b 100 to 114 whose amplitudes are not measured when this is written): min(pred_110, pred_122) <= A_meas <= max(pred_110, pred_122). FAIL if one event lies outside, INCOMPLETE if none violates and one is missing, PASS if all fifteen lie inside.
VOID (exit 2): a table hash differs from the registered one (stage table REG4 below, q1209 table f48c3ea9...), the stage 110 of the stage table differs from the q1209 table by 1e-12 or more, a file cannot be read, or the internal controls fail. The measured amplitude is the one of q1422.measured. Last line: VERDICT q1424: SL ...; BR1 ... Usage: py q1424_stage_bracket_verdict.py [--registration-run]  (from temporal_within/centre_scripts/).
"""

import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
T4 = os.path.join(HERE, "q1424_stage_table.json")
T2 = os.path.join(HERE, "q1209_predictions.json")
REG4 = "bf2c8e291c533164469cd8e31e9c40a8eca211ddd9af2274ac34a36a2d3e965d"
REG2 = "f48c3ea92d5bd7978c8d1400dd04136490bcbe08a42396afd03d75a523f4fc0d"
SL_RANGE = range(93, 100)
BR_RANGE = range(100, 115)
SL_FACTOR = 5.0
OTHER = ("98", "104", "116", "122")


def load(path, reg):
    js = json.load(open(path))
    sha = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if not (h == reg and sha == reg):
        raise ValueError(f"hash of {os.path.basename(path)} is {h[:16]}..., registered {reg[:16]}...")
    return js


def judge(st, meas):
    p110 = st["110"]
    p122 = st["122"]
    viol_sl = []
    miss_sl = []
    viol_br = []
    miss_br = []
    for K in SL_RANGE:
        m = meas.get(K)
        if m is None:
            miss_sl.append(K)
            continue
        shift = max(abs(st[d][K] - p110[K]) for d in OTHER)
        if not shift >= SL_FACTOR * abs(m - p110[K]):
            viol_sl.append((K, shift, m - p110[K]))
    for K in BR_RANGE:
        m = meas.get(K)
        if m is None:
            miss_br.append(K)
            continue
        if not (min(p110[K], p122[K]) <= m <= max(p110[K], p122[K])):
            viol_br.append((K, m - p110[K], m - p122[K]))
    sl = "FAIL" if viol_sl else ("INCOMPLETE" if miss_sl else "PASS")
    br = "FAIL" if viol_br else ("INCOMPLETE" if miss_br else "PASS")
    return sl, br, viol_sl, viol_br, miss_sl, miss_br


def main(loader=None):
    try:
        import q1422_blind_items_verdict as G22

        raw = load(T4, REG4)["stages"]
        st = {d: {int(k): float(v) for k, v in raw[d].items()} for d in raw}
        reg2 = {int(k): float(v) for k, v in load(T2, REG2)["capacity"].items()}
        meas = {K: (loader or G22.measured)(K) for K in range(93, 115)}
    except Exception as e:
        print("VOID: input unreadable or table hash differs:", repr(e))
        print("VERDICT q1424: VOID")
        sys.exit(2)
    if not max(abs(st["110"][K] - reg2[K]) for K in reg2) < 1e-12:
        print("VOID: stage 110 differs from the registered table")
        print("VERDICT q1424: VOID")
        sys.exit(2)
    r = judge(st, meas)
    p110, p122 = st["110"], st["122"]
    cA = judge(st, {K: p110[K] for K in range(93, 115)})[:2]  # events at the registered stage: both hold
    cB = judge(st, {K: 0.5 * (p110[K] + p122[K]) for K in range(93, 115)})[
        :2
    ]  # events halfway between the two stages: BR1 holds; SL holds only if the shift is five times the half distance
    top = max(p110[105], p122[105]) + 1e-9
    cC = judge(st, {**{K: p110[K] for K in range(93, 115)}, 105: top})[1]  # one event outside the bracket
    cD = judge(st, {K: p110[K] for K in range(93, 115) if K != 105})[1]  # one event missing
    far = {K: p110[K] for K in range(93, 115)}
    far[96] = p110[96] + 1e-9  # one SL event with a deviation far above the stage shift
    cE = judge(st, far)[0]
    ok = (cA, cC, cD, cE) == (("PASS", "PASS"), "FAIL", "INCOMPLETE", "FAIL") and cB[1] == "PASS"
    print(
        "controls: events at the registered stage ->",
        cA,
        "; one event outside the bracket ->",
        cC,
        "; one event missing ->",
        cD,
        "; one SL event 1e-9 off ->",
        cE,
        "; halfway events BR1 ->",
        cB[1],
        ":",
        ok,
    )
    if not ok:
        print("VERDICT q1424: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("controls only")
        sys.exit(0)
    sl, br, vsl, vbr, msl, mbr = r
    print(
        f"SL (K_b 93 to 99, factor {SL_FACTOR:g}): {sl}; measured {7 - len(msl)} of 7; ratios stage shift / D1 deviation "
        + ", ".join(
            f"{K} {max(abs(st[d][K] - p110[K]) for d in OTHER) / max(abs(meas[K] - p110[K]), 1e-300):.1f}"
            for K in SL_RANGE
            if meas.get(K) is not None
        )
    )
    print(
        f"BR1 (K_b 100 to 114): {br}; measured {15 - len(mbr)} of 15; violations {[(K, round(a * 1e9, 4), round(b * 1e9, 4)) for K, a, b in vbr]}"
    )
    print(f"VERDICT q1424: SL {sl}; BR1 {br}")
    sys.exit(0)


if __name__ == "__main__":
    main()
