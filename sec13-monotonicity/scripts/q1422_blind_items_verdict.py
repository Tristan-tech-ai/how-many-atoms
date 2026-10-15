"""q1422 (9 Oct 2026): VERDICT gate for the registered blind items on the capacity events above K_b 80. Written after the reminder (9 Oct) that a success or a failure must come from a pure-code gate with a selftest: the registered readers q1204 and q1215 print booleans and are labelled 'no verdict' by their registration, so they cannot carry the claim 'item S1 failed'. This gate takes the SAME criteria, registered earlier and not changed: S1 (amendment 1959): |A_meas - a_pred| < 0.5e-9 for K_b 81 to 99 against the table q1201_table.json (sha256 fcf88352217a8017344239510feeb38f1318f2d7751a763c7bfaf21ba844aaac); S2 (1959): < 2e-9 for K_b 100 to 114, same table; D1 (amendment 1968): |A_meas - a_pred| < 0.3e-9 for K_b 81 to 99 and < 1e-9 for K_b 100 to 114 against the order-12 table q1209_predictions.json (sha256 f48c3ea92d5bd7978c8d1400dd04136490bcbe08a42396afd03d75a523f4fc0d). The measured amplitude of K_b is A_cross (even K_b, file tower_K<K_b+1>_birth_out.json) or A_s (odd K_b, tower_K<K_b+1>_As.json) in scratch, as read by q1183.measured.
Rule per item: FAIL if any measured event of its range violates the bar (the first violator is named); otherwise INCOMPLETE if an event of the range has no measured amplitude; otherwise PASS. VOID (exit 2) if a table hash differs from the registered one or a file cannot be read. The last line is VERDICT q1422: S1 ...; S2 ...; D1 ... Usage: py q1422_blind_items_verdict.py [--registration-run]  (from temporal_within/centre_scripts/).
"""

import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(os.path.dirname(HERE), "scratch")
T1 = os.path.join(HERE, "q1201_table.json")
T2 = os.path.join(HERE, "q1209_predictions.json")
REG1 = "fcf88352217a8017344239510feeb38f1318f2d7751a763c7bfaf21ba844aaac"
REG2 = "f48c3ea92d5bd7978c8d1400dd04136490bcbe08a42396afd03d75a523f4fc0d"
ITEMS = [
    ("S1", range(81, 100), 0.5e-9, "t1"),
    ("S2", range(100, 115), 2e-9, "t1"),
    ("D1", range(81, 100), 0.3e-9, "t2"),
    ("D1b", range(100, 115), 1e-9, "t2"),
]


def load_table(path, reg, key):
    js = json.load(open(path))
    sha = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if not (h == reg and sha == reg):
        raise ValueError(f"hash of {os.path.basename(path)} is {h[:16]}..., registered {reg[:16]}...")
    return {int(k): float(v) for k, v in (js["a_pred"] if key == "a_pred" else js["capacity"]).items()}


def measured(K):
    f = os.path.join(ARCH, f"tower_K{K + 1}_birth_out.json" if K % 2 == 0 else f"tower_K{K + 1}_As.json")
    if not os.path.exists(f):
        return None
    return float(json.load(open(f))["A_cross" if K % 2 == 0 else "A_s"])


def judge(pred1, pred2, meas):
    out = {}
    for name, rng, bar, tab in ITEMS:
        pred = pred1 if tab == "t1" else pred2
        viol = []
        miss = []
        for K in rng:
            m = meas.get(K)
            if m is None:
                miss.append(K)
                continue
            if not abs(m - pred[K]) < bar:
                viol.append((K, m - pred[K]))
        out[name] = (
            ("FAIL", viol, miss) if viol else (("INCOMPLETE", viol, miss) if miss else ("PASS", viol, miss))
        )
    return out


def main(loader=None):
    try:
        p1 = load_table(T1, REG1, "a_pred")
        p2 = load_table(T2, REG2, "capacity")
        meas = {K: (loader or measured)(K) for K in range(81, 115)}
    except Exception as e:
        print("VOID: input unreadable or table hash differs:", repr(e))
        print("VERDICT q1422: VOID")
        sys.exit(2)
    r = judge(p1, p2, {K: v for K, v in meas.items()})
    cA = judge(p1, p2, {K: p1[K] for K in p1})
    cB = judge(p1, p2, {K: p2[K] for K in p2})
    cC = judge(p1, p2, {K: (p1[K] + 0.6e-9 if K == 90 else p1[K]) for K in p1})
    cD = judge(p1, p2, {K: (None if K == 90 else p1[K]) for K in p1})
    ok = (cA["S1"][0], cA["S2"][0], cB["D1"][0], cB["D1b"][0], cC["S1"][0], cD["S1"][0]) == (
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "FAIL",
        "INCOMPLETE",
    )
    print(
        "controls: events at the 1959 table -> S1, S2",
        cA["S1"][0],
        cA["S2"][0],
        "; events at the order-12 table -> D1",
        cB["D1"][0],
        cB["D1b"][0],
        "; one S1 event 0.6e-9 off ->",
        cC["S1"][0],
        "; one event missing ->",
        cD["S1"][0],
        ":",
        ok,
    )
    if not ok:
        print("VERDICT q1422: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("controls only")
        sys.exit(0)
    for name, rng, bar, tab in ITEMS:
        v, viol, miss = r[name]
        nmeas = len(list(rng)) - len(miss)
        print(
            f"{name} (K_b {rng[0]} to {rng[-1]}, bar {bar * 1e9:g}e-9): {v}; measured {nmeas} of {len(list(rng))}; violations {[(K, round(d * 1e9, 4)) for K, d in viol]}"
        )
    s1, s2, d1, d1b = r["S1"][0], r["S2"][0], r["D1"][0], r["D1b"][0]
    d1v = "FAIL" if "FAIL" in (d1, d1b) else ("INCOMPLETE" if "INCOMPLETE" in (d1, d1b) else "PASS")
    print(f"VERDICT q1422: S1 {s1}; S2 {s2}; D1 {d1v}")
    sys.exit(0)


if __name__ == "__main__":
    main()
