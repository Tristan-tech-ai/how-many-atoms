"""q1527 (10 Oct 2026): VERDICT gate for the capacity events K_b 131 to 138 predicted from the stage D = 134 of the chain with the buffer B1 = 20 (table q1527_cap_b20_table.json, sha256 REG below: stages D = 122, 128 and 134 of centre_scripts/c50_capc_D140_B20.json, with the spread max(|122 - 134|, |128 - 134|) per K_b). Written after the verdict VERDICT q1451: B115 PASS; B123 PASS (16 blind events K_b 115 to 130). No capacity amplitude above K_b 130 existed or was read when the table was registered.
Item B131 (K_b 131 to 138), BLIND: |A_meas - a_pred(134)| <= max(0.05e-9, 2 x spread(K_b)). The floor 0.05e-9 is the one of q1439 and q1451. The spread grows from 7.4e-9 at K_b 131 to 2.6e-8 at K_b 138, so the stage, not the law, limits the test.
FAIL if one measured event violates, INCOMPLETE if none violates and one is not measured, PASS otherwise. Measured amplitude: A_event of small_gap_q206/fw_prod/event_Kb<K_b>.json. VOID (exit 2) if the hash differs, a file is unreadable or a control fails. Usage: py q1527_cap_b131_verdict.py [--registration-run] (from temporal_within/centre_scripts/).
"""

import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T = os.path.join(HERE, "q1527_cap_b20_table.json")
REG = "f2900ce1e315ac366cf74fabc081bf13faa83185408f77797a8d260623634476"
FLOOR = 0.05e-9
ITEMS = [("B131", range(131, 139))]
KALL = range(131, 139)


def load(path, reg):
    js = json.load(open(path))
    sha = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if not (h == reg and sha == reg):
        raise ValueError(f"hash of {os.path.basename(path)} is {h[:16]}..., registered {reg[:16]}...")
    return {int(k): float(v) for k, v in js["stages"]["134"].items()}, {
        int(k): float(v) for k, v in js["spread"].items()
    }


def measured(K):
    f = os.path.join(ROOT, "small_gap_q206", "fw_prod", f"event_Kb{K}.json")
    return float(json.load(open(f))["A_event"]) if os.path.exists(f) else None


def judge(pred, spread, meas):
    out = {}
    for name, rng in ITEMS:
        viol = []
        miss = []
        for K in rng:
            m = meas.get(K)
            if m is None:
                miss.append(K)
                continue
            if not abs(m - pred[K]) <= max(FLOOR, 2 * spread[K]):
                viol.append((K, m - pred[K]))
        out[name] = ("FAIL" if viol else ("INCOMPLETE" if miss else "PASS"), viol, miss)
    return out


def main(loader=None, table=None):
    try:
        pred, spread = load(table or T, REG)
        meas = {K: (loader or measured)(K) for K in KALL}
    except Exception as e:
        print("VOID: input unreadable or table hash differs:", repr(e))
        print("VERDICT q1527: VOID")
        sys.exit(2)
    base = {K: pred[K] for K in KALL}
    bar = lambda K: max(FLOOR, 2 * spread[K])
    cA = judge(pred, spread, base)
    cB = judge(pred, spread, {**base, 134: pred[134] + 3 * bar(134)})
    cC = judge(pred, spread, {**base, 137: pred[137] - 3 * bar(137)})
    cD = judge(pred, spread, {K: v for K, v in base.items() if K != 135})
    cE = judge(pred, spread, {**base, 133: pred[133] + 0.9 * bar(133)})
    ok = (
        all(v[0] == "PASS" for v in cA.values())
        and cB["B131"][0] == "FAIL"
        and cC["B131"][0] == "FAIL"
        and cD["B131"][0] == "INCOMPLETE"
        and cE["B131"][0] == "PASS"
    )
    print(
        "controls: events at the table -> PASS; K_b 134 3 bars off ->",
        cB["B131"][0],
        "; K_b 137 3 bars off the other way ->",
        cC["B131"][0],
        "; K_b 135 missing ->",
        cD["B131"][0],
        "; one event 0.9 bar off ->",
        cE["B131"][0],
        ":",
        ok,
    )
    if not ok:
        print("VERDICT q1527: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        n = sum(1 for K in KALL if measured(K) is not None)
        print("controls only; events K_b >= 131 in fw_prod:", n)
        sys.exit(0)
    r = judge(pred, spread, meas)
    for name, rng in ITEMS:
        v, viol, miss = r[name]
        print(
            f"{name} (blind, K_b {rng[0]} to {rng[-1]}): {v}; measured {len(rng) - len(miss)} of {len(rng)}; largest |dev| "
            + (
                "%.4fe-9" % (max(abs(meas[K] - pred[K]) for K in rng if meas.get(K) is not None) * 1e9)
                if len(miss) < len(rng)
                else "none"
            )
            + "; violations "
            + str([(K, round(d * 1e9, 4)) for K, d in viol])
        )
    print("VERDICT q1527: " + "; ".join(f"{n} {r[n][0]}" for n, _ in ITEMS))
    sys.exit(0)


if __name__ == "__main__":
    main()
