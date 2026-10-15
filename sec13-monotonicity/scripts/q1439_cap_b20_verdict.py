"""q1439 (9 Oct 2026): VERDICT gate for the derived-series births law on the capacity half-line chain with the buffer B1 = 20 (table q1439_cap_b20_table.json, sha256 REG below: stages D = 122, 128, 134 of centre_scripts/c50_capc_D140_B20.json, K_b 82 to 114, with the stage spread max(|122 - 134|, |128 - 134|) per K_b).
Written after the readings q1438 and q1438b (9 Oct 11:3x) had shown that the deviations of K_b 82 to 101 from the stage D = 134 are within 0.0102e-9, against up to 0.0984e-9 for the B1 = 14 stage of the registered table D1 (amendment 1968).
Item P (POST HOC, K_b 82 to 101, amplitudes already measured): |A_meas - a_pred(134)| < 0.05e-9 for all twenty events. K_b 81 (+0.0968e-9 at this stage, +0.0980e-9 at the B1 = 14 stage, stage independent and not explained) is outside the range, chosen after seeing it.
Items B102 (K_b 102 to 107) and B108 (K_b 108 to 114) (BLIND: no capacity amplitude above K_b 101 existed or was read when the table was registered): |A_meas - a_pred(134)| <= max(0.05e-9, 2 x spread(K_b)). The floor 0.05e-9 is 5 times the largest deviation of K_b 82 to 101 (0.0102e-9).
Each item: FAIL if one measured event violates, INCOMPLETE if none violates and one is not measured, PASS otherwise. Measured amplitude: K_b 82 from the tower file (q1422.measured), K_b >= 83 A_event of small_gap_q206/fw_prod/event_Kb<K_b>.json. VOID (exit 2) if a hash differs, a file is unreadable or a control fails. The items D1b, S2 and BR1 of amendments 1959, 1968 and 2164 stay in force unchanged. Last line: VERDICT q1439: P ...; B102 ...; B108 ... Usage: py q1439_cap_b20_verdict.py [--registration-run] (from temporal_within/centre_scripts/).
"""

import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
T = os.path.join(HERE, "q1439_cap_b20_table.json")
REG = "6825a2e19ca8e95944ee78019290e6e2440bc9300180c80a3627e104753c7ef5"
FLOOR = 0.05e-9
ITEMS = [
    ("P", range(82, 102), "post"),
    ("B102", range(102, 108), "blind"),
    ("B108", range(108, 115), "blind"),
]


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
    if K == 82:
        import q1422_blind_items_verdict as G22

        return G22.measured(82)
    f = os.path.join(ROOT, "small_gap_q206", "fw_prod", f"event_Kb{K}.json")
    return float(json.load(open(f))["A_event"]) if os.path.exists(f) else None


def judge(pred, spread, meas):
    out = {}
    for name, rng, kind in ITEMS:
        viol = []
        miss = []
        for K in rng:
            m = meas.get(K)
            if m is None:
                miss.append(K)
                continue
            d = abs(m - pred[K])
            inside = (d < FLOOR) if kind == "post" else (d <= max(FLOOR, 2 * spread[K]))
            if not inside:
                viol.append((K, m - pred[K]))
        out[name] = ("FAIL" if viol else ("INCOMPLETE" if miss else "PASS"), viol, miss)
    return out


def main(loader=None):
    try:
        pred, spread = load(T, REG)
        meas = {K: (loader or measured)(K) for K in range(82, 115)}
    except Exception as e:
        print("VOID: input unreadable or table hash differs:", repr(e))
        print("VERDICT q1439: VOID")
        sys.exit(2)
    base = {K: pred[K] for K in range(82, 115)}
    cA = judge(pred, spread, base)
    cB = judge(pred, spread, {**base, 90: pred[90] + 0.06e-9})
    cC = judge(pred, spread, {**base, 105: pred[105] + 3 * max(FLOOR, 2 * spread[105])})
    cD = judge(pred, spread, {K: v for K, v in base.items() if K != 110})
    cE = judge(pred, spread, {**base, 105: pred[105] + 0.9 * max(FLOOR, 2 * spread[105])})
    ok = (
        all(v[0] == "PASS" for v in cA.values())
        and cB["P"][0] == "FAIL"
        and cC["B102"][0] == "FAIL"
        and cD["B108"][0] == "INCOMPLETE"
        and cE["B102"][0] == "PASS"
    )
    print(
        "controls: events at the table -> all PASS; one P event 0.06e-9 off ->",
        cB["P"][0],
        "; one B102 event 3 bars off ->",
        cC["B102"][0],
        "; K_b 110 missing -> B108",
        cD["B108"][0],
        "; one event 0.9 bar off -> B102",
        cE["B102"][0],
        ":",
        ok,
    )
    if not ok:
        print("VERDICT q1439: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("controls only")
        sys.exit(0)
    r = judge(pred, spread, meas)
    for name, rng, kind in ITEMS:
        v, viol, miss = r[name]
        print(
            f"{name} ({kind}, K_b {rng[0]} to {rng[-1]}): {v}; measured {len(rng) - len(miss)} of {len(rng)}; largest |dev| "
            + (
                "%.4fe-9" % (max(abs(meas[K] - pred[K]) for K in rng if meas.get(K) is not None) * 1e9)
                if len(miss) < len(rng)
                else "none"
            )
            + "; violations "
            + str([(K, round(d * 1e9, 4)) for K, d in viol])
        )
    print("VERDICT q1439: " + "; ".join(f"{n} {r[n][0]}" for n, _, _ in ITEMS))
    sys.exit(0)


if __name__ == "__main__":
    main()
