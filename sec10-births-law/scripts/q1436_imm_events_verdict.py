"""q1436 (9 Oct 2026): VERDICT gate for the BLIND immune transitions K_b 100 to 150 against the derived-series table q1435_imm_table.json (order 12, no immune input, immune half-line stage c33_immc_D146 at D = 116, 128 and 134; sha256 REG below), registered BEFORE any immune event above K_b 99 was located (the control walk of q1431 ended at A = 60.985, before K_b 100).
Item B<k> (blocks of ten: 100-109, 110-119, 120-129, 130-139, 140-150): for every K_b of the block, |a_t - a_pred(D = 134)| <= bar(K_b) with bar = max(0.05e-9, 2 x spread(K_b)) and spread(K_b) = max(|a_pred(116) - a_pred(134)|, |a_pred(128) - a_pred(134)|) stored in the table (the sensitivity of the prediction to the stage depth, 2.5e-11 at K_b 100 and 2.6e-7 at K_b 150). The bar 0.05e-9 is about 3.5 times the largest deviation of the stored immune events K_b 79 to 99 (0.014e-9) from the same prediction. A block is FAIL if one measured event of it violates the bar, INCOMPLETE if none violates and one is not measured, PASS if all are measured and inside.
The measured amplitude of K_b is A_event in small_gap_q206/fw_imm/event_Kb<K_b>.json (sequential walk of q1431_imm_walk.py) and/or small_gap_q206/fw_imm_par/event_Kb<K_b>.json (one-event driver q1437_imm_event_at.py, added by amendment 2168), both 192 bits and located by the secant on the centre signal to a bracket below 1e-12; when both exist they must agree to 1e-10 (otherwise VOID) and the first is used.
VOID (exit 2): table hash differs, a file is unreadable, or the internal controls fail. Last line: VERDICT q1436: B100 ...; B110 ...; B120 ...; B130 ...; B140 ... Usage: py q1436_imm_events_verdict.py [--registration-run] (from temporal_within/centre_scripts/).
"""

import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T = os.path.join(HERE, "q1435_imm_table.json")
REG = "bb38b43d77ad1d5dad4d635933e6d2eef607c5ac4f8dd48b08e3962a4219e37b"
BLOCKS = [
    ("B100", range(100, 110)),
    ("B110", range(110, 120)),
    ("B120", range(120, 130)),
    ("B130", range(130, 140)),
    ("B140", range(140, 151)),
]
FLOOR = 0.05e-9


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
    """the amplitude of K_b from the sequential walk (fw_imm) and/or the one-event driver q1437 (fw_imm_par); a number, None, or a tuple of two numbers when both exist"""
    vals = []
    for d in ("fw_imm", "fw_imm_par"):
        f = os.path.join(ROOT, "small_gap_q206", d, f"event_Kb{K}.json")
        if os.path.exists(f):
            vals.append(float(json.load(open(f))["A_event"]))
    return None if not vals else (vals[0] if len(vals) == 1 else tuple(vals))


def single(K, m):
    """one amplitude per K_b; when two sources exist they must agree to 1e-10, otherwise the input is unusable (VOID)"""
    if isinstance(m, tuple):
        if max(m) - min(m) > 1e-10:
            raise ValueError(f"K_b {K}: the two sources differ by {max(m) - min(m):.2e}")
        return m[0]
    return m


def judge(pred, spread, meas):
    out = {}
    for name, rng in BLOCKS:
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


def main(loader=None):
    try:
        pred, spread = load(T, REG)
        meas = {}
        for K in range(100, 151):
            meas[K] = single(K, (loader or measured)(K))
    except Exception as e:
        print("VOID: input unreadable or table hash differs:", repr(e))
        print("VERDICT q1436: VOID")
        sys.exit(2)
    cA = judge(pred, spread, {K: pred[K] for K in range(100, 151)})
    cB = judge(
        pred,
        spread,
        {K: (pred[K] + 3 * max(FLOOR, 2 * spread[K]) if K == 125 else pred[K]) for K in range(100, 151)},
    )
    cC = judge(pred, spread, {K: pred[K] for K in range(100, 151) if K != 115})
    cD = judge(
        pred,
        spread,
        {K: (pred[K] + 0.9 * max(FLOOR, 2 * spread[K]) if K == 105 else pred[K]) for K in range(100, 151)},
    )
    ok = (
        all(v[0] == "PASS" for v in cA.values())
        and cB["B120"][0] == "FAIL"
        and cB["B100"][0] == "PASS"
        and cC["B110"][0] == "INCOMPLETE"
        and cD["B100"][0] == "PASS"
    )
    print(
        "controls: events at the table -> all PASS;",
        "one event 3 bars off (K_b 125) -> B120",
        cB["B120"][0],
        ";",
        "one event missing (K_b 115) -> B110",
        cC["B110"][0],
        ";",
        "one event 0.9 bar off -> B100",
        cD["B100"][0],
        ":",
        ok,
    )
    if not ok:
        print("VERDICT q1436: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("controls only")
        sys.exit(0)
    r = judge(pred, spread, meas)
    for name, rng in BLOCKS:
        v, viol, miss = r[name]
        print(
            f"{name} (K_b {rng[0]} to {rng[-1]}): {v}; measured {len(rng) - len(miss)} of {len(rng)}; deviations (1e-9) "
            + ", ".join(f"{K} {(meas[K] - pred[K]) * 1e9:+.4f}" for K in rng if meas.get(K) is not None)
        )
    print("VERDICT q1436: " + "; ".join(f"{n} {r[n][0]}" for n, _ in BLOCKS))
    sys.exit(0)


if __name__ == "__main__":
    main()
