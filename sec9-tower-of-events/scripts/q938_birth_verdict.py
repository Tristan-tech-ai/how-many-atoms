"""q938 (7 Oct 2026): verdict script of T67 (items B1-B4 and controls C1, C2, PC of amendment 1714). It reads the output of q937.
Usage: py q938_birth_verdict.py OUT.json    Last line: VERDICT q938 T67: PASS | FAIL | VOID (reason). Exit 0 PASS, 1 FAIL, 2 VOID.
"""

import json, sys


def void(msg):
    print("VOID q938: " + msg, flush=True)
    print("VERDICT q938 T67: VOID (" + msg + ")", flush=True)
    sys.exit(2)


def items(d, flip=False, shift=0.0):
    k61 = [r for r in d["K61"] if not r.get("secant")]
    k60 = d["K60"][:2]
    if len(k61) < 4 or len(k60) < 2:
        void(f"too few states (K 61 {len(k61)}, K 60 {len(k60)})")
    sgn = -1.0 if flip else 1.0
    s = sorted(k61, key=lambda r: r["A"])
    inc = all(sgn * b["wc"] > sgn * a["wc"] for a, b in zip(s, s[1:]))
    root = abs(d["wc_at_cross"]) < 1e-12 and s[0]["wc"] * 1.0 < 0
    b1 = bool(inc and root)
    b2 = bool(abs(d["A_cross"] - (d["A_b0"] + shift)) <= 5e-4)
    b3 = bool(d["slope_wc"] is not None and sgn * d["slope_wc"] > 0)
    kk = sorted(k60, key=lambda r: r["A"])
    b4 = bool(kk[0]["centre"] < 0 < kk[1]["centre"])
    return {
        "B1": (
            b1,
            f"{len(s)} states, wc increasing with A {inc}, lowest state wc {s[0]['wc']:.3e}, root wc {d['wc_at_cross']:.2e}",
        ),
        "B2": (
            b2,
            f"A_cross {d['A_cross']:.9f}, A_b0 {d['A_b0'] + shift:.9f}, difference {abs(d['A_cross'] - d['A_b0'] - shift):.2e}",
        ),
        "B3": (b3, f"slope dwc/dA at the root {d['slope_wc']:.4f}"),
        "B4": (
            b4,
            f"K 60 centre signals {kk[0]['centre']:.3e} at A {kk[0]['A']}, {kk[1]['centre']:.3e} at A {kk[1]['A']}",
        ),
    }


def synthetic():
    ks = [{"A": 41.21 - 0.0025 * k, "wc": 0.0028 - 0.3 * 0.0025 * k} for k in range(5)]
    return {
        "K61": ks,
        "K60": [{"A": 41.21, "centre": 2.7e-15}, {"A": 41.20, "centre": -2.3e-16}],
        "A_cross": 41.2007,
        "wc_at_cross": 1e-15,
        "slope_wc": 0.3,
        "A_b0": 41.2007,
    }


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        d = json.load(open(sys.argv[1]))
    except Exception as e:
        void(f"unreadable {sys.argv[1]}: {e}")
    pc = items(synthetic())
    if not all(v[0] for v in pc.values()):
        void("positive control PC does not pass: " + str(pc))
    clean = items(d)
    for k, (ok, txt) in clean.items():
        print(f"   {k}: {ok}  {txt}")
    c1 = items(d, flip=True)
    c2 = items(d, shift=0.002)
    print(
        f"controls: PC passes {len(pc)} items; C1 (wc -> -wc) gives B1 {c1['B1'][0]}, B3 {c1['B3'][0]}; C2 (A_b0 + 0.002) gives B2 {c2['B2'][0]}"
    )
    if clean["B1"][0] and c1["B1"][0]:
        void("control C1 does not move B1")
    if clean["B3"][0] and c1["B3"][0]:
        void("control C1 does not move B3")
    if clean["B2"][0] and c2["B2"][0]:
        void("control C2 does not move B2")
    if not clean["B4"][0]:
        void("B4 (instrument consistency of the K 60 centre signals) violated")
    fails = [k for k, (ok, _) in clean.items() if not ok]
    v = "PASS" if not fails else "FAIL"
    print(f"VERDICT q938 T67: {v}; failed items {fails}")
    sys.exit(0 if v == "PASS" else 1)


if __name__ == "__main__":
    main()
