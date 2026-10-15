"""q956 (7 Oct 2026): verdict script of T70 (amendment 1735): do the local constants of the tower events drift linearly with the event index? Two blind items.
N1 the slope dwc/dA at the birth K 64 -> 65 (tower_K65_birth_out.json, key slope_wc) lies within 0.0010 of 0.38889 (linear extrapolation of the birth slopes 0.3949182 at K_b 60 and 0.3919038 at K_b 62: -0.0030144 per two events).
N2 the pitchfork width a2 of the split K 65 -> 66 (fit A = a0 + a2 eps^2 over the four smallest eps of the branch in tower_K66_scan_hi.json) lies within 0.0010 of 0.25889 (linear extrapolation of a2 = 0.26281 at K_b 61 and 0.26085 at K_b 63: -0.00196 per two events).
Verdict (last line): PASS if both are measured and within their bars; FAIL if a measured item is outside; INCOMPLETE if none fails and one is missing; VOID if a file is unreadable or a control does not behave.
Controls: PC (the predicted values) must PASS; NC1 (the constant model 0.3919 and 0.2608) must FAIL; NC2 (one item missing) must be INCOMPLETE.
Usage: py q956_constants_verdict.py"""

import json, os, sys
import numpy as np

A = "scratch/"
P1, P2, BAR = 0.38889, 0.25889, 0.0010


def a2_of(f):
    d = json.load(open(f))
    br = [r for r in d["branch"] if not r.get("failed")]
    if len(br) < 6:
        return None
    e = np.array([r["eps"] for r in br])
    y = np.array([r["A_f"] for r in br])
    o = np.argsort(e)
    return float(np.polyfit(e[o][:4] ** 2, y[o][:4], 1)[0])


def verdict(n1, n2):
    items = {
        "N1": (None if n1 is None else abs(n1 - P1) <= BAR),
        "N2": (None if n2 is None else abs(n2 - P2) <= BAR),
    }
    if any(v is False for v in items.values()):
        return "FAIL", items
    if any(v is None for v in items.values()):
        return "INCOMPLETE", items
    return "PASS", items


def main():
    c = {
        "PC": verdict(P1, P2)[0] == "PASS",
        "NC1": verdict(0.3919, 0.2608)[0] == "FAIL",
        "NC2": verdict(P1, None)[0] == "INCOMPLETE",
    }
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q956 T70: VOID (a control does not behave)")
        sys.exit(2)
    n1 = n2 = None
    try:
        if os.path.exists(A + "tower_K65_birth_out.json"):
            n1 = float(json.load(open(A + "tower_K65_birth_out.json"))["slope_wc"])
        if os.path.exists(A + "tower_K66_scan_hi.json"):
            n2 = a2_of(A + "tower_K66_scan_hi.json")
    except Exception as e:
        print(f"VERDICT q956 T70: VOID (unreadable file: {e})")
        sys.exit(2)
    v, it = verdict(n1, n2)
    print(
        f"   N1 slope at K_b 64: "
        + (f"{n1:.5f} against {P1} (bar {BAR}) -> {it['N1']}" if n1 is not None else "not measured yet")
    )
    print(
        f"   N2 a2 at K_b 65: "
        + (f"{n2:.5f} against {P2} (bar {BAR}) -> {it['N2']}" if n2 is not None else "not measured yet")
    )
    print(f"VERDICT q956 T70: {v}; items {it}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
