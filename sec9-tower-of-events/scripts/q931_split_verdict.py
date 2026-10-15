"""q931 (7 Oct 2026): verdict script of T64 (items S0-S3 and controls C1, C2, C3, PC of amendment 1707). It reads the outputs of q930 and applies the criteria registered before any K 62 state existed.
Usage: py q931_split_verdict.py SCAN_41.7375.json SCAN_41.7325.json SCAN_41.6525.json AS.json [OLD_41.7375.json OLD_41.7325.json OLD_41.6525.json]
The optional old files (the first run of amendment 1708) give the instrument check I1 of amendment 1709: at every point converged in both runs g agrees to a relative difference of 1e-9; a larger difference or fewer than 30 common points makes the verdict VOID.
Last line: VERDICT q931 T64: PASS | FAIL | VOID (reason). Exit 0 for PASS, 1 for FAIL, 2 for VOID."""

import json, sys
import numpy as np

NPTS = 13  # amendment 1709: the scan has 13 points (eps = 0.01 * 1.3^k, k = 0..12)


def void(msg):
    print("VOID q931: " + msg, flush=True)
    print("VERDICT q931 T64: VOID (" + msg + ")", flush=True)
    sys.exit(2)


def load(p):
    try:
        d = json.load(open(p))
    except Exception as e:
        void(f"unreadable {p}: {e}")
    if "scan" not in d and "A_s" not in d:
        void(f"{p} has neither scan nor A_s")
    return d


def s0(d):
    sc = d["scan"]
    if not sc or not sc[0]["ok"]:
        return False, "first scan point missing or not converged"
    return (
        bool(np.sign(sc[0]["g"]) == np.sign(d["centre_signal_K61"])),
        f"sign g(0.01) {np.sign(sc[0]['g']):+.0f}, sign centre signal {np.sign(d['centre_signal_K61']):+.0f}",
    )


def s1(d, flip=False):
    sc = d["scan"]
    r = d.get("root")
    if len(sc) != NPTS or not all(p["ok"] for p in sc):
        return False, f"scan points {len(sc)} (converged {sum(1 for p in sc if p['ok'])})"
    g = [(-p["g"] if flip else p["g"]) for p in sc]
    chg = [i for i in range(NPTS - 1) if g[i] > 0 and g[i + 1] < 0]
    if not chg:
        return False, "no sign change from + to -"
    if not r or "eps" not in r:
        return False, f"sign change at scan index {chg[0]} but no refined root ({r})"
    ok = (
        bool(r["full_newton_ok"])
        and r["full_residual"] < 1e-30
        and r["ordered"]
        and r["wmin"] > 0
        and r["centre"] <= 0
        and r["off"] <= 0
        and r["verdict"] == "OPTIMAL"
    )
    return (
        ok,
        f"sign change between eps {sc[chg[0]]['eps']:.4f} and {sc[chg[0] + 1]['eps']:.4f}; root eps {r['eps']:.6f}, full residual {r['full_residual']:.2e}, verdict {r['verdict']}, centre {r['centre']:.3e}, off {r['off']:.3e}",
    )


def s2(d, flip=False):
    sc = d["scan"]
    if len(sc) != NPTS or not all(p["ok"] for p in sc):
        return None, f"scan points {len(sc)} (converged {sum(1 for p in sc if p['ok'])})"
    g = [(-p["g"] if flip else p["g"]) for p in sc]
    return (
        all(x < 0 for x in g),
        f"g negative at {sum(1 for x in g if x < 0)} of {NPTS} points, largest g {max(g):.3e}",
    )


def s3(d_branch, A_s, shift=0.0, reverse=False):
    b = sorted([r for r in d_branch if not r.get("failed")], key=lambda r: r["eps"])
    if len(b) < 6:
        return False, f"only {len(b)} branch points"
    eps = np.array([r["eps"] for r in b])
    A = np.array([r["A_f"] for r in b])
    if reverse:
        eps = eps[::-1]  # control C2: the eps values paired with the A values in reverse order
    o = np.argsort(eps)
    eps = eps[o]
    A = A[o]
    inc = bool(np.all(np.diff(A) > 0))
    a2, a0 = np.polyfit(eps[:4] ** 2, A[:4], 1)
    ok = bool(inc and abs(a0 - (A_s + shift)) <= 5e-4 and a2 > 0)
    return (
        ok,
        f"{len(b)} points, A increasing {inc}, fit a0 {a0:.7f}, a2 {a2:.4f}, A_s {A_s + shift:.7f}, |a0 - A_s| {abs(a0 - (A_s + shift)):.2e}",
    )


def synthetic():
    eps = [0.01 * 1.3**k for k in range(NPTS)]
    pos = {
        "centre_signal_K61": 1.4e-14,
        "scan": [{"eps": e, "ok": True, "g": 1e-13 - 5e-12 * e * e} for e in eps],
        "root": {
            "eps": 0.1414,
            "full_newton_ok": True,
            "full_residual": 1e-40,
            "ordered": True,
            "wmin": 0.01,
            "centre": -1e-15,
            "off": -1e-16,
            "verdict": "OPTIMAL",
        },
    }
    neg = {
        "centre_signal_K61": -3.5e-15,
        "scan": [{"eps": e, "ok": True, "g": -1e-13 - 5e-13 * e * e} for e in eps],
    }
    br = [{"eps": e, "A_f": 41.7335 + 0.5 * e * e} for e in np.geomspace(0.02, 0.234, 10)]
    return pos, neg, br, 41.7335


def run(pos, n1, n2, br, A_s, label):
    out = {}
    for name, d in (("41.7375", pos), ("41.7325", n1), ("41.6525", n2)):
        out["S0 " + name] = s0(d)
    out["S1 41.7375"] = s1(pos)
    out["S2 41.7325"] = s2(n1)
    out["S2 41.6525"] = s2(n2)
    out["S3"] = s3(br, A_s)
    return out


def main():
    if len(sys.argv) not in (5, 8):
        sys.exit(__doc__)
    pos, n1, n2, As = (load(p) for p in sys.argv[1:5])
    br = pos.get("branch", [])
    if len(sys.argv) == 8:
        common = 0
        worst = 0.0
        for new, oldp in zip((pos, n1, n2), sys.argv[5:8]):
            old = load(oldp)
            for a in new["scan"]:
                for b in old["scan"]:
                    if a["ok"] and b["ok"] and abs(a["eps"] - b["eps"]) < 1e-12:
                        common += 1
                        worst = max(worst, abs(a["g"] - b["g"]) / abs(b["g"]))
        print(
            f"I1: common converged points {common}, largest relative difference of g between the two runs {worst:.3e}"
        )
        if common < 30 or worst > 1e-9:
            void(f"instrument check I1 fails (common {common}, worst {worst:.3e})")
    # controls on the synthetic family (positive control PC) and on the same data (C1, C2, C3)
    sp, sn, sbr, sA = synthetic()
    pc = run(sp, sn, sn, sbr, sA, "PC")
    if not all(v[0] is True for v in pc.values()):
        void("positive control PC does not pass: " + str({k: v for k, v in pc.items() if v[0] is not True}))
    clean = run(pos, n1, n2, br, As["A_s"], "real")
    c1 = (s1(pos, flip=True)[0], s2(n1, flip=True)[0] if s2(n1)[0] is not None else None)
    c2 = s3(br, As["A_s"], reverse=True)[0]
    c3 = s3(br, As["A_s"], shift=0.002)[0]
    print(
        f"controls: PC passes all {len(pc)} items True; C1 (g -> -g) gives S1 {c1[0]}, S2 {c1[1]}; C2 (reverse eps order) gives S3 {c2}; C3 (A_s + 0.002) gives S3 {c3}"
    )
    for k, (ok, txt) in clean.items():
        print(f"   {k}: {ok}  {txt}")
    if clean["S1 41.7375"][0] and c1[0]:
        void("control C1 does not move S1")
    if clean["S2 41.7325"][0] and c1[1]:
        void("control C1 does not move S2")
    if clean["S3"][0] and (c2 or c3):
        void("control C2 or C3 does not move S3")
    if not all(clean[f"S0 {n}"][0] for n in ("41.7375", "41.7325", "41.6525")):
        void("S0 violated (instrument)")
    if clean["S2 41.7325"][0] is None or clean["S2 41.6525"][0] is None:
        void("a scan point of S2 did not converge")
    if len(pos["scan"]) != NPTS or not all(p["ok"] for p in pos["scan"]):
        void("a scan point of S1 did not converge")
    fails = [k for k, (ok, _) in clean.items() if ok is not True]
    verdict = "PASS" if not fails else "FAIL"
    print(f"VERDICT q931 T64: {verdict}; failed items {fails}; A_s {As['A_s']:.7f}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
