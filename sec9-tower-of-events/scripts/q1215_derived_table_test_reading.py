"""q1215 (8 Oct 2026, prepared under amendment 1968): READING of the blind item D1 of amendment 1968 for each capacity event K_b >= 81 whose measured amplitude exists: the deviation A_meas - a_pred from the registered table of the derived series (q1209_predictions.json, order 12, sha256 f48c3ea92d5bd7978c8d1400dd04136490bcbe08a42396afd03d75a523f4fc0d), and as a reading from the order-16 table (q1209_predictions_o16.json).
D1: |A_meas - a_pred| < 0.3e-9 for K_b 81 to 99, < 1e-9 for K_b 100 to 114.
Controls (checked first): the hash of the order-12 table content equals the registered one; the function of q1209 reproduces the table value of K_b 81 and K_b 90 within 1e-12; the deviations of K_b 79 and K_b 80 with the order-12 series are -0.014e-9 and -0.008e-9 (q1209 --read, tolerance 0.002e-9).
Usage: py q1215_derived_table_test_reading.py  (from temporal_within/centre_scripts/)"""

import hashlib, json, os, sys

os.environ.setdefault("E_FILE", "d22_out_12_keep.json")
sys.argv = ["x"]
import q1209_births_law_derived_series as G

REG = "f48c3ea92d5bd7978c8d1400dd04136490bcbe08a42396afd03d75a523f4fc0d"


def main():
    js = json.load(open("q1209_predictions.json"))
    sha = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    ok = h == REG and sha == REG
    print(f"control: hash of the order-12 table {h[:16]}... equals the registered hash: {ok}")
    for K in (81, 90):
        a = G.pred_series(G.Ncap, K, 12)
        g = abs(a - js["capacity"][str(K)]) < 1e-12
        ok = ok and g
        print(f"control: K_b {K} recomputed {a:.12f} against the table {js['capacity'][str(K)]:.12f} -> {g}")
    os.chdir("..")
    sys.path.insert(0, ".")
    import q1183_births_residual_halfline_depth_reading as Q

    for K, e in ((79, -0.014e-9), (80, -0.008e-9)):
        d = Q.measured(K) - G.pred_series(G.Ncap, K, 12)
        g = abs(d - e) < 2e-12
        ok = ok and g
        print(f"control: K_b {K} deviation at order 12 {d * 1e9:+.4f}e-9 against {e * 1e9:+.3f}e-9 -> {g}")
    if not ok:
        print("READING q1215: VOID (control)")
        sys.exit(2)
    o16 = json.load(open("centre_scripts/q1209_predictions_o16.json"))
    done = {}
    for K in range(81, 115):
        try:
            m = Q.measured(K)
        except Exception:
            continue
        d12 = m - js["capacity"][str(K)]
        d16 = m - o16["capacity"][str(K)]
        bar = 0.3e-9 if K <= 99 else 1e-9
        done[K] = abs(d12) < bar
        print(
            f"K_b {K}: A_meas {m:.12f}; order-12 table {js['capacity'][str(K)]:.12f}, deviation {d12 * 1e9:+.4f}e-9, item D1 (< {bar * 1e9:.1f}e-9): {abs(d12) < bar}; order-16 table deviation {d16 * 1e9:+.4f}e-9"
        )
    if not done:
        print("READING q1215: no measured event above K_b 80 yet")
        sys.exit(0)
    print("READING q1215: " + ", ".join(f"K_b {K} {v}" for K, v in done.items()) + " (no verdict)")


if __name__ == "__main__":
    main()
