"""q1204 (8 Oct 2026, prepared under amendments 1959 and 1960): READING of the blind items S1 and S2 of amendment 1959: for each capacity event K_b >= 81 whose measured amplitude exists, the deviation A_meas - a_pred from the table q1201_table.json (sha256 fcf88352217a8017344239510feeb38f1318f2d7751a763c7bfaf21ba844aaac).
S1 for K_b 81 to 99: |A_meas - a_pred| < 0.5e-9; S2 for K_b 100 to 114: |A_meas - a_pred| < 2e-9.
Controls (registration run, no K_b >= 81 value is read): the sha256 of the table file content (without its hash key) equals the registered one; the function of q1201 reproduces the table value of K_b 81 and 90 within 1e-12; the deviations A_meas - a_pred of K_b 79 and 80 with the same interpolation are +0.027e-9 and +0.024e-9 (amendment 1958 and the post hoc table, tolerance 0.002e-9).
Usage: py q1204_table_test_reading.py [--registration-run]  (from temporal_within/centre_scripts/)"""

import hashlib, json, os, sys

sys.argv_saved = list(sys.argv)
sys.argv = ["x"]
import q1201_births_law_table_shared_centre as T

REG = "fcf88352217a8017344239510feeb38f1318f2d7751a763c7bfaf21ba844aaac"


def main():
    js = json.load(open("q1201_table.json"))
    sha = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    ok = h == REG and sha == REG
    print(f"control: hash of the table content {h[:16]}... equals the registered hash: {ok}")
    Dm, oc, N = T.stage_fn("c50_capc_D110.json")
    for K in (81, 90):
        a = T.pred(K, N, oc)
        g = abs(a - js["a_pred"][str(K)]) < 1e-12
        ok = ok and g
        print(f"control: K_b {K} recomputed {a:.10f} against the table {js['a_pred'][str(K)]:.10f} -> {g}")
    os.chdir("..")
    sys.path.insert(0, ".")
    import q1183_births_residual_halfline_depth_reading as Q

    exp = {79: 0.027e-9, 80: 0.024e-9}
    for K, e in exp.items():
        d = Q.measured(K) - T.pred(K, N, oc)
        g = abs(d - e) < 2e-12
        ok = ok and g
        print(f"control: K_b {K} deviation {d * 1e9:+.4f}e-9 against {e * 1e9:+.3f}e-9 -> {g}")
    if not ok:
        print("READING q1204: VOID (control)")
        sys.exit(2)
    if "--registration-run" in sys.argv_saved:
        print("READING q1204: controls only")
        sys.exit(0)
    done = {}
    for K in sorted(int(k) for k in js["a_pred"]):
        try:
            m = Q.measured(K)
        except Exception:
            continue
        d = m - js["a_pred"][str(K)]
        bar = 0.5e-9 if K <= 99 else 2e-9
        done[K] = (d, abs(d) < bar)
        print(
            f"K_b {K}: A_meas {m:.12f}, a_pred {js['a_pred'][str(K)]:.10f}, deviation {d * 1e9:+.4f}e-9, item {'S1' if K <= 99 else 'S2'} (< {bar * 1e9:.1f}e-9): {abs(d) < bar}"
        )
    if not done:
        print("READING q1204: no measured event above K_b 80 yet")
        sys.exit(0)
    print(
        "READING q1204: "
        + ", ".join(f"K_b {K} {'True' if ok_ else 'False'}" for K, (d, ok_) in done.items())
        + " (no verdict)"
    )


if __name__ == "__main__":
    main()
