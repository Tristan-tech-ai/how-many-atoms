"""q1442 (9 Oct 2026, INSTRUMENT, prints no amplitude): merge the capacity events of the one-event driver (fw_cap_par/event_Kb<K>.json, q1440) into fw_prod, where the readers q1413 / q1422 / q1424 / q1439 look for them. Rule: if fw_prod has no file for K_b, the file of fw_cap_par is copied there with the extra key 'source': 'fw_cap_par'; if fw_prod has a file WITHOUT that key (the sequential walker) and fw_cap_par has one, the two A_event must agree to 1e-10 or the script prints 'DISAGREE' for that K_b and exits 2 (nothing is overwritten); a file in fw_prod with the key is replaced by the sequential one when the walker writes it, and then compared the same way on the next run. Only differences are printed, never amplitudes. Usage: py q1442_merge_cap_events.py [--selftest] (from temporal_within/small_gap_q206/)."""

import json, os, shutil, sys, tempfile
from decimal import Decimal


def merge(
    par_dir, seq_dir, ks=range(100, 150)
):  # was range(100, 115) until 9 Oct 18:5x: events K_b >= 115 were NOT merged into fw_prod and the gate q1451 read 0 of 8 (instrument defect found at 18:55)
    status = {}
    for K in ks:
        p = os.path.join(par_dir, f"event_Kb{K}.json")
        s = os.path.join(seq_dir, f"event_Kb{K}.json")
        if not os.path.exists(p):
            continue
        jp = json.load(open(p))
        if not os.path.exists(s):
            jp["source"] = "fw_cap_par"
            json.dump(jp, open(s, "w"))
            status[K] = "copied"
            continue
        js = json.load(open(s))
        if js.get("source") == "fw_cap_par":
            status[K] = "copy-present"
            continue
        d = abs(Decimal(js["A_event"]) - Decimal(jp["A_event"]))
        status[K] = f"agree {float(d):.1e}" if d < Decimal("1e-10") else f"DISAGREE {float(d):.1e}"
    return status


def selftest():
    t = tempfile.mkdtemp()
    par = os.path.join(t, "par")
    seq = os.path.join(t, "seq")
    os.makedirs(par)
    os.makedirs(seq)

    def w(d, K, a, **kw):
        json.dump({"K_b": K, "A_event": a, **kw}, open(os.path.join(d, f"event_Kb{K}.json"), "w"))

    w(par, 110, "60.123456789012345678")
    w(par, 111, "61.1")
    w(seq, 111, "61.1000000000001")
    w(par, 112, "62.2")
    w(seq, 112, "62.2002")
    w(par, 113, "63.3")
    w(seq, 113, "63.3", source="fw_cap_par")
    st = merge(par, seq, range(110, 115))
    ok = (
        st.get(110) == "copied"
        and st.get(111, "").startswith("agree")
        and st.get(112, "").startswith("DISAGREE")
        and st.get(113) == "copy-present"
        and 114 not in st
        and json.load(open(os.path.join(seq, "event_Kb110.json")))["source"] == "fw_cap_par"
        and json.load(open(os.path.join(seq, "event_Kb112.json")))["A_event"] == "62.2002"
    )
    print("selftest q1442:", st, "->", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    st = merge("fw_cap_par", "fw_prod")
    print({K: v for K, v in st.items()})
    sys.exit(2 if any(v.startswith("DISAGREE") for v in st.values()) else 0)
