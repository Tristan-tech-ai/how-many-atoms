"""undelivered_events.py (10 Oct 2026, HELPER, no verdict word): lists, by file name only, the event files that exist in the chain directories (fw_cap_par_chain*, fw_imm_chain*, fw_imm_seedcap_*) but are not yet in the merged directories fw_cap_par and fw_imm_par. It prints K_b numbers and directory names, never an amplitude. Written after the chain check of 06:47 missed K_b 132 and 134, which lay in their chain directories since 05:40 and 06:02: check_progress.sh prints the merged list (CAP par, IMM par) and a chain list that was cut off in the output. Run it at every chain check, then cap_cycle.sh (capacity) or imm_merge.py (immune) if it prints anything. Usage: py undelivered_events.py (from small_gap_q206/)."""

import os, re, glob, sys

os.chdir(
    os.path.dirname(os.path.abspath(__file__))
)  # 10 Oct 07:5x: the first version used paths relative to the caller's directory and printed 'merged events: 0' when called from temporal_within (chain checks of 07:17 and 07:48)


def ev(d):
    return (
        {int(re.search(r"Kb(\d+)", f).group(1)) for f in os.listdir(d) if re.match(r"event_Kb\d+\.json$", f)}
        if os.path.isdir(d)
        else set()
    )


for kind, merged, pats in (
    ("capacity", "fw_cap_par", ["fw_cap_par_chain*"]),
    ("immune", "fw_imm_par", ["fw_imm_chain*", "fw_imm_seedcap_*"]),
):
    have = ev(merged)
    found = {}
    for pat in pats:
        for d in sorted(glob.glob(pat)):
            for k in ev(d):
                if k not in have:
                    found.setdefault(k, []).append(d)
    if len(have) == 0:
        print("ERROR: zero merged events counted for", kind, "- the readout is invalid")
        sys.exit(3)
    print(kind, "merged events:", len(have), "; undelivered K_b:", sorted(found) if found else "none")
    for k in sorted(found):
        print("   K_b", k, "in", ", ".join(found[k]))
print("HELPER undelivered_events done (no verdict word)")
