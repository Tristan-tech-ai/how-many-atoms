#!/bin/bash
# run_k80_hi2_verify.sh (8 Oct 2026, registered in amendment 1925): repair of the split verdict K 79 -> 80: a second non-optimal state above the split by the unchanged q942, its scan by the unchanged q935, and q931 again.
cd "." || exit 1
export PYTHONIOENCODING=utf-8
A=scratch
mkdir -p tower/K80_hi2 tower/K80_split_hi2
py q942_trace_cubic.py tower/K79/results_q220_gauss_K79_A50.9867.json tower/K80_hi2 0.005 51.0 60.0 > $A/tower_K80_hi2_trace.log 2>&1
echo "trace exit $? at $(date +%H:%M:%S)"
ls tower/K80_hi2 > $A/tower_K80_hi2_files.txt
# the state above hi (the largest A of the directory) is the state hi2; the check of the registered conditions is printed, the decision is taken by reading it
python - <<'EOF'
import glob, json, os
fs = sorted(glob.glob("tower/K80_hi2/results_q220_gauss_K79_A*.json"))
rows = []
for f in fs:
    j = json.load(open(f)); rows.append((float(j["A"]), f, j["verdict"], float(j["residual"])))
rows.sort()
for r in rows: print("state", r[0], r[2], "residual", r[3], "A - A_s", r[0] - 50.986686686158414)
cand = [r for r in rows if r[0] > 50.9867 and r[2] == "centre positive" and r[3] < 1e-20 and 0.0025 <= r[0] - 50.986686686158414 <= 0.0060]
print("hi2 candidate:", cand[-1][1] if cand else None)
open("scratch/tower_K80_hi2_state.txt", "w").write(cand[-1][1] if cand else "NONE")
EOF
H2=$(cat $A/tower_K80_hi2_state.txt)
if [ "$H2" = "NONE" ]; then echo "VOID: no state hi2 satisfies the registered conditions"; touch $A/k80_hi2_verify.done; exit 3; fi
Q935_GRID=60.0 Q935_ODIR=tower/K80_split_hi2 py q935_split_generic.py scan "$H2" $A/tower_K80_scan_hi2.json > $A/tower_K80_scan_hi2.log 2>&1
echo "scan exit $? at $(date +%H:%M:%S)"
py q931_split_verdict.py $A/tower_K80_scan_hi2.json $A/tower_K80_scan_lo.json $A/tower_K80_scan_mid.json $A/tower_K80_As.json > $A/tower_K80_q931_rerun.txt 2>&1
echo "q931 exit $? at $(date +%H:%M:%S): $(tail -n 1 $A/tower_K80_q931_rerun.txt | cut -c1-200)"
touch $A/k80_hi2_verify.done
