#!/bin/bash
# usage: bash scratch/cap_cycle.sh TAG   (capacity: copy new chain events into fw_cap_par, merge, run readers and gates; prints summary lines only)
TAG=$1
cd "." || exit 9
A=scratch
for d in small_gap_q206/fw_cap_par_chain*; do
  for f in $d/event_Kb*.json; do
    [ -e "$f" ] || continue
    b=$(basename $f)
    if [ ! -e small_gap_q206/fw_cap_par/$b ]; then cp $f small_gap_q206/fw_cap_par/$b; echo "copied $b from $d"; fi
  done
done
(cd small_gap_q206 && PYTHONIOENCODING=utf-8 py q1483_gpu_events_transfer.py; echo "gpu transfer exit $?")   # R1-R3 of amendment 2186: GPU-route events enter fw_cap_par only if no chain event exists; prints differences only
(cd small_gap_q206 && PYTHONIOENCODING=utf-8 py q1442_merge_cap_events.py; echo "merge exit $?")
cd centre_scripts
export PYTHONIOENCODING=utf-8
py q1413_events_to_tower_files.py 2>&1 | tail -n 1
py q1204_table_test_reading.py > ../$A/q1204_out_$TAG.txt 2>&1; echo "q1204 exit $?"
py q1215_derived_table_test_reading.py > ../$A/q1215_out_$TAG.txt 2>&1; echo "q1215 exit $?"
py q1422_blind_items_verdict.py > ../$A/q1422_out_$TAG.txt 2>&1; echo "q1422 exit $?"
py q1424_stage_bracket_verdict.py > ../$A/q1424_out_$TAG.txt 2>&1; echo "q1424 exit $?"
py q1439_cap_b20_verdict.py > ../$A/q1439_out_$TAG.txt 2>&1; echo "q1439 exit $?"
tail -n 1 ../$A/q1422_out_$TAG.txt; tail -n 1 ../$A/q1424_out_$TAG.txt; tail -n 1 ../$A/q1439_out_$TAG.txt
py q1451_cap_b20_ext_verdict.py > ../$A/q1451_out_$TAG.txt 2>&1; echo "q1451 exit $?"; tail -n 3 ../$A/q1451_out_$TAG.txt | cut -c1-250
py q1527_cap_b131_verdict.py > ../$A/q1527_out_$TAG.txt 2>&1; echo "q1527 exit $?"; tail -n 3 ../$A/q1527_out_$TAG.txt | cut -c1-250
