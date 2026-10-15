#!/bin/bash
# run_k80_split_verify.sh (8 Oct 2026, registered in amendment 1921): the three scans and the verdict q931 of the split K 79 -> 80, exactly as q945_tower.split() runs them
# (same scripts, same arguments, same grid 60.0); the driver q1164 stopped after the amplitude A_s (tower_K80_As.json).
cd "." || exit 1
export PYTHONIOENCODING=utf-8 Q935_GRID=60.0 Q935_ODIR=tower/K80_split
A=scratch
mkdir -p tower/K80_split
py q935_split_generic.py scan tower/K79/results_q220_gauss_K79_A50.9867.json $A/tower_K80_scan_hi.json > $A/tower_K80_scan_hi.log 2>&1 &
P1=$!
py q935_split_generic.py scan tower/K79/results_q220_gauss_K79_A50.9817.json $A/tower_K80_scan_lo.json > $A/tower_K80_scan_lo.log 2>&1 &
P2=$!
py q935_split_generic.py scan tower/K79/results_q220_gauss_K79_A50.9017.json $A/tower_K80_scan_mid.json > $A/tower_K80_scan_mid.log 2>&1 &
P3=$!
wait $P1; R1=$?; wait $P2; R2=$?; wait $P3; R3=$?
echo "scans exit $R1 $R2 $R3 at $(date +%H:%M:%S)"
py q931_split_verdict.py $A/tower_K80_scan_hi.json $A/tower_K80_scan_lo.json $A/tower_K80_scan_mid.json $A/tower_K80_As.json > $A/tower_K80_q931.txt 2>&1
echo "q931 exit $? at $(date +%H:%M:%S): $(tail -n 1 $A/tower_K80_q931.txt | cut -c1-200)"
touch $A/k80_split_verify.done
