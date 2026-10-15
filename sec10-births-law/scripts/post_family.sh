#!/bin/bash
# post_family.sh K (7 Oct 2026): the standard steps after the family K of the tower (K >= 74) is complete.
# 1. the family list (mk_family_list.py refuses without the event file of the end of the family);
# 2. the readings of the parity (q953 for odd K, q954 for even K) into scratch/tower_K<K>_readings.txt;
# 3. the g reading q1114 with the upper bound K + 1 into q1114_K<K>_out.txt;
# 4. the registered verdict scripts T121 (q1099), T122 (q1102), T124 (q1109), T125 (q1117), T128 (q1129), T129 (q1133) and T116 (q1084), each into <script>_partial_after_K<K>_out.txt (their rules give INCOMPLETE until all six families exist and FAIL at the first failed item).
# Nothing here writes an amendment; the amendment is written by hand after the outputs are read in full.
set -u
cd "." || exit 1
K=$1; A=scratch
if [ -f $A/tower_family_list_K$K.txt ]; then echo "list exists (written by the driver family readouts): $(wc -l < $A/tower_family_list_K$K.txt) states"; else py mk_family_list.py $K || { echo "STOP: no list written"; exit 2; }; fi
if [ $((K % 2)) -eq 0 ]; then R=q954_readings_even.py; else R=q953_readings_odd.py; fi
PYTHONIOENCODING=utf-8 py $R $K tower/K$K $A/tower_family_list_K$K.txt > $A/tower_K${K}_readings.txt 2>&1; echo "readings: $(tail -n 1 $A/tower_K${K}_readings.txt | cut -c1-80)"
PYTHONIOENCODING=utf-8 py q1114_log_flow_along_family_reading.py $((K + 1)) > $A/q1114_K${K}_out.txt 2>&1; echo "q1114: $(grep -c 'linear extrapolation' $A/q1114_K${K}_out.txt) families"
for S in q1099_min_flow_extension_verdict q1102_min_flow_density_law_verdict q1109_birth_slope_law_verdict q1117_event_log_flow_law_verdict q1129_lattice_shape_law_verdict q1133_flow_profile_collapse_verdict q1084_law_tower_extension4_verdict q1154_closed_form_k76_79_verdict q1155_apriori_lattice_k76_79_verdict q1161_centre_signal_slope_k76_78_verdict q1163_odd_centre_slope_k77_79_verdict q1166_centre_rate_at_event_k76_79_verdict q1168_log_flow_offset_law_k77_79_verdict q1170_flow_correction_curve_k77_79_verdict q1174_two_term_operator_k78_79_verdict q1179_first_order_response_k78_79_verdict; do
  PYTHONIOENCODING=utf-8 py $S.py > $A/${S%%_*}_partial_after_K${K}_out.txt 2>&1; echo "$S: $(tail -n 1 $A/${S%%_*}_partial_after_K${K}_out.txt | cut -c1-110)"
done
