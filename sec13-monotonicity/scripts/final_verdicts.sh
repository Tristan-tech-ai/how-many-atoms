#!/bin/bash
# final_verdicts.sh (7 Oct 2026): the full runs of the registered verdict scripts after the family K 79 and the event K_b 79 exist.
# Order: selftest of the readouts, lint of the seven scripts, then each script once; every output is written whole to scratch/<script>_T<n>_out.txt and the last line of each is its verdict.
# It writes no amendment; the result amendments are written by hand from the outputs, by the letter of the registered rules.
cd "." || exit 1
export PYTHONIOENCODING=utf-8; A=scratch
py gate_selftest/selftest_readouts.py 2>&1 | tail -n 1
py gate_selftest/lint_verdict_scripts.py q1084_law_tower_extension4_verdict.py q1099_min_flow_extension_verdict.py q1102_min_flow_density_law_verdict.py q1109_birth_slope_law_verdict.py q1117_event_log_flow_law_verdict.py q1129_lattice_shape_law_verdict.py q1133_flow_profile_collapse_verdict.py q1154_closed_form_k76_79_verdict.py q1155_apriori_lattice_k76_79_verdict.py q1161_centre_signal_slope_k76_78_verdict.py q1163_odd_centre_slope_k77_79_verdict.py q1166_centre_rate_at_event_k76_79_verdict.py q1168_log_flow_offset_law_k77_79_verdict.py q1170_flow_correction_curve_k77_79_verdict.py q1174_two_term_operator_k78_79_verdict.py q1179_first_order_response_k78_79_verdict.py q1181_residual_ratio_k79_verdict.py 2>&1 | tail -n 1
for pair in q1084_law_tower_extension4_verdict:T116 q1099_min_flow_extension_verdict:T121 q1102_min_flow_density_law_verdict:T122 q1109_birth_slope_law_verdict:T124 q1117_event_log_flow_law_verdict:T125 q1129_lattice_shape_law_verdict:T128 q1133_flow_profile_collapse_verdict:T129 q1154_closed_form_k76_79_verdict:T134 q1155_apriori_lattice_k76_79_verdict:T135 q1161_centre_signal_slope_k76_78_verdict:T136 q1163_odd_centre_slope_k77_79_verdict:T137 q1166_centre_rate_at_event_k76_79_verdict:T138 q1168_log_flow_offset_law_k77_79_verdict:T139 q1170_flow_correction_curve_k77_79_verdict:T140 q1174_two_term_operator_k78_79_verdict:T141 q1179_first_order_response_k78_79_verdict:T142 q1181_residual_ratio_k79_verdict:T143; do
  S=${pair%%:*}; T=${pair##*:}
  py $S.py > $A/${S%%_*}_${T}_out.txt 2>&1; rc=$?
  echo "$S $T exit $rc: $(tail -n 1 $A/${S%%_*}_${T}_out.txt | cut -c1-120)"
done
