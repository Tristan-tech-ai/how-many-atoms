#!/bin/bash
# chain_check.sh (10 Oct 2026): one chain check in the right order, from any directory.
# Order: undelivered_events.py (names only; exits 3 on a zero count) -> check_progress.sh (counts) -> imm_merge.py (merge, differences only) -> q1624_kappa_check.py (labels or NO EVENT) -> chain_due.py mark.
# The mark is written only if the helper counted non-zero merged events. --selftest runs the helper from an empty temporary directory and expects the count to be non-zero anyway (the helper changes into its own directory), then checks that a helper with a bad path would stop.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
if [ "$1" == "--selftest" ]; then
  T="$(mktemp -d)"; cd "$T"; out="$(py "$HERE/undelivered_events.py" 2>&1)"; echo "$out" | head -n 3
  echo "$out" | grep -q "merged events: 0" && { echo "SELFTEST FAILED: zero merged events from a foreign directory"; exit 4; }
  echo "$out" | grep -q "capacity merged events: [1-9]" || { echo "SELFTEST FAILED: no capacity count"; exit 4; }
  echo "SELFTEST helper counts non-zero from a foreign directory: ok"; exit 0
fi
cd "$ROOT"
if [ "$1" != "--force" ] && ! py chain_due.py | grep -q "^DUE"; then echo "chain check not due: nothing done (use --force to override)"; exit 0; fi
py "$HERE/undelivered_events.py" || { echo "helper failed: no mark"; exit 3; }
bash "$HERE/check_progress.sh" 2>&1 | tail -n 22 | cut -c1-240
( cd "$HERE" && py imm_merge.py --verdict 2>&1 | tail -n 5 | cut -c1-240 )
( cd "$HERE" && py q1624_kappa_check.py 2>&1 | tail -n 4 | cut -c1-240 )
( cd "$HERE" && py q1642_births_136_138.py 2>&1 | tail -n 4 | cut -c1-240 )
( cd "$HERE" && py q1668_births_136_138_labels.py 2>&1 | tail -n 4 | cut -c1-240 )
( cd "$HERE" && py q1682_cells_136_labels.py 2>&1 | tail -n 5 | cut -c1-240 )
( cd "$HERE" && py q1685_events_139_150_labels.py 2>&1 | tail -n 14 | cut -c1-240 )
( cd "$HERE" && py q1691_aitken_labels.py 2>&1 | tail -n 14 | cut -c1-240 )
py chain_due.py mark | tail -n 1
