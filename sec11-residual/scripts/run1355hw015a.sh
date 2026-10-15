#!/bin/bash
# run1355hw015a.sh (8 Oct 2026): hard-wall chain c61 for k = 0.15, K_b 14-23; start A0 = a_med - 0.0045, step 0.003 (q1355_predictions.json)
cd "./centre_scripts"
export KEXP=0.15 HL_JSON=c52_hw015c.json HL_STAGE=92 LAW0=-0.05325,0.58551 WM0=3.606972
py c61_hw_fullchain.py 14 12.862593 0.003 160 u1355_hw015_K14 > c61_u1355_hw015_K14.log 2>&1
py c61_hw_fullchain.py 15 13.626483 0.003 160 u1355_hw015_K15 > c61_u1355_hw015_K15.log 2>&1
py c61_hw_fullchain.py 16 14.377473 0.003 160 u1355_hw015_K16 > c61_u1355_hw015_K16.log 2>&1
py c61_hw_fullchain.py 17 15.116587 0.003 160 u1355_hw015_K17 > c61_u1355_hw015_K17.log 2>&1
py c61_hw_fullchain.py 18 15.844713 0.003 160 u1355_hw015_K18 > c61_u1355_hw015_K18.log 2>&1
py c61_hw_fullchain.py 19 16.562626 0.003 160 u1355_hw015_K19 > c61_u1355_hw015_K19.log 2>&1
py c61_hw_fullchain.py 20 17.271006 0.003 160 u1355_hw015_K20 > c61_u1355_hw015_K20.log 2>&1
py c61_hw_fullchain.py 21 17.970458 0.003 160 u1355_hw015_K21 > c61_u1355_hw015_K21.log 2>&1
py c61_hw_fullchain.py 22 18.661521 0.003 160 u1355_hw015_K22 > c61_u1355_hw015_K22.log 2>&1
py c61_hw_fullchain.py 23 19.344675 0.003 160 u1355_hw015_K23 > c61_u1355_hw015_K23.log 2>&1
echo done $(date +%H:%M:%S) > run1355hw015a.done
