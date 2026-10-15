#!/bin/bash
# run1271.sh (8 Oct 2026): the ten full-chain locators c35 (KEXP 3) for K_b 28 to 37 of amendments 2046 to 2049; starts a0 from q1271_predictions.json
cd "./centre_scripts"
export KEXP=3 HL_JSON=c33_k3c.json HL_STAGE=92 D0=0.50382 NU0=0.01310
py c35_chain_k.py 28 14 160 22.835484 0.003 u1271_K28 > q607_u1271_K28.log 2>&1
py c35_chain_k.py 29 14 160 23.472630 0.003 u1271_K29 > q607_u1271_K29.log 2>&1
py c35_chain_k.py 30 15 160 24.104477 0.003 u1271_K30 > q607_u1271_K30.log 2>&1
py c35_chain_k.py 31 15 160 24.731239 0.003 u1271_K31 > q607_u1271_K31.log 2>&1
py c35_chain_k.py 32 16 160 25.353115 0.003 u1271_K32 > q607_u1271_K32.log 2>&1
py c35_chain_k.py 33 16 160 25.970292 0.003 u1271_K33 > q607_u1271_K33.log 2>&1
py c35_chain_k.py 34 17 160 26.582942 0.003 u1271_K34 > q607_u1271_K34.log 2>&1
py c35_chain_k.py 35 17 160 27.191227 0.003 u1271_K35 > q607_u1271_K35.log 2>&1
py c35_chain_k.py 36 18 160 27.795300 0.003 u1271_K36 > q607_u1271_K36.log 2>&1
py c35_chain_k.py 37 18 160 28.395302 0.003 u1271_K37 > q607_u1271_K37.log 2>&1
echo done $(date +%H:%M:%S) > run1271.done
