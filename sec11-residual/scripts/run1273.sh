#!/bin/bash
# run1273.sh (8 Oct 2026): the eight full-chain locators c35 (KEXP 4) for K_b 29 to 36 of amendments 2052 and 2053; starts a0 from q1273_predictions.json
cd "./centre_scripts"
export KEXP=4 HL_JSON=c33_k4c.json HL_STAGE=92 D0=0.70887 NU0=-0.03259
py c35_chain_k.py 29 14 160 23.325231 0.003 u1273_K29 > q607_u1273_K29.log 2>&1
py c35_chain_k.py 30 15 160 23.956570 0.003 u1273_K30 > q607_u1273_K30.log 2>&1
py c35_chain_k.py 31 15 160 24.582845 0.003 u1273_K31 > q607_u1273_K31.log 2>&1
py c35_chain_k.py 32 16 160 25.204252 0.003 u1273_K32 > q607_u1273_K32.log 2>&1
py c35_chain_k.py 33 16 160 25.820977 0.003 u1273_K33 > q607_u1273_K33.log 2>&1
py c35_chain_k.py 34 17 160 26.433192 0.003 u1273_K34 > q607_u1273_K34.log 2>&1
py c35_chain_k.py 35 17 160 27.041058 0.003 u1273_K35 > q607_u1273_K35.log 2>&1
py c35_chain_k.py 36 18 160 27.644725 0.003 u1273_K36 > q607_u1273_K36.log 2>&1
echo done $(date +%H:%M:%S) > run1273.done
