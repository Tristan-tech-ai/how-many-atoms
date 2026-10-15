#!/bin/bash
# run1276.sh (8 Oct 2026): full-chain locators c35 for k = 1.5, 2.5, 5 and K_b 29 to 36 of amendments 2055 and 2056; starts a0 from q1276_predictions.json
cd "./centre_scripts"
export KEXP=1.5 HL_JSON=c33_k15c.json HL_STAGE=92 D0=0.01758 NU0=0.11577
py c35_chain_k.py 29 14 160 23.828184 0.003 u1276_k15_K29 > q607_u1276_k15_K29.log 2>&1
py c35_chain_k.py 30 15 160 24.461075 0.003 u1276_k15_K30 > q607_u1276_k15_K30.log 2>&1
py c35_chain_k.py 31 15 160 25.088839 0.003 u1276_k15_K31 > q607_u1276_k15_K31.log 2>&1
py c35_chain_k.py 32 16 160 25.711679 0.003 u1276_k15_K32 > q607_u1276_k15_K32.log 2>&1
py c35_chain_k.py 33 16 160 26.329783 0.003 u1276_k15_K33 > q607_u1276_k15_K33.log 2>&1
py c35_chain_k.py 34 17 160 26.943326 0.003 u1276_k15_K34 > q607_u1276_k15_K34.log 2>&1
py c35_chain_k.py 35 17 160 27.552473 0.003 u1276_k15_K35 > q607_u1276_k15_K35.log 2>&1
py c35_chain_k.py 36 18 160 28.157378 0.003 u1276_k15_K36 > q607_u1276_k15_K36.log 2>&1
export KEXP=2.5 HL_JSON=c33_k25c.json HL_STAGE=92 D0=0.36935 NU0=0.04508
py c35_chain_k.py 29 14 160 23.566223 0.003 u1276_k25_K29 > q607_u1276_k25_K29.log 2>&1
py c35_chain_k.py 30 15 160 24.198368 0.003 u1276_k25_K30 > q607_u1276_k25_K30.log 2>&1
py c35_chain_k.py 31 15 160 24.825416 0.003 u1276_k25_K31 > q607_u1276_k25_K31.log 2>&1
py c35_chain_k.py 32 16 160 25.447568 0.003 u1276_k25_K32 > q607_u1276_k25_K32.log 2>&1
py c35_chain_k.py 33 16 160 26.065010 0.003 u1276_k25_K33 > q607_u1276_k25_K33.log 2>&1
py c35_chain_k.py 34 17 160 26.677915 0.003 u1276_k25_K34 > q607_u1276_k25_K34.log 2>&1
py c35_chain_k.py 35 17 160 27.286447 0.003 u1276_k25_K35 > q607_u1276_k25_K35.log 2>&1
py c35_chain_k.py 36 18 160 27.890757 0.003 u1276_k25_K36 > q607_u1276_k25_K36.log 2>&1
export KEXP=5.0 HL_JSON=c33_k5c.json HL_STAGE=92 D0=0.87512 NU0=-0.07393
py c35_chain_k.py 29 14 160 23.211376 0.003 u1276_k5_K29 > q607_u1276_k5_K29.log 2>&1
py c35_chain_k.py 30 15 160 23.842288 0.003 u1276_k5_K30 > q607_u1276_k5_K30.log 2>&1
py c35_chain_k.py 31 15 160 24.468153 0.003 u1276_k5_K31 > q607_u1276_k5_K31.log 2>&1
py c35_chain_k.py 32 16 160 25.089166 0.003 u1276_k5_K32 > q607_u1276_k5_K32.log 2>&1
py c35_chain_k.py 33 16 160 25.705511 0.003 u1276_k5_K33 > q607_u1276_k5_K33.log 2>&1
py c35_chain_k.py 34 17 160 26.317360 0.003 u1276_k5_K34 > q607_u1276_k5_K34.log 2>&1
py c35_chain_k.py 35 17 160 26.924873 0.003 u1276_k5_K35 > q607_u1276_k5_K35.log 2>&1
py c35_chain_k.py 36 18 160 27.528199 0.003 u1276_k5_K36 > q607_u1276_k5_K36.log 2>&1
echo done $(date +%H:%M:%S) > run1276.done
