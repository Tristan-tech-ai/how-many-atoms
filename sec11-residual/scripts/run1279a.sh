#!/bin/bash
# run1279a.sh (8 Oct 2026): data acquisition, K_b 38 to 52, start a0 = a_med - 0.004 (q1279_extend_prepare.py)
cd "./centre_scripts"
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 38 19 160 29.565473 0.003 u1279_K38 > q607_u1279_K38.log 2>&1
py q622_c19_hlseed.py 38 19 160 29.204484 0.003 u1279_K38 > c19_u1279_K38.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 39 19 160 30.158848 0.003 u1279_K39 > q607_u1279_K39.log 2>&1
py q622_c19_hlseed.py 39 19 160 29.797203 0.003 u1279_K39 > c19_u1279_K39.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 40 20 160 30.748498 0.003 u1279_K40 > q607_u1279_K40.log 2>&1
py q622_c19_hlseed.py 40 20 160 30.386218 0.003 u1279_K40 > c19_u1279_K40.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 41 20 160 31.334537 0.003 u1279_K41 > q607_u1279_K41.log 2>&1
py q622_c19_hlseed.py 41 20 160 30.971641 0.003 u1279_K41 > c19_u1279_K41.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 42 21 160 31.917074 0.003 u1279_K42 > q607_u1279_K42.log 2>&1
py q622_c19_hlseed.py 42 21 160 31.553581 0.003 u1279_K42 > c19_u1279_K42.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 43 21 160 32.496213 0.003 u1279_K43 > q607_u1279_K43.log 2>&1
py q622_c19_hlseed.py 43 21 160 32.132138 0.003 u1279_K43 > c19_u1279_K43.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 44 22 160 33.072049 0.003 u1279_K44 > q607_u1279_K44.log 2>&1
py q622_c19_hlseed.py 44 22 160 32.707410 0.003 u1279_K44 > c19_u1279_K44.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 45 22 160 33.644676 0.003 u1279_K45 > q607_u1279_K45.log 2>&1
py q622_c19_hlseed.py 45 22 160 33.279489 0.003 u1279_K45 > c19_u1279_K45.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 46 23 160 34.214181 0.003 u1279_K46 > q607_u1279_K46.log 2>&1
py q622_c19_hlseed.py 46 23 160 33.848460 0.003 u1279_K46 > c19_u1279_K46.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 47 23 160 34.780649 0.003 u1279_K47 > q607_u1279_K47.log 2>&1
py q622_c19_hlseed.py 47 23 160 34.414408 0.003 u1279_K47 > c19_u1279_K47.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 48 24 160 35.344158 0.003 u1279_K48 > q607_u1279_K48.log 2>&1
py q622_c19_hlseed.py 48 24 160 34.977411 0.003 u1279_K48 > c19_u1279_K48.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 49 24 160 35.904786 0.003 u1279_K49 > q607_u1279_K49.log 2>&1
py q622_c19_hlseed.py 49 24 160 35.537545 0.003 u1279_K49 > c19_u1279_K49.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 50 25 160 36.462603 0.003 u1279_K50 > q607_u1279_K50.log 2>&1
py q622_c19_hlseed.py 50 25 160 36.094882 0.003 u1279_K50 > c19_u1279_K50.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 51 25 160 37.017680 0.003 u1279_K51 > q607_u1279_K51.log 2>&1
py q622_c19_hlseed.py 51 25 160 36.649490 0.003 u1279_K51 > c19_u1279_K51.log 2>&1
HL_JSON=c33_npm_D92.json D0=-0.22676 NU0=0.14770 py q626_npmle_hlseed.py 52 26 160 37.570084 0.003 u1279_K52 > q607_u1279_K52.log 2>&1
py q622_c19_hlseed.py 52 26 160 37.201435 0.003 u1279_K52 > c19_u1279_K52.log 2>&1
echo done $(date +%H:%M:%S) > run1279a.done
