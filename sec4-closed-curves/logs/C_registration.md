# Registration: critical temperatures of mass-constrained deterministic annealing on a fixed ellipse (copying-ring rule)

Registered 02 Oct 2026 18:29 (clock: `date` printed 18:29:22 just before writing). No deterministic-annealing run of any kind (grid stage,
polish or trace) has been made at this geometry before this file was written. The only DA runs so far: the validation polish at
rp 1.0, rm 0.711935124688, s 2, T 2 (one temperature, step 1 of C_da_bridge.md) and a machinery test at an unrelated geometry
(rp 1.0, rm 0.5, s 1.5, T 2.0 -> 0.9, C_runs/test/).

## Problem in DA language
Source Y ~ N(0, s^2 I), codevectors y_j on the ellipse z(t) = (rp cos t, rm sin t), squared-error distortion, temperature T = 1/beta,
free energy F = -T E_Y log sum_j m_j exp(-|Y - y_j|^2 / T). Component width sigma = sqrt(T/2); T = 2 is the unit-variance NPMLE.
Geometry (fixed): rp 1.0, rm 0.7 (eps 0.3), s 2. Cooling from T near 3.5 to T near 1.5.

## How the predictions were made
C_predict.py (sha256 4a61b73d...): at each T, q790 (sha256 54d6fe99..., unchanged; FUNC rd, NA 96, DEG 6, RMAX 20, 256 bits, LAM 0, SHAPE
unset) is run on the scaled problem (rp/sigma, eps/sigma, SRC_S = s/sigma) for the full smooth density c_2..c_K; the copying ring (q795's
moment solve copied as text: the D2-symmetric K-ring with b_j = c_j for j = 2..K-2) gives b_K; the K-ring is lost on cooling at the T
where |c_K| = |b_K| (secant in ln T, |ln(c/b)| < 1e-10).
Checks made before registering: (a) at T 2 the wrapper reproduces q795's registered NPMLE s 2 K 12 crossing (eps 0.287664565: ln|c/b|
= -5.7e-9); (b) at the predicted T_c(8), q790 on a finer grid (NA 128, DEG 7) gives ln|c/b| = -1.0e-9 (shift in T near 2e-10).

## Signs and layouts
c_K > 0 for every K = 6..16 at every scanned T (1.5-5.0; logs C_runs/pred_scan_K*_vertex.log). So every ring is in the vertex phase:
an atom at t = 0 (the major vertex, X orbit) in every ring; the flank layouts would carry b_K < 0 against c_K > 0 (inconsistent).
Every count change happens at the MINOR vertex (t = 90 deg): K = 2 mod 4 rings (X + G orbits, no Y) are lost by a Y birth, K = 0 mod 4
rings (X + Y + G orbits) by a Y split. Predicted sequence on cooling:
X + G (6) -> Y birth -> X + Y + G (8) -> Y split -> X + 2G (10) -> Y birth -> X + Y + 2G (12) -> Y split -> X + 3G (14) -> Y birth
-> X + Y + 3G (16) -> Y split -> X + 4G (18).

## Registered predictions (copying-ring form; obs/pred - 1 band +-1 % in T_c for K >= 8)
| K before | ring before (masses per atom; G deg, at the predicted crossing) | T_c | beta_c | sigma_c | c_K = b_K | kind and place | scored |
| 6 | X + G (0.288533, 0.105733; 67.132) | 2.958603753 | 0.3379972728 | 1.216265545 | 1.7748277 | Y birth at t = 90 deg | reported only (smallest ring) |
| 8 | X + Y + G (0.248569, 0.0948489, 0.0782909; 51.105) | 2.403468696 | 0.4160653315 | 1.096236447 | 1.7858966 | Y split at t = 90 deg | yes |
| 10 | X + 2G (0.219768, 0.0668359, 0.0732803; 41.636, 75.016) | 2.127813942 | 0.4699659028 | 1.031458662 | 1.6821407 | Y birth at t = 90 deg | yes |
| 12 | X + Y + 2G (0.189563, 0.0701451, 0.05985, 0.060296; 33.496, 62.356) | 1.896254868 | 0.527355271 | 0.9737183546 | 1.8197617 | Y split at t = 90 deg | yes |
| 14 | X + 3G (0.172873, 0.057151, 0.0523951, 0.0540175; 29.661, 55.738, 79.411) | 1.784631078 | 0.5603399003 | 0.9446245491 | 1.5269514 | Y birth at t = 90 deg | yes |
| 16 | X + Y + 3G (0.142722, 0.0576623, 0.0539858, 0.0454463, 0.0503755; 23.807, 45.982, 67.716) | 1.601383168 | 0.6244601665 | 0.8948137148 | 1.9572405 | Y split at t = 90 deg | yes, if the solver reaches it |
K 4: no prediction. The X + Y ring has b_4 = 2 identically and c_4 stays below about 1.8 for every T up to 5.7, so the form gives no
crossing (the smallest rings are outside the rule's regime, as in the record). The state above T_c(6) is therefore not predicted beyond
"the six-ring X + G is the optimum just above 2.9586".

## Known before registering (consistency, not test)
At T = 2 (sigma 1) this geometry is the NPMLE s 2 rp 1.0 point eps 0.30 of the q792 tracker (Amendments 1292-1308): the twelve-ring
X + Y + 2G is the optimum there (K 10 lost at eps 0.314769, K 12 at 0.288020). So T_c(10) > 2 > T_c(12) is already known; the values,
kinds and places of all events in T are not.

## Protocol for step 3 (fixed now)
C_da_solver.py (sha256 14de578f...): grid stage (float64 Blahut-Arimoto fixed point on 4096 curve points) at T 3.5, clustered and polished;
then the cooling trace at NGH 300, 128 bits, step 0.01 in ln T, each event located by bracketing the first event function to turn positive
(Illinois in ln T, bracket < 1e-10). Observed T_c = midpoint of the final bracket. One event re-traced at NGH 400 as a quadrature check.
Scored for each event K = 8..16: (a) value: obs/pred - 1 within +-1 %; (b) kind (birth or split) and resolving vertex (the orbit born or split,
and its angle); (c) reported: the copying test (the pre-event ring's b_j against the full c_j at the observed T_c, j < K).
FAIL modes named now: a value outside +-1 %; an event at the major vertex or in a G orbit; a count change not in the sequence above
(e.g. a G birth between atoms, a split of the X atom); the start state or any traced state violating KKT.
Gates: no gate word used.
