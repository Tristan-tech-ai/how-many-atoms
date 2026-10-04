# Files in sec3-one-dimension

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/cap_beta.py` | beta(A) = 1/c* - 2A = e^{h(Y)} - 2A on stored Gaussian capacity states (pred_cap_beta.txt). | `cap_beta.py` |
| `scripts/cap_edge_pointwise.py` | Step checks for the pointwise capacity edge law (pred_cap_edge_pointwise.txt). | `cap_edge_pointwise.py` |
| `scripts/q470_support_certificate.py` | T2 of 613(7)/614(4) - a rigorous SUPPORT-SIZE certificate for one capacity-achieving state of the amplitude-constrained Poisson channel, in ball arithmetic (python-flint arb). | `q470_support_certificate.py` |
| `scripts/q484_lfp_bnm.py` | the HIGHER-TARGET cheapest test (23 Sep). | `q484_lfp_bnm.py` |
| `scripts/q486_lfp_sym.py` | least-favourable priors for the bounded normal mean (squared error), SYMMETRIC continuation with the two centre transitions handled explicitly, so that every transition is located by bisection on a sign condition. | `q486_lfp_sym.py` |
| `scripts/q487_lfp_forward.py` | FROZEN forward predictions of the least-favourable-prior transitions for 18.7 < m <= 30 (bounded normal mean, squared error), written before q486 reaches that range. | `q487_lfp_forward.py` |
| `scripts/q489_lfp_certificate.py` | a rigorous SUPPORT-SIZE certificate for the least-favourable prior (LFP) of the bounded normal mean under squared error, in ball arithmetic (python-flint arb/acb). | `q489_lfp_certificate.py` |
| `scripts/q493_compare_mp.py` | the 631(6) forward test on the MULTIPRECISION transitions of q492 (results_q492_lfp_mpcont.json), against the frozen predictions (lfp_forward_predictions.json, read only, restored per amendment 634). | `q493_compare_mp.py` |
| `scripts/q494_freeze_30_40.py` | FROZEN forward predictions of the LFP transitions in (30, 40], written BEFORE q492 reaches m = 30, from the 04:47:51 frozen fit read from lfp_forward_predictions.json (T, d0, rho0, MLNM (c, b), FREEP (c, p, b); | `q494_freeze_30_40.py` |
| `scripts/q495_score_40_50.py` | scoring of the THIRD frozen LFP forward test on (40, 50] (pred_lfp_40_50.txt) against the multiprecision transitions of q492 (results_q492_lfp_mpcont.json), and of the frozen offset extrapolation (pred_lfp_offset_extrap.txt). | `q495_score_40_50.py` |
| `scripts/q496_npmle_uniform.py` | number of atoms K(a) of the POPULATION NPMLE of a Gaussian location mixture when the data density is uniform on [-a, a]. | `q496_npmle_uniform.py` |
| `scripts/q497_npmle_freeze.py` | FROZEN forward predictions of the population-NPMLE change points (uniform data on [-a, a]) beyond a = 33.65, written before any continuation goes past 33.65 (q496's arb run stopped there). | `q497_npmle_freeze.py` |
| `scripts/q498_npmle_mpcont.py` | arb continuation of the population NPMLE (Gaussian location mixture, data uniform on [-a, a]) past a = 33.65, where the arb run of q496 (mpsweep) stopped. | `q498_npmle_mpcont.py` |

`logs/` holds 10 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
