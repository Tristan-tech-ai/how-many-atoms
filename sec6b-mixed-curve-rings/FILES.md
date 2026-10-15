# Files in sec6b-mixed-curve-rings

Scripts are formatted with black (layout only). The original path is the name used in the registration log. The last column gives the sections of the paper whose sources name the script.

| script | what it does (from its own docstring) | original path | cited by |
|---|---|---|---|
| `scripts/D_core2.py` | D_core generalised to three KKT functions and the mixed curve. | `D_core2.py` | VIII |
| `scripts/D_solver.py` | a solver for the least-favourable-prior ring on the ellipse past the flat-KKT wall. | `D_solver.py` | VIII |
| `scripts/D_track.py` | continuation of one ring structure in rm with D_solver's harmonic-coordinate Newton, and the tracker's event test copied from q788 (max of D - C on NSCAN + 1 points of [0, 90] deg above TOLV = KKT lost; | `D_track.py` |  |
| `scripts/I_lfp_centre.py` | centre excess r(0) - C of a solved LFP boundary ring (a helper script state json from D_solver, FUNC lfp), with D_core2.Ring's own risk function (_D_lfp, the record's q786 text) at q = (0, 0); | `I_lfp_centre.py` | VIII |
| `scripts/I_rd_centre.py` | TEXT COPY of I_lfp_centre for FUNC rd (SRC_S from the state). | `I_rd_centre.py` | VIII |
| `scripts/c35_chain_k.py` | c35 = q626_npmle_hlseed.py copied as text (8 Oct 2026, amendment 2046); | `centre_scripts/c35_chain_k.py` | S5, S6 |
| `scripts/kkt_scan_q607.py` | scan of the NPMLE optimality function for the stored interval states of the extended test (centre_scripts/q607_u1207_K*.json, amendment 2040) that hold every atom. | `gate_truth/kkt_scan_q607.py` | VIII |
| `scripts/make_q1889.py` | writes q1889 (LFP, amendment 1461) and mix_rm_walk_lfp.py as text copies of q1888 and mix_rm_walk.py; | `small_gap_q206/make_q1889.py` | VIII |
| `scripts/mix_lam_continuation2.py` | continuation in the curve parameter LAM with a secant predictor, from the NPMLE ellipse ring (rd, rp 2.1, K 20, rm 1.7416) towards the mixed curve LAM 0.2 at fixed rm. | `mix_lam_continuation2.py` | VIII |
| `scripts/mix_lam_continuation_lfp.py` | continuation in the curve parameter LAM with a secant predictor, from the NPMLE ellipse ring (rd, rp 2.1, K 20, rm 1.7416) towards the mixed curve LAM 0.2 at fixed rm. | `mix_lam_continuation_lfp.py` | VIII |
| `scripts/mix_lam_continuation_lfp2.py` | continuation in the curve parameter LAM with a secant predictor, from the NPMLE ellipse ring (rd, rp 2.1, K 20, rm 1.7416) towards the mixed curve LAM 0.2 at fixed rm. | `mix_lam_continuation_lfp2.py` | VIII |
| `scripts/mix_lam_continuation_lfp3.py` | continuation in the curve parameter LAM with a secant predictor, from the NPMLE ellipse ring (rd, rp 2.1, K 20, rm 1.7416) towards the mixed curve LAM 0.2 at fixed rm. | `mix_lam_continuation_lfp3.py` | VIII |
| `scripts/mix_lam_continuation_lfp4.py` | continuation in the curve parameter LAM with a secant predictor, from the NPMLE ellipse ring (rd, rp 2.1, K 20, rm 1.7416) towards the mixed curve LAM 0.2 at fixed rm. | `mix_lam_continuation_lfp4.py` | VIII |
| `scripts/mix_rm_walk.py` | walk the positive NPMLE ring on the mixed curve (rd, s 2, rp 2.1, LAM 0.2, K 20, 200 Gauss-Hermite nodes, 160 bits) from rm 1.7416 down through the band [1.73615, 1.73655] registered in amendment 1463, with D_solver.py newton (tangent predi | `mix_rm_walk.py` | VIII |
| `scripts/mix_rm_walk_lfp.py` | walk the positive LFP ring on the mixed curve (lfp, rp 1.7, LAM 0.1, X1 Y1 G3, 120 Gauss-Hermite nodes, 128 bits) from rm 1.4036 down through the band [1.40137, 1.40177] of 1461, with D_solver.py newton (tangent predictor for the first step | `mix_rm_walk_lfp.py` | VIII |
| `scripts/mix_rm_walk_x0.py` | continue the positive NPMLE ring on the mixed curve (rd, s 2, rp 2.1, LAM 0.2, 200 nodes, 160 bits) past the death of its vertex pair. | `mix_rm_walk_x0.py` | VIII |
| `scripts/q1888_mixed_rd_birth_positive_rings_verdict.py` | the registered test of amendment 1463 applied to NPMLE rings on the mixed curve whose masses are all positive. | `small_gap_q206/q1888_mixed_rd_birth_positive_rings_verdict.py` | VIII |
| `scripts/q1889_mixed_lfp_birth_positive_rings_verdict.py` | the registered test of amendment 1461 applied to LFP rings on the mixed curve whose masses are all positive. | `small_gap_q206/q1889_mixed_lfp_birth_positive_rings_verdict.py` | VIII |
| `scripts/resolve_q607_odd.py` | re-solve the odd-K_b located NPMLE interval states of amendment 2040 with their centre mass, which the stored records (centre_scripts/q607_u1207_K*.json) do not hold. | `gate_truth/resolve_q607_odd.py` | VIII |

`logs/` holds 12 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
