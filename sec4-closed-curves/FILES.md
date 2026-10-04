# Files in sec4-closed-curves

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/B_ana.py` | analysis helpers, mpmath only, no file writes at import. | `B_ana.py` |
| `scripts/B_event.py` | the linear-response (LR) event for one ring. | `B_event.py` |
| `scripts/B_formula.py` | the leading-order correction to the copying-ring event, term by term, from two B_lr runs at the copying-ring eps (M = K/2 and M' = K/2 + 1) and one M = K/2 run at eps e^DX (for the slope). | `B_formula.py` |
| `scripts/B_lr.py` | q790's full formal density, copied AS TEXT (q790 itself untouched), plus the linear-response matrix L[i][j] = d(KKT cosine harmonic 2i)/d(density cosine coefficient c_2j) at the converged density, by a one-sided probe 1e-30 at 256 bits (q79 | `B_lr.py` |
| `scripts/C_da_solver.py` | an independent solver for mass-constrained deterministic annealing (Rose, Gurewitz, Fox, PRL 65 (1990) 945) with squared-error distortion and codevectors confined to the ellipse z(t) = (RP cos t, RM sin t), source Y ~ N(0, S^2 I), temperatu | `C_da_solver.py` |
| `scripts/C_predict.py` | the copying-ring rule (q795 form) in deterministic-annealing language. | `C_predict.py` |
| `scripts/D_core.py` | the least-favourable-prior KKT system of q786_lfp_wallchain.py, copied as text into importable functions (no top-level computation, no file writes). | `D_core.py` |
| `scripts/D_core2.py` | D_core generalised to three KKT functions and the mixed curve. | `D_core2.py` |
| `scripts/D_diag.py` | step 1, diagnosis at the converged eps 0.1723 twelve-ring (lfp10/t6, rm 0.827708514103). | `D_diag.py` |
| `scripts/D_quad2.py` | a second quadrature for the capacity KKT function, independent of tensor Gauss-Hermite. | `D_quad2.py` |
| `scripts/D_solver.py` | a solver for the least-favourable-prior ring on the ellipse past the flat-KKT wall. | `D_solver.py` |
| `scripts/D_track.py` | continuation of one ring structure in rm with D_solver's harmonic-coordinate Newton, and the tracker's event test copied from q788 (max of D - C on NSCAN + 1 points of [0, 90] deg above TOLV = KKT lost; | `D_track.py` |
| `scripts/E_score.py` | measured near-end shortfalls of an arc state against (a) a 1D optimum at the same half-length (q567-format JSON) and (b) the record's capacity half-line depths (centre_scripts/c50_cap_D100_norefit.json, last stage), and the registered test: | `E_score.py` |
| `scripts/F_core.py` | rigorous ball-arithmetic evaluation of the capacity KKT function on the ellipse curve. | `F_core.py` |
| `scripts/F_kkt.py` | KKT system with rigorous enclosures, Krawczyk existence test, and the curve certificate. | `F_kkt.py` |
| `scripts/F_small.py` | end-to-end certificate on a small record state. | `F_small.py` |
| `scripts/F_small_yz.py` | end-to-end certificate on a small record state. | `F_small_yz.py` |
| `scripts/G_arb18.py` | the rp 3 eighteen-ring X + 4G in ball arithmetic, past the K 16 Y split (Amendments 1348/1349). | `G_arb18.py` |
| `scripts/G_probe.py` | coarse probes and bisection for a tracked ring (coordinator's pace note, 06:42), with the Solver (harmonic Newton; | `G_probe.py` |
| `scripts/G_track22.py` | the tracker (D_track.cmd_track, imported, unchanged) with richer reads at every stored state (coordinator's rule after the K 20 tracker missed a 1e-21 bump): D - C on 85..90 deg in 0.05-deg steps (101 points, written to FINEDIR/fine_rm<rm>. | `G_track22.py` |
| `scripts/G_trackE.py` | the tracker (D_track.cmd_track, imported, unchanged) with reads at both vertices at every stored state: D - C at t = 0 and t = pi/2, D - C on 0..5 deg and 85..90 deg in 0.05-deg steps (written to FINEDIR/fine_rm<rm>.json), the maxima of bot | `G_trackE.py` |
| `scripts/H_arb.py` | seeds and reads for the mixed curve (LAM 0.02, RP 1.2) in ball arithmetic. | `H_arb.py` |
| `scripts/H_track.py` | continuation of one ring in rm on the mixed curve, adapted from the D_track.cmd_track (copied as text; | `H_track.py` |
| `scripts/d90_mpD.py` | the KKT function D of d89 in mpmath, for checks below double precision. | `centre_scripts/d90_mpD.py` |
| `scripts/q719_wallchain_arb.py` | the D2-orbit wall-chain solve of q708/q712 in ball arithmetic (python-flint arb), with the separable Gauss-Hermite rule of q718: for a query point q, Ex[i, j] = exp(-(qx + T_i - x_j)^2/2), Ey[j, l] = exp(-(qy + T_l - y_j)^2/2), p = Ex diag( | `centre_scripts/q719_wallchain_arb.py` |
| `scripts/q786_lfp_wallchain.py` | the least favourable prior for a normal mean known to lie on the ellipse (unit noise, squared-error loss). | `centre_scripts/q786_lfp_wallchain.py` |
| `scripts/q790_full_numeric.py` | the full formal smooth density on the real ellipse (all orders in eps), by Newton on its Fourier coefficients. | `centre_scripts/q790_full_numeric.py` |
| `scripts/q795_copy_ring_predict.py` | the copying-ring form of the comb rule, parameter-free. | `centre_scripts/q795_copy_ring_predict.py` |
| `scripts/q808_b36_check.py` | my check of the flank 32-ring (BACKLOG 36) with the record's capacity function (q719's D_grad, copied as text from q802), not D's code. | `centre_scripts/q808_b36_check.py` |
| `scripts/q811_szego_series.py` | the Szego loss condition evaluated on the leading-order (extremal-sector) formal density of the ellipse, for every ring size, with no ring solves. | `centre_scripts/q811_szego_series.py` |
| `scripts/q812_szego_full_grid.py` | q811's event finder on the FULL formal density (q790 logs on an eps grid, harmonics to 2M). | `centre_scripts/q812_szego_full_grid.py` |
| `scripts/q813_count_rule.py` | for every finished q790 log in a folder, the count K = 1 + the first odd j with /alpha_j/ >= 1 (alpha_j the Verblunsky coefficients of the full formal density, Levinson-Durbin), with K eps and K eps^(1/2). | `centre_scripts/q813_count_rule.py` |
| `scripts/q814_quadcheck.py` | D - C at chosen angles under capfix (FIX_N, FIX_B) and Gauss-Hermite (NGH), using the D_core2 (unchanged). | `centre_scripts/lam02k20check/q814_quadcheck.py` |
| `scripts/q815_onedlimit.py` | the K/2-convention copying loss for the two smallest rings on the pure ellipse, evaluated toward rm -> 0. | `centre_scripts/q815_onedlimit.py` |
| `scripts/szego_check.py` |  | `scratch/szego_check.py` |

`logs/` holds 31 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
