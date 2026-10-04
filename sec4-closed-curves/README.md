# Section IV: closed curves

On a closed curve near a circle the optimal atoms copy a smooth "formal density": they sit at the nodes of its Szegő
(para-orthogonal) quadrature, and a ring of K atoms is lost where the Verblunsky coefficient α_{K−1} of the density reaches ±1.

| paper item | scripts | outputs in `logs/` |
|---|---|---|
| the formal density (Section II) | `q790_full_numeric.py` | |
| registered loss values, form S (Tables II and III) | `q795_copy_ring_predict.py` | |
| the identity c_K − b_K = 2(α_{K−1} − τ)∏(1 − α_j²) | `szego_check.py` | |
| the linear-response correction (LR) | `B_lr.py`, `B_event.py`, `B_formula.py` | `B_out__fam_rp3_K26*.log` |
| tracking the optimal ring and bisecting its loss | `D_solver.py` (with `D_core*.py`), `H_track.py`, `H_arb.py`, `G_track22.py`, `G_probe.py` | `G_work__e34__*.log`, `G_work__e36__*.log`, `H_work__*.log` |
| certified small rings on the ellipse r_p = 3 | `F_small.py`, `F_small_yz.py` | `F_work__K24_*_cert.log` |
| scoring against the registered bands | `E_score.py` | |
| the count as the curve becomes a circle (Fig. 1) | `q811_szego_series.py`, `q812_szego_full_grid.py`, `q813_count_rule.py` | `q811__R1.0_K320_dps*.log` |
| the segment limit (first two scalar changes) | `q815_onedlimit.py` | |
| quadrature check of the mixed-curve 20-ring | `q814_quadcheck.py` | |
| NPMLE and LFP ring losses (Table II) | `D_solver.py` | `rd2__*`, `lfp10__*` |
| deterministic annealing ring losses (Table II) | `C_predict.py`, `C_da_solver.py` | `C_registration.md`, `C_runs__main__grid_T3.5.log`, `da_check__*` |

`D_solver.py` handles the three problems (`FUNC cap`, `FUNC lfp`, `FUNC rd`). The 36-ring states at r_m = 0.9690 and 0.9692 in
`G_work__e36__solve_*_p448.log` are the two 448-bit states cited in Section IV-D.

See [`FILES.md`](FILES.md) for every script.
