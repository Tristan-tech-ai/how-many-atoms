# Files in sec6-filled-regions

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/D_core.py` | the least-favourable-prior KKT system of q786_lfp_wallchain.py, copied as text into importable functions (no top-level computation, no file writes). | `D_core.py` |
| `scripts/D_core2.py` | D_core generalised to three KKT functions and the mixed curve. | `D_core2.py` |
| `scripts/D_diag.py` | step 1, diagnosis at the converged eps 0.1723 twelve-ring (lfp10/t6, rm 0.827708514103). | `D_diag.py` |
| `scripts/D_solver.py` | a solver for the least-favourable-prior ring on the ellipse past the flat-KKT wall. | `D_solver.py` |
| `scripts/D_track.py` | continuation of one ring structure in rm with D_solver's harmonic-coordinate Newton, and the tracker's event test copied from q788 (max of D - C on NSCAN + 1 points of [0, 90] deg above TOLV = KKT lost; | `D_track.py` |
| `scripts/G_core.py` | float64 capacity KKT on the ellipse CURVE x = (rp cos t, rm sin t), unit noise, D2-symmetric rings. | `G_core.py` |
| `scripts/I_cap_centre.py` | TEXT COPY of I_rd_centre for FUNC cap (LAM from the state). | `I_cap_centre.py` |
| `scripts/I_centre_ring.py` | the filled-ellipse optimum as a D2 boundary ring plus one atom at the centre (float64, G_core quadrature). | `I_centre_ring.py` |
| `scripts/I_centre_scan.py` | for boundary-ring states of the curve-restricted problem (G_cont states.jsonl, double precision), D - C at the centre and the largest D - C on an interior grid of the filled ellipse; | `I_centre_scan.py` |
| `scripts/I_interior.py` | D - C inside the filled ellipse for a solved boundary ring (D_solver state file). | `I_interior.py` |
| `scripts/I_lfp_centre.py` | centre excess r(0) - C of a solved LFP boundary ring (a helper script state json from D_solver, FUNC lfp), with D_core2.Ring's own risk function (_D_lfp, the record's q786 text) at q = (0, 0); | `I_lfp_centre.py` |
| `scripts/I_pair_ring.py` | the filled-ellipse optimum as a D2 boundary ring plus a PAIR of atoms at (+-a, 0) on the major axis (float64, G_core quadrature). | `I_pair_ring.py` |
| `scripts/I_rd_centre.py` | TEXT COPY of I_lfp_centre for FUNC rd (SRC_S from the state). | `I_rd_centre.py` |
| `scripts/I_steps.py` | continuation of one ring by plain Newton (G_core, float64) through a list of rm values, with the scan validity read at each; | `I_steps.py` |
| `scripts/d90_mpD.py` | the KKT function D of d89 in mpmath, for checks below double precision. | `centre_scripts/d90_mpD.py` |
| `scripts/q719_wallchain_arb.py` | the D2-orbit wall-chain solve of q708/q712 in ball arithmetic (python-flint arb), with the separable Gauss-Hermite rule of q718: for a query point q, Ex[i, j] = exp(-(qx + T_i - x_j)^2/2), Ey[j, l] = exp(-(qy + T_l - y_j)^2/2), p = Ex diag( | `centre_scripts/q719_wallchain_arb.py` |
| `scripts/q786_lfp_wallchain.py` | the least favourable prior for a normal mean known to lie on the ellipse (unit noise, squared-error loss). | `centre_scripts/q786_lfp_wallchain.py` |
| `scripts/q816_centre_birth.py` | where the filled-ellipse optimum first needs an interior atom, from the formal density alone. | `centre_scripts/q816_centre_birth.py` |
| `scripts/q818_centre_atom.py` | the formal density with a centre atom. | `centre_scripts/q818_centre_atom.py` |
| `scripts/q819_centre_split.py` | when does the centre atom split? Model of q818 (formal density on the ellipse plus an atom at the centre with D(0) = C); | `centre_scripts/q819_centre_split.py` |
| `scripts/q820_centre_pair.py` | after the centre atom's split, the formal-density model with a PAIR on the major axis. | `centre_scripts/q820_centre_pair.py` |
| `scripts/q821_pair_loss.py` | TEXT COPY of q820 (pair model) with one addition after each solve: where the pair stops being optimal. | `centre_scripts/q821_pair_loss.py` |
| `scripts/q822_inner_formal.py` | the INNER ring's own formal density. | `centre_scripts/q822_inner_formal.py` |
| `scripts/q823_quartet.py` | the inner QUARTET past the pair's loss. | `centre_scripts/q823_quartet.py` |
| `scripts/q823b_quartet.py` | TEXT COPY of q823 with more digits printed (Hessians %.4e) and D - C at 45 deg; | `centre_scripts/q823b_quartet.py` |
| `scripts/q824_moment_offset.py` | the nested-copying offset from planar moments. | `centre_scripts/q824_moment_offset.py` |
| `scripts/q825_eccentric_loss.py` | the inner rings' copying loss in the ECCENTRIC parameter of the inner curve (the boundary convention of A2 Section III), from q822 logs. | `centre_scripts/q825_eccentric_loss.py` |
| `scripts/q826_inner_curve2.py` | TEXT COPY of q822 with the inner CURVE truncated separately at MC (a_0..a_MC) and the inner DENSITY at MI (d_1..d_MI). | `centre_scripts/q826_inner_curve2.py` |
| `scripts/q827_lfp_centre.py` | the centre birth of the filled-ellipse problem for the LEAST FAVOURABLE PRIOR (bounded normal mean in 2D, squared loss, unit noise), from the formal density alone. | `centre_scripts/q827_lfp_centre.py` |
| `scripts/q828_lfp_models.py` | the LFP (least favourable prior, bounded normal mean in 2D, squared loss, unit noise) interior models on the filled ellipse, the counterparts of q818/q819 (centre atom), q820/q821 (pair) and q822 (inner formal density) for capacity. | `centre_scripts/q828_lfp_models.py` |
| `scripts/q829_rd_centre.py` | TEXT COPY of q827 with the RATE-DISTORTION / NPMLE dual KKT function (q790 FUNC rd, D_core2 _D_rd, as text): source N(0, s^2 I) (env SRC_S, default 2 as the record's rd family), mixing law pi = formal density f on the ellipse; | `centre_scripts/q829_rd_centre.py` |
| `scripts/q830_models.py` | TEXT COPY of q828 (LFP interior models) with a FUNC switch (env FUNC lfp / rd). | `centre_scripts/q830_models.py` |
| `scripts/q831_mixed_centre.py` | TEXT COPY of q816 on the MIXED curve of q790/q805 (env LAM): z(t) = R e^(it) + eta (e^(-it) + LAM e^(3it)), R = rp - eps/2, eta = eps/(2 (1 + LAM)), eps = rp - rm (so the curve passes through (rp, 0) and (0, rm)); | `centre_scripts/q831_mixed_centre.py` |
| `scripts/q832_models.py` | TEXT COPY of q830 with FUNC cap added (capacity: D(q) = -int phi(y - q) log p(y) dy, constants dropped, as q819/q820) and the outer curve on the mixed geometry of q790/q831 (env LAM, 0 = ellipse): z(t) = R e^(it) + eta (e^(-it) + LAM e^(3it | `centre_scripts/q832_models.py` |
| `scripts/q833_models.py` | TEXT COPY of q832 with two more modes: quartet (X pair mass mx/2 at (+-a, 0) and Y pair my/2 at (0, +-b), equations r(a,0) = C, d/dx1 r(a,0) = 0, r(0,b) = C, d/dx2 r(0,b) = 0, as q823) and inner2 (inner formal density with two harmonics, g  | `centre_scripts/q833_models.py` |
| `scripts/q834_lr_pair.py` | TEXT COPY of q833 whose pair mode also prints the linear-response diagnostic of A2 III C on the pair's circle: g(th) = r(a cos th, a sin th) - C, q(th) = g/sin^2 th on th in (0, pi/2] (the pair sits at th = 0), q_0 = mean of q over (0, pi/2 | `centre_scripts/q834_lr_pair.py` |
| `scripts/q835_lfp_mixed_centre.py` | TEXT COPY of q827 (LFP centre birth from the formal density) on the mixed curve of q790/q831 (env LAM; | `centre_scripts/q835_lfp_mixed_centre.py` |
| `scripts/q836_rd_mixed_centre.py` | TEXT COPY of q829 (rate-distortion dual centre birth) on the mixed curve of q790/q831 (env LAM; | `centre_scripts/q836_rd_mixed_centre.py` |
| `scripts/q837_eccq_converged.py` | eccentric fourth harmonic of the inner formal density from converged rows only (residual <= 1e-9), merging the two capacity mixed-curve inner2 logs; | `centre_scripts/q837_eccq_converged.py` |
| `scripts/q839_norm_scan.py` | which normalisation of the inner pair's copying condition is exactly met at the observed losses? For each case (q822/q833 MI 1 log of the inner formal density, observed pair loss), interpolate (a_0, a_1, d_1) at the loss and evaluate candid | `centre_scripts/q839_norm_scan.py` |
| `scripts/q840_crossing_poly.py` | interpolation error of q825's eccentric crossing on a coarse scan. | `centre_scripts/q840_crossing_poly.py` |
| `scripts/q841_harm_at.py` | the K-th eccentric harmonic of the inner formal density at given rm values (cubic fit through the four logged rows around each rm), to state a pair or quartet loss as a miss in the harmonic (E - 1 = harmonic/2 - 1). | `centre_scripts/q841_harm_at.py` |
| `scripts/q842_miss_scaling.py` | TEXT COPY of q839 with the two second-aspect-ratio losses of 1477 added and the inner major radius A printed, to ask whether the miss E - 1 scales with eps or with A. | `centre_scripts/q842_miss_scaling.py` |

`logs/` holds 111 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
