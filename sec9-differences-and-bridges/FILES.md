# Files in sec9-differences-and-bridges

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/d50_lib.py` | radial KKT solver for the 2D amplitude-constrained Gaussian channel (input in the disc /x/ <= A, noise N(0, I_2)). | `centre_scripts/d50_lib.py` |
| `scripts/d70_lib3d.py` | radial KKT solver for the 3D amplitude-constrained Gaussian channel (input in the ball /x/ <= A, noise N(0, I_3)). | `centre_scripts/d70_lib3d.py` |
| `scripts/d72_libnd.py` | radial KKT solver for the amplitude-constrained Gaussian channel with input in the d-ball /x/ <= A and noise N(0, I_d). | `centre_scripts/d72_libnd.py` |
| `scripts/d74_newton_scaled.py` | d50_lib.newton copied as text with one change: the acceptance norm (and the linear solve) uses residual rows of interior rings' stationarity equations divided by max(r_j, 1e-3), because i'(r) ~ r i''(0) near the origin makes the plain max-n | `centre_scripts/d74_newton_scaled.py` |
| `scripts/d76_npmle2d_lib.py` | radial KKT solver for the 2D NPMLE (Gaussian location mixture N(theta, I_2) fitted to data uniform on the disc /y/ <= A; | `centre_scripts/d76_npmle2d_lib.py` |
| `scripts/d85_annulus_lib.py` | d85_annulus_lib (copy of d72_libnd.py as text; | `centre_scripts/d85_annulus_lib.py` |
| `scripts/q505_handcheck.py` | Q505 hand check (verification gate): two change points recomputed by a DIFFERENT method. | `q505_handcheck.py` |
| `scripts/q505_rd_bridge.py` | the RATE-DISTORTION bridge of the highest target (memory project-highest-target-optimal-discreteness, lapor177). | `q505_rd_bridge.py` |
| `scripts/q506_ri_replication.py` | bridge 2 of the highest target, rational inattention. | `q506_ri_replication.py` |
| `scripts/q562_diag.py` | Post hoc diagnostic (label M, after the sealed file was opened): I at the certified transitions near the failed cases. | `q562_diag.py` |
| `scripts/q562_part2.py` | Appends PART 2 (the prediction numbers) to q562_prereg.txt; | `q562_part2.py` |
| `scripts/q562_predict.py` | Q562 part 2: compute the blind predictions (see q562_prereg.txt part 1). | `q562_predict.py` |
| `scripts/q562_rdlib.py` | Q562 tool: certified NPMLE (= rate-distortion for squared error, = rational inattention with quadratic loss and Shannon cost) on a source in noise units, with the test-channel distortion D_noise and mutual information I (nats) of the optimu | `q562_rdlib.py` |
| `scripts/q649_rings2d_cont.py` | Amendment 969. Continuation in A of the 2D disc channel's optimal input (rings + origin point), with event handling, using d50_lib (definitions only). | `centre_scripts/q649_rings2d_cont.py` |
| `scripts/q650_rings2d_far.py` | far continuation of the 2D disc channel. | `centre_scripts/q650_rings2d_far.py` |
| `scripts/q653_bulk2d_fit.py` | Amendment 973. The 2D bulk law with its curvature term, fitted to ring depths. | `centre_scripts/q653_bulk2d_fit.py` |
| `scripts/q659_rings2d_evstates.py` | q659_rings2d_evstates (copy of q650_rings2d_far.py as text; | `centre_scripts/q659_rings2d_evstates.py` |
| `scripts/q661_centre_const.py` | empirical centre constant e = 2 N(A) - K_eff at each event, N from the d-dimensional bulk law fitted to the state's own bulk rings (c = d - 1 fixed, two free constants, rings with depth >= 6 and radius >= 6) and integrated to depth A (the c | `centre_scripts/q661_centre_const.py` |
| `scripts/q664_rings2d_resume.py` | q664 (copy of q659_rings2d_evstates.py as text: RESUME from a saved state, wider lift-off seeds). | `centre_scripts/q664_rings2d_resume.py` |
| `scripts/q668_npmle2d_cont.py` | continuation of the 2D NPMLE (uniform data on the disc). | `centre_scripts/q668_npmle2d_cont.py` |
| `scripts/q686_annulus_births.py` | births of the optimal input on the shell a <= /x/ <= a + W at fixed inner radius a, by continuation in the width W from WSTART (two walls only) to WMAX. | `centre_scripts/q686_annulus_births.py` |
| `scripts/q686f_annulus_births.py` | q686f (copy of q686e as text: the split seeds reach distance 0.7 on both sides at W_hi + 1e-3, 3e-3, 1e-2, and the fast solve is two stage (30 Newton iterations, then 120 more only if the residual already fell 100-fold); | `centre_scripts/q686f_annulus_births.py` |
| `scripts/q686g_annulus_births.py` | q686g (copy of q686f as text: split signals are read only at rings whose nearest support neighbour is more than 0.15 away, so the two rings of a pair just born at a split cannot trigger a second, spurious split; | `centre_scripts/q686g_annulus_births.py` |
| `scripts/q686h_annulus_births.py` | so a noise-level maximum between the two rings of a pair just born at a split is not read as a birth; | `centre_scripts/q686h_annulus_births.py` |
| `scripts/q693_centre_dfrac.py` | q661's centre readout (definitions copied as text) for a fractional-d ball run of q684; | `centre_scripts/q693_centre_dfrac.py` |
| `scripts/q696_ba2d_exactwall.py` | q694 with the wall candidates placed exactly on the boundary curve. | `centre_scripts/q696_ba2d_exactwall.py` |
| `scripts/q699_wall_kkt.py` | wall support of a q696/q698 solution read from the KKT function D on the wall. | `centre_scripts/q699_wall_kkt.py` |
| `scripts/q700_ba2d_ellipse.py` | q700 (copy of q698 as text with a second semi-axis B along y: the domain is (/x//A)^p + (/y//B)^p <= 1; | `centre_scripts/q700_ba2d_ellipse.py` |
| `scripts/q701_ring_wall.py` | for a q700 solution, per 2-degree bin of the polar angle over the whole turn, the maximum of D - max D over interior grid points with gauge radius in [RLO, RHI] (the ring band) and over the wall points, in units of the bracket width. | `centre_scripts/q701_ring_wall.py` |
| `scripts/q701b_ring_wall.py` | q701b (copy of q701 as text with 6-degree bins: at the ring radius 1.15 a 2-degree bin spans 0.04, less than the grid spacing 0.05, so some bins held no grid point on the ridge; | `centre_scripts/q701b_ring_wall.py` |
| `scripts/q702_ba2d_rose.py` | env SHAPE=rose gives the analytic domain r <= A (1 - EPS cos 4 theta) (bulges toward the diagonals, like a superellipse with p > 2); | `centre_scripts/q702_ba2d_rose.py` |

`logs/` holds 114 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
