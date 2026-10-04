# Files in sec7-value-laws

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/fit_k.py` | Fit the strip response (k, k2) of Amendment 1519 to the accepted q850 rows at A = 16 and 20.3 (reads logs; | `scratch/fit_k.py` |
| `scripts/pred1533.py` | Predictions for 1533 (written before any 2D C value is read): reads only 'W' from the 2D annulus files. | `scratch/pred1533.py` |
| `scripts/pred1535.py` | Predictions for 1535 (3D shell capacity), written before any shell C value is read: reads only 'W' from the shell file. | `scratch/pred1535.py` |
| `scripts/pred1540.py` | Predictions for 1540 (NPMLE 3D shell), from the 1523 ball fits; | `scratch/pred1540.py` |
| `scripts/pred1542.py` | width prediction for the eccentric annulus (R 18, r 8): Delta(e) = oint 2 delta0(W/2) ds_mid - 2 pi R_mid 2 delta0(5). | `scratch/pred1542.py` |
| `scripts/pred1546.py` | Capacity on the eccentric annulus R 15, r 10: Delta(e) = int eps(W/2) (R - W/2) d theta - 2 pi R_mid eps(2.5), in e^(C + log 2 pi e). | `scratch/pred1546.py` |
| `scripts/pred1547.py` | 1547 predictions: NPMLE eccentric annulus R 18, r 8 at e = 3 (value and ring-count changes), from the merged q858 scans. | `scratch/pred1547.py` |
| `scripts/pred1549.py` | 1547 predictions: NPMLE eccentric annulus R 18, r 8 at e = 3 (value and ring-count changes), from the merged q858 scans. | `scratch/pred1549.py` |
| `scripts/q696_ba2d_exactwall.py` | q694 with the wall candidates placed exactly on the boundary curve. | `centre_scripts/q696_ba2d_exactwall.py` |
| `scripts/q845_npmle_disc.py` | the population NPMLE for data uniform on the disc of radius R, unit Gaussian noise. | `centre_scripts/q845_npmle_disc.py` |
| `scripts/q846_npmle_disc_cnm.py` | the disc NPMLE of q845 solved by the constrained Newton method (CNM, Wang 2007, J. | `centre_scripts/q846_npmle_disc_cnm.py` |
| `scripts/q847_npmle_annulus.py` | TEXT COPY of q846 with the data uniform on the annulus R1 <= /y/ <= R (env R1, default 0; | `centre_scripts/q847_npmle_annulus.py` |
| `scripts/q848_npmle_ball.py` | TEXT COPY of q846 for the BALL of radius R in R^3 with unit 3D Gaussian noise. | `centre_scripts/q848_npmle_ball.py` |
| `scripts/q849_npmle_ellipse.py` | the population NPMLE for data uniform on the filled ellipse x^2/A^2 + y^2/B^2 <= 1 in the plane, unit Gaussian noise, without rotation symmetry. | `centre_scripts/q849_npmle_ellipse.py` |
| `scripts/q850_cap1d_tilt.py` | 1D amplitude-constrained Gaussian channel with a linear input reward, maximise I(X; | `centre_scripts/q850_cap1d_tilt.py` |
| `scripts/q851_npmle1d_tilt.py` | population NPMLE in 1D with tilted data, f(x) proportional to exp(t x) on [-a, a], unit Gaussian kernel. | `centre_scripts/q851_npmle1d_tilt.py` |
| `scripts/q852_npmle_ellipse_gt.py` | TEXT COPY of q849 that also prints Gt = int over the ellipse of log(m/f0) = (L - log f0)/f0 with L = sum Q log m (the optimal value; | `centre_scripts/q852_npmle_ellipse_gt.py` |
| `scripts/q853_npmle_star_gt.py` | TEXT COPY of q852 for the region r <= R0 (1 + EPS cos 4t): polar data quadrature x = s r(t) cos t, y = s r(t) sin t (Jacobian s r(t)^2), f0 = 1/area, candidate and fine-grid masks /x/ <= r(arg x). | `centre_scripts/q853_npmle_star_gt.py` |
| `scripts/q854_npmle1d_defect.py` | TEXT COPY of q851 that also prints Def(t) = int f log(m/f) dx (f the normalised data density) and Dslope = [Def - D (f(a) + f(-a))]/(t (f(a) - f(-a))), D = -0.304944111846. | `centre_scripts/q854_npmle1d_defect.py` |
| `scripts/q855_fit_strip_k.py` | fit the strip response (k, k2) of Amendment 1519 to the accepted q850 rows at A = 16 and 20.3 (reads logs; | `centre_scripts/q855_fit_strip_k.py` |
| `scripts/q856_npmle_shell3d.py` | TEXT COPY of q848 with the data uniform on the shell R1 <= /y/ <= R in R^3 (env R1, default 0; | `centre_scripts/q856_npmle_shell3d.py` |
| `scripts/q857_npmle_eccannulus.py` | population NPMLE for data uniform on an ECCENTRIC annulus in the plane: the disc /y/ <= R minus the disc /y - (e, 0)/ < r (inner circle centred at (e, 0)), unit Gaussian noise. | `centre_scripts/q857_npmle_eccannulus.py` |
| `scripts/q858_npmle1d_defect_anya.py` | TEXT COPY of q854 with one change: the data panels are np.linspace(-a, a, round(4a) + 1), so any a is valid (q854's np.arange(-a, a + 1e-9, 0.5) panels end short of +a unless a is a multiple of 0.5). | `centre_scripts/q858_npmle1d_defect_anya.py` |
| `scripts/q859_ba2d_annulus.py` | TEXT COPY of q696 for an ANNULUS with two exact walls: the outer circle /y/ = R (centre 0) and the inner circle /y - (E, 0)/ = RI (eccentric if E != 0). | `centre_scripts/q859_ba2d_annulus.py` |
| `scripts/q861_lfp_disc.py` | the least favourable prior (LFP) for the 2D bounded normal mean on the disc /theta/ <= m, unit noise, squared loss (sum over both coordinates), restricted to rotation-invariant priors (rings). | `centre_scripts/q861_lfp_disc.py` |
| `scripts/q861b_lfp_disc_active.py` | q861b = q861 + active-set loop (see the comment block after polish). | `centre_scripts/q861b_lfp_disc_active.py` |
| `scripts/q862_lfp1d.py` | the 1D least favourable prior for the bounded normal mean /theta/ <= m (unit noise, squared loss), symmetric priors. | `centre_scripts/q862_lfp1d.py` |
| `scripts/q863_lfp_grid2d.py` | the 2D least favourable prior on a general region (here the filled ellipse x^2/A^2 + y^2/B^2 <= 1), unit noise, squared loss summed over both coordinates, by exponentiated-gradient ascent of the Bayes risk on a grid (step H). | `centre_scripts/q863_lfp_grid2d.py` |
| `scripts/q863b_lfp_grid2d_annulus.py` | the 2D least favourable prior on a general region (here the filled ellipse x^2/A^2 + y^2/B^2 <= 1), unit noise, squared loss summed over both coordinates, by exponentiated-gradient ascent of the Bayes risk on a grid (step H). | `centre_scripts/q863b_lfp_grid2d_annulus.py` |
| `scripts/q863c_lfp_grid2d_polygon.py` | the 2D least favourable prior on a general region (here the filled ellipse x^2/A^2 + y^2/B^2 <= 1), unit noise, squared loss summed over both coordinates, by exponentiated-gradient ascent of the Bayes risk on a grid (step H). | `centre_scripts/q863c_lfp_grid2d_polygon.py` |
| `scripts/q864_dirichlet_offset.py` | principal Dirichlet eigenvalue (-Laplace u = lambda u, u = 0 outside) of the filled ellipse x^2/A^2 + y^2/B^2 <= 1 widened by distance DL (the set of points within DL of the ellipse), by the 5-point Laplacian on a grid of step H (grid point | `centre_scripts/q864_dirichlet_offset.py` |
| `scripts/q864b_dirichlet_annulus.py` | principal Dirichlet eigenvalue (-Laplace u = lambda u, u = 0 outside) of the filled ellipse x^2/A^2 + y^2/B^2 <= 1 widened by distance DL (the set of points within DL of the ellipse), by the 5-point Laplacian on a grid of step H (grid point | `centre_scripts/q864b_dirichlet_annulus.py` |
| `scripts/q864d_dirichlet_triangle.py` | principal Dirichlet eigenvalue of an equilateral triangle (inradius RHO, centroid at the origin, as in q863c) widened by distance DL in one of two ways: MODE "sharp" = every edge moved outward by DL (a triangle of inradius RHO + DL, exact v | `centre_scripts/q864d_dirichlet_triangle.py` |
| `scripts/q865_lfp_ball3d.py` | q861b = q861 + active-set loop (see the comment block after polish). | `centre_scripts/q865_lfp_ball3d.py` |
| `scripts/score1522.py` |  | `scratch/score1522.py` |
| `scripts/score1524.py` |  | `scratch/score1524.py` |
| `scripts/score1526.py` |  | `scratch/score1526.py` |
| `scripts/score1528.py` |  | `scratch/score1528.py` |
| `scripts/score1530.py` |  | `scratch/score1530.py` |
| `scripts/score1533.py` | Scoring of 1533: reads the 2D C values (run only after 1533 is appended). | `scratch/score1533.py` |
| `scripts/score1535.py` | Scoring of 1535: reads the 3D shell C values (run only after 1535 is appended). | `scratch/score1535.py` |
| `scripts/score1538.py` |  | `scratch/score1538.py` |
| `scripts/score1540.py` |  | `scratch/score1540.py` |
| `scripts/score1542.py` |  | `scratch/score1542.py` |
| `scripts/score1544.py` |  | `scratch/score1544.py` |
| `scripts/score1546.py` |  | `scratch/score1546.py` |
| `scripts/score1547.py` |  | `scratch/score1547.py` |
| `scripts/score1549.py` |  | `scratch/score1549.py` |
| `scripts/score1553.py` |  | `scratch/score1553.py` |
| `scripts/score1556.py` |  | `scratch/score1556.py` |
| `scripts/score1559.py` |  | `scratch/score1559.py` |
| `scripts/score1561.py` |  | `scratch/score1561.py` |
| `scripts/score1562.py` |  | `scratch/score1562.py` |
| `scripts/score1565.py` |  | `scratch/score1565.py` |
| `scripts/score1566.py` |  | `scratch/score1566.py` |
| `scripts/score1570.py` | fit R_eff of the bulk shape of sqrt(p) to the ground state of a disc (J0) or ball (sinc), rho in [0, m - 1.5]. | `scratch/score1570.py` |

`logs/` holds 67 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
