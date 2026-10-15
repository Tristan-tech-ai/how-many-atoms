# Files in appendix-checks

Scripts are formatted with black (layout only). The original path is the name used in the registration log. The last column gives the sections of the paper whose sources name the script.

| script | what it does (from its own docstring) | original path | cited by |
|---|---|---|---|
| `scripts/fit_k.py` | Fit the strip response (k, k2) of Amendment 1519 to the accepted q850 rows at A = 16 and 20.3 (reads logs; | `scratch/fit_k.py` | S8 |
| `scripts/q845_npmle_disc.py` | the population NPMLE for data uniform on the disc of radius R, unit Gaussian noise. | `centre_scripts/q845_npmle_disc.py` | S8 |
| `scripts/q846_npmle_disc_cnm.py` | the disc NPMLE of q845 solved by the constrained Newton method (CNM, Wang 2007, J. | `centre_scripts/q846_npmle_disc_cnm.py` | S8 |
| `scripts/q849_npmle_ellipse.py` | the population NPMLE for data uniform on the filled ellipse x^2/A^2 + y^2/B^2 <= 1 in the plane, unit Gaussian noise, without rotation symmetry. | `centre_scripts/q849_npmle_ellipse.py` | S8 |
| `scripts/q850_cap1d_tilt.py` | 1D amplitude-constrained Gaussian channel with a linear input reward, maximise I(X; | `centre_scripts/q850_cap1d_tilt.py` | S8 |
| `scripts/q852_npmle_ellipse_gt.py` | TEXT COPY of q849 that also prints Gt = int over the ellipse of log(m/f0) = (L - log f0)/f0 with L = sum Q log m (the optimal value; | `centre_scripts/q852_npmle_ellipse_gt.py` | S8 |
| `scripts/q853_npmle_star_gt.py` | TEXT COPY of q852 for the region r <= R0 (1 + EPS cos 4t): polar data quadrature x = s r(t) cos t, y = s r(t) sin t (Jacobian s r(t)^2), f0 = 1/area, candidate and fine-grid masks /x/ <= r(arg x). | `centre_scripts/q853_npmle_star_gt.py` | S8 |

`logs/` holds 6 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
