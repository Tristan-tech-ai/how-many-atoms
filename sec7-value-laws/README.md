# Section VII: optimal values in d dimensions

The one-dimensional edge constants fix the optimal value on discs, balls, shells and annuli: a widened volume for capacity, a
perimeter law for the NPMLE, and a widened Dirichlet eigenvalue for the LFP.

| paper item | scripts | outputs in `logs/` |
|---|---|---|
| capacity on intervals: strip response k (Table I) | `q850_cap1d_tilt.py`, `q855_fit_strip_k.py`, `fit_k.py` | `q850_eps1533__*.log` |
| capacity on superellipses and annuli | `q696_ba2d_exactwall.py`, `q859_ba2d_annulus.py` | `q859_runs__*.log` |
| NPMLE on intervals: defect D and slope response D′(0) | `q851_npmle1d_tilt.py`, `q854_npmle1d_defect.py`, `q858_npmle1d_defect_anya.py` | |
| NPMLE on discs, annuli, balls, ellipses, stars, shells | `q845` to `q849`, `q852_npmle_ellipse_gt.py`, `q853_npmle_star_gt.py`, `q856_npmle_shell3d.py`, `q857_npmle_eccannulus.py` | `q856_runs__*.log` |
| LFP on intervals | `q862_lfp1d.py` | `q862_runs__*.log` |
| LFP on discs, ellipses, annuli, balls | `q861_lfp_disc.py`, `q861b_lfp_disc_active.py`, `q863_lfp_grid2d.py`, `q863b_*`, `q863c_*`, `q865_lfp_ball3d.py` | `q861_runs__*`, `q861b_runs__*`, `q863_runs__*`, `q865_runs__*` |
| Dirichlet eigenvalues of widened regions | `q864_dirichlet_offset.py`, `q864b_dirichlet_annulus.py`, `q864d_dirichlet_triangle.py` | |
| registered predictions and their scoring (Table VII, close walls) | `pred15xx.py`, `score15xx.py` | |

`pred15xx.py` wrote a prediction before the run, and `score15xx.py` scored it afterwards; the number is the entry of the
registration log that holds the prediction.

See [`FILES.md`](FILES.md) for every script.
