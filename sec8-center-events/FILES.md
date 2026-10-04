# Files in sec8-center-events

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/d50_lib.py` | radial KKT solver for the 2D amplitude-constrained Gaussian channel (input in the disc /x/ <= A, noise N(0, I_2)). | `centre_scripts/d50_lib.py` |
| `scripts/d70_lib3d.py` | radial KKT solver for the 3D amplitude-constrained Gaussian channel (input in the ball /x/ <= A, noise N(0, I_3)). | `centre_scripts/d70_lib3d.py` |
| `scripts/d72_libnd.py` | radial KKT solver for the amplitude-constrained Gaussian channel with input in the d-ball /x/ <= A and noise N(0, I_d). | `centre_scripts/d72_libnd.py` |
| `scripts/d74_newton_scaled.py` | d50_lib.newton copied as text with one change: the acceptance norm (and the linear solve) uses residual rows of interior rings' stationarity equations divided by max(r_j, 1e-3), because i'(r) ~ r i''(0) near the origin makes the plain max-n | `centre_scripts/d74_newton_scaled.py` |
| `scripts/d79_npmle_nd_lib.py` | D(r) = (d/A^d) int_0^A s^(d-1) E(s, r)/ptil(s) ds <= C, C = 1 forced by sum w = 1; | `centre_scripts/d79_npmle_nd_lib.py` |
| `scripts/merge_1596.py` | merged event lists for the 1590/1591/1593/1597 readouts, joining CHAIN entries that numpy printing wrapped over several lines (continuation lines start with whitespace). | `centre_scripts/q868_runs/merge_1596.py` |
| `scripts/q649_rings2d_cont.py` | Amendment 969. Continuation in A of the 2D disc channel's optimal input (rings + origin point), with event handling, using d50_lib (definitions only). | `centre_scripts/q649_rings2d_cont.py` |
| `scripts/q656_shells3d_cont.py` | continuation of the 3D ball channel (shells + origin). | `centre_scripts/q656_shells3d_cont.py` |
| `scripts/q659_shells3d_evstates.py` | q659_shells3d_evstates (copy of q656_shells3d_cont.py as text; | `centre_scripts/q659_shells3d_evstates.py` |
| `scripts/q662_ball_nd.py` | continuation for the d-ball. | `centre_scripts/q662_ball_nd.py` |
| `scripts/q679_npmle_nd.py` | NPMLE continuation on the d-ball (data uniform), env DIM, ASTART. | `centre_scripts/q679_npmle_nd.py` |
| `scripts/q866_lfp1d_events.py` | the first 1D LFP events on [-m, m] read as thresholds of explicit symmetric priors (no EG stage). | `centre_scripts/q866_lfp1d_events.py` |
| `scripts/q867_lfp_radial_events.py` | first LFP events on the disc (d = 2) and the ball (d = 3) of radius m, read as thresholds of explicit radial priors. | `centre_scripts/q867_lfp_radial_events.py` |
| `scripts/q868_lfp_events_dimd.py` | q867 for any dimension d (radial kernel Gamma(d/2) (2/x)^nu ive(nu, x), nu = d/2 - 1; | `centre_scripts/q868_lfp_events_dimd.py` |
| `scripts/q869_event_chain.py` | chain of LFP centre events in dimension d by sequential q868 tracks (1579). | `centre_scripts/q869_event_chain.py` |
| `scripts/q870_cap_shift_quadratic.py` | capacity centre-event shifts in d = 4, 5, 8 predicted from d = 2, 3 by shift_d = (d - 1)(alpha + beta d) at matched K_eff. | `centre_scripts/q870_cap_shift_quadratic.py` |
| `scripts/q871_mcmahon2.py` | second-order McMahon model of centre-event shifts. | `centre_scripts/q871_mcmahon2.py` |
| `scripts/q872_npmle_shift_quadratic.py` | NPMLE centre-event shifts against the 1D NPMLE transitions; | `centre_scripts/q872_npmle_shift_quadratic.py` |
| `scripts/q872b_npmle_d8.py` | NPMLE d = 8 quadratic-law residuals against the capacity d = 8 residuals at the same K_eff. | `centre_scripts/q872b_npmle_d8.py` |
| `scripts/q873_lfp_deficit_events.py` | LFP deficit d - B at the tracked event states against 4 j_{d/2-1,1}^2/(m + delta)^2. | `centre_scripts/q873_lfp_deficit_events.py` |
| `scripts/q874_offset_readout.py` | implied LFP offsets delta_impl = 2 j_{d/2-1,1}/sqrt(d - B) - m at the events of a q869 chain log, and the 1590 readout. | `centre_scripts/q874_offset_readout.py` |
| `scripts/q875_crossdim_offsets.py` | 3D LFP offsets predicted from the 1D and 2D long chains, offset_3D = offset_1D + 2 kappa_2D/(m + delta). | `centre_scripts/q875_crossdim_offsets.py` |
| `scripts/q876_kappa_d1.py` | kappa_1(m) = (offset_d(m) - offset_1(m)) (m + delta)/(d - 1) at d = 1.05, 1.10, at the midpoints of 1D event intervals. | `centre_scripts/q876_kappa_d1.py` |

`logs/` holds 60 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
