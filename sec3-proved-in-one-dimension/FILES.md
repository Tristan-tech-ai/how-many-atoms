# Files in sec3-proved-in-one-dimension

Scripts are formatted with black (layout only). The original path is the name used in the registration log. The last column gives the sections of the paper whose sources name the script.

| script | what it does (from its own docstring) | original path | cited by |
|---|---|---|---|
| `scripts/q1796_lower_bound_constants_arb.py` | ball-arithmetic evaluation of the only numerical input of the proof of Theorem 1 (lower bound of order A^(4/3) for the capacity-achieving input on [-A, A]) and of the constants that follow from it. | `small_gap_q206/q1796_lower_bound_constants_arb.py` | III-B |
| `scripts/q1848_printed_edge_config_gate.py` | gate for the lower bound of the strip constant given by ONE explicit edge configuration, the n = 3 configuration exactly as q1846 printed it (gap and atoms to 5 or 6 decimals). | `small_gap_q206/q1848_printed_edge_config_gate.py` | III-B |
| `scripts/q1849_edge_bound_ball_certificate.py` | ball-arithmetic certificate of the edge functional of the configuration gated in q1848, so that the lower bound of the strip constant rests on rigorous arithmetic and not on floating quadrature. | `small_gap_q206/q1849_edge_bound_ball_certificate.py` | III-B |
| `scripts/q1876_pr2b_kkt_edge_certificate.py` | ball-arithmetic certificate that the edge KKT function K(x) = int phi(t - x) log G(t) dt is >= 0 for every x <= 0, for the test profile G(t) = Phi(-g - t) + sum_k nu_k phi(t - s_k), g = 4.361045, (s_k, nu_k) = (0, 3.49413), (-2.68983, 2.250 | `small_gap_q206/q1876_pr2b_kkt_edge_certificate.py` | III-B |
| `scripts/q1877_pr2d_MA_limit_reading.py` | numerical check of the lemma used by the dual upper bound, M_A - 2A -> 2 int (G - 1{t<0}) dt = 2(sum nu - g), where M_A = int G(y - A) G(-y - A) dy and G is the test profile of q1876 (g = 4.361045; | `small_gap_q206/q1877_pr2d_MA_limit_reading.py` | III-B |
| `scripts/q1878_pr2c_profile_optimisation_reading.py` | float64 search for a lower dual upper bound on beta. | `small_gap_q206/q1878_pr2c_profile_optimisation_reading.py` | III-B |
| `scripts/q1879_pr2e_kkt_edge_certificate_profile.py` | ball-arithmetic certificate that the edge KKT function K(x) = int phi(t - x) log G(t) dt is >= 0 for every x <= 0, for a test profile read from small_gap_q206/q1878_profiles.json, G(t) = Phi(-g - t) + sum_k nu_k phi(t - s_k), s_n = -g exact | `small_gap_q206/q1879_pr2e_kkt_edge_certificate_profile.py` | III-B |
| `scripts/q1880_pr2f_profile_reoptimisation_reading.py` | second float64 search for a dual test profile, after q1879 showed that the q1878 optimum is negative between its grid points (K(-2.71875) = -6.5e-6) and fails lemma T at X0 = 16. | `small_gap_q206/q1880_pr2f_profile_reoptimisation_reading.py` | III-B |

`logs/` holds 12 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
