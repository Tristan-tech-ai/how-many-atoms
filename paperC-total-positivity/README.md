# Total Positivity and the Monotonicity Question for the Amplitude-Constrained Gaussian Channel

Code and data for the paper of this title by I Made Tristan Hope Firdaus (2026).

The paper asks whether the number of mass points of the capacity-achieving input can decrease as the amplitude grows. It reduces
this question to two quantities at the origin (Proposition 1, proved). It proves that the Gram part of the matrix that governs the
motion of the mass points is strictly totally positive at every configuration (Theorem 1). At 91 computed optimal states it proves
the outward motion by a computer-assisted proof in ball arithmetic (Proposition 2). The conjecture itself stays open. The
computer-assisted proof is in [`../sec12-total-positivity/`](../sec12-total-positivity/), and the data of the observations of
Section VII are in [`../sec13-monotonicity/`](../sec13-monotonicity/). This folder holds the worked example of Section V, the
scripts of Figs. 1 to 4 with the data they need, and the check of every printed number.

## Where to find what

| paper | file |
|---|---|
| Proposition 2, Steps 1 to 5 of Appendix B | [`../sec12-total-positivity/scripts/q1882_exact_state_closed_chain.py`](../sec12-total-positivity/scripts/q1882_exact_state_closed_chain.py); one output per state, `../sec12-total-positivity/logs/q1882_states__*.txt` |
| Proposition 2, Step 6 of Appendix B | [`../sec12-total-positivity/scripts/q1891_centre_mass_flow_exact.py`](../sec12-total-positivity/scripts/q1891_centre_mass_flow_exact.py); outputs `../sec12-total-positivity/logs/q1891_out__*.txt` |
| the bounds quoted in Appendix B | [`../sec12-total-positivity/logs/q1883_out.txt`](../sec12-total-positivity/logs/q1883_out.txt) |
| the stationary states rerun with the smaller box (Section VI) | [`../sec12-total-positivity/scripts/q1886_nonoptimal_exact_states.py`](../sec12-total-positivity/scripts/q1886_nonoptimal_exact_states.py) (its docstring fixes the box radius and the step), output `../sec12-total-positivity/logs/q1886_out.txt` |
| Section V (the worked example with four mass points) | `scripts/example_k4.py`, output `scripts/example_k4.json` |
| Section VI (the counts and positions of the certified states) | `scripts/certified_states_summary.py` and `scripts/certified_positions.py`, outputs `scripts/certified_states.json` and `scripts/certified_positions.json` |
| Fig. 1 | `scripts/make_fig_face_C.py`, data [`data/q1382_states_to15p8.json`](data/q1382_states_to15p8.json) |
| Fig. 2 | `scripts/make_fig_reduction.py`, same data |
| Fig. 3 | `scripts/make_fig_tp.py`, same data and `scripts/example_k4.json` |
| Fig. 4 and Observation 2 | `scripts/make_fig_margin_C.py`, data in [`../sec13-monotonicity/logs/`](../sec13-monotonicity/logs/) and [`../sec10-births-law/logs/q1416_rows.json`](../sec10-births-law/logs/q1416_rows.json) |
| Observation 1 | [`../sec13-monotonicity/`](../sec13-monotonicity/) |

## The data of Figs. 1 to 3

[`data/q1382_states_to15p8.json`](data/q1382_states_to15p8.json) holds 2504 optimal inputs at amplitudes from 1.01 to 15.797 with
2 to 18 mass points, computed by a continuation in A in double precision, each with its Kuhn-Tucker residual, together with the
amplitudes of the 16 changes of the support at the origin. Each state gives `K`, `A`, the positive interior positions `u`, their
masses followed by the mass of each end point in `w`, and the mass `wc` of the center point. These are checked numerically only.

## Running the scripts

Install the packages of [`../requirements.txt`](../requirements.txt), then run any script from any folder, for example

```
python scripts/make_fig_face_C.py
```

The figure scripts and `scripts/example_k4.py` find their inputs through `scripts/layout.py`, write their outputs next to themselves and run in
seconds; `scripts/make_fig_tp.py` first reproduces the matrix of the worked example and stops if it does not. The scripts
`scripts/certified_states_summary.py` and `scripts/certified_positions.py` read the input folder of the certificate in the layout of the original
project; the states they list are those of the logs `../sec12-total-positivity/logs/q1882_states__*.txt`. The certificate scripts of
[`../sec12-total-positivity/`](../sec12-total-positivity/) are described in its README.
