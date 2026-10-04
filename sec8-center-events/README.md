# Section VIII: center events in d dimensions

On a d-ball the center alternates between a birth (an atom appears at the center) and a lift-off (the center atom becomes a
small sphere). Each event is displaced from its one-dimensional partner by Δ_d = (d − 1)(u_n + v_n d), with two parameters per
event fixed by d = 2 and 3.

| paper item | scripts | outputs in `logs/` |
|---|---|---|
| LFP events on the interval and in d dimensions | `q866_lfp1d_events.py`, `q867_lfp_radial_events.py`, `q868_lfp_events_dimd.py` | `q868_runs__*.log` |
| chains of exact LFP events (d = 2 to 8) | `q869_event_chain.py`, `merge_1596.py` | `q868_runs__chain_*.log` |
| the shift law for capacity, the NPMLE and the LFP | `q870_cap_shift_quadratic.py`, `q871_mcmahon2.py`, `q872_npmle_shift_quadratic.py`, `q872b_npmle_d8.py`, `q873_lfp_deficit_events.py` | |
| NPMLE on d-balls (d = 4, 8) | `q679_npmle_nd.py` | `q679_d*.log` |
| the LFP strip in two to five dimensions (Fig. 4) | `q874_offset_readout.py`, `q875_crossdim_offsets.py`, `q876_kappa_d1.py` | |

`q869_event_chain.py` follows the exact LFP of a d-ball from one center event to the next; `q874_offset_readout.py` turns each
event state into the effective strip plotted in Fig. 4.

See [`FILES.md`](FILES.md) for every script.
