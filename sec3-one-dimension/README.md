# Section III: one dimension

On an interval [−A, A] the optimal atoms crowd near the two ends. Each problem has an edge constant (Table I of the paper),
and the number of atoms grows like 0.4002 L^(4/3).

| paper item | scripts | outputs in `logs/` |
|---|---|---|
| capacity strip β = 3.98160794709 (Table I) | `cap_beta.py`, `cap_edge_pointwise.py` | `walk_K50_zero_726.log` |
| LFP solver and its count changes | `q486_lfp_sym.py` | `q486_lfp_sym.log`, `results_q486_lfp_sym.json` |
| LFP forward test: count changes predicted before they were computed | `q487_lfp_forward.py`, `q489_lfp_certificate.py`, `q493_compare_mp.py`, `q494_freeze_30_40.py`, `q495_score_40_50.py` | `lfp_forward_predictions*.json` |
| NPMLE forward test and the edge defect D | `q497_npmle_freeze.py`, `q498_npmle_mpcont.py` | `npmle_forward_predictions.json`, `results_q498_npmle_mpcont.json` |

The constants k and D′(0) of Table I come from the scripts in [`../sec7-value-laws/`](../sec7-value-laws/) (`q850`, `q854`,
`q855`, `fit_k.py`).

**A note on `lfp_forward_predictions_OVERWRITTEN_0740.json`.** On 23 September 2026 a helper script imported
`q487_lfp_forward.py`, which had no `__main__` guard, and the import rewrote the frozen prediction file with a fit that had seen
the test data. The overwritten file is kept under this name for the record; the comparison made with it was declared void. The
fit was rerun on the pre-registered window only and written to a new file (`lfp_forward_predictions_restored.json`), and
`q487_lfp_forward.py` was changed so that importing it writes nothing.

See [`FILES.md`](FILES.md) for every script.
