# Section VI: leaving the boundary

On a filled region the boundary ring stays optimal until an atom is born at the center. The formal density predicts when.

| paper item | predictions | checks | outputs in `logs/` |
|---|---|---|---|
| disc thresholds 2.4534, 1.53499, 1.8982 | `q816_centre_birth.py`, `q827_lfp_centre.py`, `q829_rd_centre.py` | | |
| center births on ellipses (Table V) | `q816`, `q827`, `q829` | `I_centre_scan.py`, `I_interior.py`, `I_lfp_centre.py`, `I_rd_centre.py`, `D_solver.py` | `q816_*.log`, `q827_*.log`, `q829_*.log` |
| center births on mixed curves (Table V) | `q831_mixed_centre.py`, `q835_lfp_mixed_centre.py`, `q836_rd_mixed_centre.py` | `I_cap_centre.py`, `I_lfp_centre.py`, `I_rd_centre.py` | `q831_*.log`, `births__*.log` |
| birth curves (Fig. 3) | | | `births__*.log` |
| mass of the first center atom, its instability and split | `q818_centre_atom.py`, `q819_centre_split.py`, `q820_centre_pair.py` | `I_centre_ring.py`, `I_pair_ring.py` | |
| rings inside rings (Table VI; full list in `../paper/`) | `q822_inner_formal.py` to `q842_miss_scaling.py` | `q821_pair_loss.py`, `q823_quartet.py`, `q823b_quartet.py` | `q821_*.log` to `q833_*.log` |

The `q8xx` model scripts are float64 numpy; the ring solvers use python-flint ball arithmetic at 128 to 160 bits.

See [`FILES.md`](FILES.md) for every script.
