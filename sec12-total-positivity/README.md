# sec12-total-positivity

The material behind Section V-B of the paper and Appendices C and D. It concerns total positivity of the matrix that governs how the atoms move with the amplitude, and the proof of the part of it built from the Gaussian kernel.

## What is here

- `scripts/` holds the scripts that build the matrix at stored states, test its minors in ball arithmetic or in high-precision floating point, and check the sign structure. `FILES.md` lists every script with the first sentence of its own docstring and the name it had when it ran.
- `logs/` holds the outputs of those runs.

## How to read it

This folder is a record. The scripts are the ones that ran, and the file names match the registration log. The paper proves total positivity for the part of the matrix built from the Gaussian kernel. For the whole matrix it reports a check at stored states and, at the 91 optimal states of the families, a computer-assisted proof at the exact optimal states (Proposition 1 of Section V-B, proved in Appendix D).

The files of that proof are in `logs/`. Each `q1882_states__*.txt` is the output of `scripts/q1882_exact_state_closed_chain.py` for one state; its last line is the verdict, and all 94 states end with VERDICT q1882: PASS. Each `q1884_polished__*.json` holds the polished state that is the centre of the box for one state (positions and masses to 60 digits, the residual, and a sha256 of the values). `q1883_out.txt` collects the numbers that the appendix quotes. One state is rerun with `python q1882_exact_state_closed_chain.py STATE.json --optimal 1`; like the other scripts it expects the folder layout of the original project.

The log is in [`../registration-log/`](../registration-log/).

## Scope

The folder holds the files that the cited entries name and that the build found in the project tree. Outputs larger than one megabyte are not copied.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
