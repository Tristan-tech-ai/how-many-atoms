# sec10-births-law

The material behind Section IV-B of the paper. It holds the births law, which gives the amplitude of the next event at the origin from the atom count of the half-line optimum, and the blind tests of that law. The stored coefficients `e_2` to `e_20` of the series are in `logs/d22_out_20.json`.

## What is here

- `scripts/` holds the half-line solvers, the series and resummation scripts, the predictors and the verdict scripts that the registration-log entries cited by Section IV-B name. `FILES.md` lists every script with the first sentence of its own docstring, the name it had when it ran, and the sections that cite it.
- `logs/` holds the outputs: the predictions written before each blind block, the measured events, the verdict outputs, and the per-state outputs of the structure scans of the stored branches.

## How to read it

This folder is a record. The scripts are the ones that ran, and the file names match the registration log. A verdict script prints its verdict as its last line, and a run that crashed or printed no verdict line counts as void. The log states each criterion before the output existed.

The log is in [`../registration-log/`](../registration-log/). A blind block is introduced there by an entry that fixes the prediction table with a hash, the bar for each event and the verdict script. The result is appended as a later entry.

## Scope

The folder holds the files that the cited entries name and that the build found in the project tree. Outputs larger than one megabyte are not copied.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
