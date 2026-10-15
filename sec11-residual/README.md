# sec11-residual

The material behind Supplementary Notes S5 and S6. It holds the residual that the births law with its series leaves, its tie to the ambiguity of the resummation of the divergent series, and the structure found in the coefficients of the law.

## What is here

- `scripts/` holds the chain solvers, the Borel and Padé resummation scripts, the closure tests and the readout scripts that the registration-log entries cited by the Notes name. `FILES.md` lists every script with the first sentence of its own docstring and the name it had when it ran.
- `logs/` holds the outputs: the stored events and their residuals, the fits, and the registered prediction files.

## How to read it

This folder is a record. The scripts are the ones that ran, and the file names match the registration log. Several tests in this part failed by the letter of their registered bar. Their entries stay in the log with the failure recorded, and the Notes list them in Section S7.

The log is in [`../registration-log/`](../registration-log/).

## Scope

The folder holds the files that the cited entries name and that the build found in the project tree. Outputs larger than one megabyte are not copied.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
