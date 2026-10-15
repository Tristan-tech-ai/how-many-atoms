# sec9-tower-of-events

The material behind Section IV-A of the paper: following one optimum of the peak-limited channel as the amplitude grows, and the events (births and splits) at the origin that end each branch.

## What is here

- `scripts/` holds the solvers, the continuation codes that follow a branch, and the readouts that the registration-log entries cited by Section IV-A name. `FILES.md` lists every script with the first sentence of its own docstring, the name it had when it ran, and the sections of the paper (or the Supplementary Notes, S1 and so on) that cite it.
- `logs/` holds the outputs of those runs: family tables, birth and split outputs, trace logs and the registered prediction files. A `__` in a name stands for a folder separator of the original tree.

## How to read it

This folder is a record, like the folders of the closed curves, the certificate and the filled regions. The scripts are the ones that ran. File names are the original names, so they match the names in the registration log. A script that starts a chain expects the folder layout of the original project, so rerunning one usually means adjusting a path at the top.

Every prediction was written into the registration log before the run it concerns. The log is in [`../registration-log/`](../registration-log/). The entries cited by Section IV-A are the entry point: each names its script, its output and its registered bar.

## Scope

The folder holds the files that the cited entries name and that the build found in the project tree. Outputs larger than one megabyte are not copied.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
