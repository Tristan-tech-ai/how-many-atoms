# sec15-open-problems

The material behind Section IX of the paper: what is open, with the registered tests and instruments that the open problems refer to.

## What is here

- `scripts/` holds the scripts of the registered tests and instruments that the registration-log entries cited by Section IX name. `FILES.md` lists every script with the first sentence of its own docstring and the name it had when it ran.
- `logs/` holds the registration files and outputs of those scripts that existed when the paper was written.

## How to read it

This folder is a record. The scripts are the ones that ran, and the file names match the registration log. A registration file is a text written before the quantity it predicts was computed; the log entry that cites it gives its hash.

The log is in [`../registration-log/`](../registration-log/).

## Scope

The folder holds the files that the cited entries name and that the build found in the project tree. Outputs larger than one megabyte are not copied.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
