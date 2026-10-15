# sec3-proved-in-one-dimension

The material behind Section III-B and Appendix A of the paper: the lower bound of order A^(4/3) for the capacity-achieving input (Theorem 1).

## What is here

- `scripts/` holds `q1796_lower_bound_constants_arb.py`, the one numerical input of the proof. It evaluates the function J of Appendix A at 1.1648 and at 1.1647 in ball arithmetic (python-flint, 256 bits). From the two signs it derives the constants of the proof: the root bracket of J, the constant rho, the largest admissible c0 and the final constant 0.166075... Run it with `python q1796_lower_bound_constants_arb.py`. `FILES.md` lists the script with the first sentence of its own docstring.
- `logs/` holds its output, `q1796_out.txt`, whose last line is `CHECK q1796: ALL THREE STATEMENTS HOLD`.

## How to read it

This folder is a record. The proof of Theorem 1 is in Appendix A of the paper and uses no other computation: every other constant in it is a closed form or an inequality proved in the appendix.

The log is in [`../registration-log/`](../registration-log/).

## Scope

The folder holds the script and the output that Appendix A cites.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
