# The registration log

`PREREG_Q32.md` is the append-only log of the project, from 4 September to 4 October 2026. A prediction was written into it,
with its acceptance band, before the quantity was computed or read. The result was appended later as a separate entry that
names the prediction it scores. Nothing in the log was rewritten afterwards; corrections are new entries.

## How to read it

- Each entry starts with `## Amendment N (date, time)` (a few early entries use `## Note (...)` or `AMENDMENT N.`). Times are
  local machine time.
- A prediction entry says **PRE-REGISTERED** (or **REGISTERED**) and gives the predicted value and its band. **Blind** means the
  optimum did not exist yet when the prediction was written.
- A result entry quotes the prediction's number and gives the verdict: **PASS**, **FAIL**, or a third outcome stated in advance.
  Failures stay in the log; Table IX of the paper lists those that matter for its claims.
- `M` and `B` after a failure mean that it was traced to the mathematics (`M`) or to a bug in code, grid or precision (`B`), with
  the evidence given in the entry.

## Where to find the paper's results

Search the file for the amendment numbers below (for example `## Amendment 800 `).

| paper item | entries |
|---|---|
| ring losses predicted in advance (Tables II and III) | 1299-1302, 1305, 1320 (predictions); 1323-1385, 1399, 1409, 1413 (results) |
| the certified 32-atom ring (Theorem 1) | 1329-1335, 1401-1402 |
| center births on filled regions (Table V) | 1403-1409, 1442-1465 |
| rings inside rings (Table VI) | 1421-1479 |
| value laws in d dimensions (Table VII) | 1482-1576 |
| center events of d-balls (Section VIII) | 1577-1604 |
| the blind test against twelve published solutions (Section IX) | 800 |
| the one-dimensional count law and its forward tests (Section III) | 631-650, 725-757 |

## About this copy

This public copy omits notes on private correspondence and on the project's working arrangements, and rewords the sentences
that referred to them. No number, time stamp, prediction, band, result or verdict was changed. [`REDACTIONS.md`](REDACTIONS.md)
lists every entry that differs from the original and the kind of change. The original log is kept by the author.
