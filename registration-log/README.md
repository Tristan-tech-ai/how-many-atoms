# The registration log

`PREREG_Q32.md` is the append-only log of the project, from 4 September to 13 October 2026. A prediction was written into it,
with its acceptance band, before the quantity was computed or read. The result was appended later as a separate entry that
names the prediction it scores. Nothing in the log was rewritten afterwards; corrections are new entries.

## How to read it

- Each entry starts with `## Amendment N (date, time)` (a few early entries use `## Note (...)` or `AMENDMENT N.`). Times are
  local machine time.
- A prediction entry says **PRE-REGISTERED** (or **REGISTERED**) and gives the predicted value and its band. **Blind** means the
  optimum did not exist yet when the prediction was written.
- A result entry quotes the prediction's number and gives the verdict: **PASS**, **FAIL**, or a third outcome stated in advance.
  Failures stay in the log; Table VII of the paper and Table S7 of the Supplementary Notes list those that matter for their claims.
- `M` and `B` after a failure mean that it was traced to the mathematics (`M`) or to a bug in code, grid or precision (`B`), with
  the evidence given in the entry.

## Where to find the paper's results

Search the file for the amendment numbers below (for example `## Amendment 800 `). `S1`, `S2`, ... are the sections of the
Supplementary Notes that are submitted with the paper.

| paper item | entries |
|---|---|
| the one-dimensional count law and its forward tests (Section III-A) | 631-650, 725-757 |
| the ball-arithmetic input of the lower bound of order A^(4/3) (Theorem 1, Section III-B, Appendix A) | 2236 |
| ring losses predicted in advance (Section VI, Tables III and IV) | 1299-1302, 1305, 1320 (predictions); 1323-1385, 1399, 1409, 1413 (results) |
| the certified 32-atom ring (Theorem 3, Section VII, Appendix E) | 1329-1335, 1401-1402 |
| center births on filled regions (Section VIII, Table VI) | 1403-1409, 1442-1465, 2252-2254 |
| the tower of events (Section IV-A, Table II) | 1686, 1840, 1925, 1940, 2004, 2050, 2152, 2224-2226 |
| the births law and its blind tests (Section IV-B) | 1820, 1916, 1931, 1956, 1962, 1965-1966, 1968-1969, 1972-1973, 2004, 2007, 2009, 2016, 2021-2022, 2038, 2040, 2160, 2188, 2202, 2221, 2224-2226 |
| total positivity (Section V-B, Appendices C and D) | 1748-1749, 1752, 1756-1757, 1760, 1763-1764, 1801, 1806-1807, 1815, 1818-1819, 1823-1824, 1829-1831, 1837, 1839, 1843, 1850-1851, 1854, 1894, 1896, 1920 |
| the monotonicity reduction (Section V-A, Appendix B; statement (S1) at the exact states, Appendix D) | 2128-2136, 2138-2140, 2148-2151, 2175-2178, 2182, 2189, 2203-2204, 2255-2257 |
| open problems and registered tests (Section IX, Table VII) | 2152, 2226-2229 |
| value laws in d dimensions (S1, Table S1) | 1482-1576 |
| center events of d-balls (S2) | 1577-1604 |
| the continuation past the birth and rings inside rings (S3, S4, Table S5) | 1414-1479 |
| the residual of the births law and the structure of its coefficients (S5, S6) | 1973-1974, 2036-2037, 2043, 2046, 2049, 2052-2056, 2066-2067, 2070-2072, 2075-2077, 2079, 2085-2086, 2121-2123 |
| what does not carry over, and the blind test against twelve published solutions (S7) | 800, 1574, 2152, 2226-2229 |
| deformation identity of the planar NPMLE (S8) | 1482, 1504-1505, 1508, 1520 |
| one-minimum lemma (Appendix C) | 2128, 2130 |

## About this copy

This public copy omits notes on private correspondence and on the project's working arrangements, and rewords the sentences
that referred to them. Entries that only record the drafting of the papers keep their header and are marked omitted, and the
first person singular of the working log is written as we. No prediction, band, result or verdict of the research was changed. [`REDACTIONS.md`](REDACTIONS.md)
lists every entry that differs from the original and the kind of change. Some entries are replaced by a note that their
content is omitted: their headers keep the number and the time stamp. The original log is kept by the author.

The folder [`timestamps/`](timestamps/) holds independent time stamps of this log from 14 October 2026 on; its README gives the commands
that check them.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
