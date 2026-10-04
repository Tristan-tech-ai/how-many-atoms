"""Readers for figure scripts: every plotted number is read by code from a primary file (machine output or the registration log),
never typed into a script. The figure gate (gate_figs.py) checks that figure scripts contain no data literals and that every file
they read is a primary source or a gate builder output.
"""
import os, re, json

TW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG = None


def path(rel):
    return os.path.join(TW, rel)


def num(rel, rx, group=1):
    """first match of regex rx in a text file (path relative to temporal_within), as float"""
    m = re.search(rx, open(path(rel), encoding="utf-8", errors="replace").read(), re.S)
    if not m:
        raise ValueError(f"pattern not found in {rel}: {rx}")
    return float(m.group(group))


def rows(rel, rx):
    """every match of regex rx in a text file, as tuples of floats (one tuple per match)"""
    t = open(path(rel), encoding="utf-8", errors="replace").read()
    return [tuple(float(g) for g in m.groups()) for m in re.finditer(rx, t, re.M)]


def amend(n, rx, group=1):
    """first match of regex rx in amendment n of the registration log PREREG_Q32.md, as float"""
    global _LOG
    if _LOG is None:
        _LOG = open(path("PREREG_Q32.md"), encoding="utf-8", errors="replace").read()
    b = re.search(r"^## Amendment %d\b.*?(?=^## Amendment )" % n, _LOG, re.M | re.S)
    m = re.search(rx, b.group(0), re.S) if b else None
    if not m:
        raise ValueError(f"pattern not found in amendment {n}: {rx}")
    return float(m.group(group))


def amend_rows(n, rx):
    """every match of regex rx inside amendment n only, as tuples of floats"""
    global _LOG
    if _LOG is None:
        _LOG = open(path("PREREG_Q32.md"), encoding="utf-8", errors="replace").read()
    b = re.search(r"^## Amendment %d\b.*?(?=^## Amendment )" % n, _LOG, re.M | re.S)
    if not b:
        raise ValueError(f"amendment {n} not found")
    return [tuple(float(g) for g in m.groups()) for m in re.finditer(rx, b.group(0), re.M)]


def jsonv(rel, *keys):
    d = json.load(open(path(rel), encoding="utf-8"))
    for k in keys:
        d = d[k]
    return d
