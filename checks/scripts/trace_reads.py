"""Run a script and record every file it opens for reading (4 Oct 2026, for the repository gate).
Usage: py trace_reads.py SCRIPT [OUT.json]. Writes the sorted list of absolute paths read (outside the Python install)."""
import builtins, io, os, sys, json, runpy

script = os.path.abspath(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else None
reads = set()
_open = builtins.open


def tracking_open(file, mode="r", *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)) and not any(c in mode for c in "wax+"):
        p = os.path.abspath(os.fsdecode(file))
        if "Python" not in p or "research startup" in p:
            reads.add(p)
    return _open(file, mode, *a, **k)


builtins.open = tracking_open
io.open = tracking_open
os.chdir(os.path.dirname(script))
sys.argv = [script]
sys.path.insert(0, os.path.dirname(script))
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit:
    pass
finally:
    builtins.open = _open
    res = sorted(p for p in reads if os.path.isfile(p) and os.path.normcase(p) != os.path.normcase(script))
    if out:
        _open(out, "w", encoding="utf-8").write(json.dumps(res, indent=0))
    else:
        print("\n".join(res))
