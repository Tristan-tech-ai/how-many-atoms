"""Record the python-flint version installed in the interpreter that runs the certificate scripts (py), and when it was installed.

The certificate (Amendment 1335, run 2 Oct 2026 21:07-21:12; re-run 1401 on 3 Oct) imports flint; no log of that run prints the version,
so this records the installed distribution and the time of its dist-info folder. Writes only env_versions.json in this folder.
"""
import os, json, datetime
import importlib.metadata as md
import flint

HERE = os.path.dirname(os.path.abspath(__file__))
d = md.distribution("python-flint")
info = os.path.getmtime(d._path)
major, minor = d.version.split(".")[:2]
out = dict(python_flint_version=d.version, python_flint_major_minor=float(f"{major}.{minor}"), flint_module_version=flint.__version__,
           dist_info=str(d._path), dist_info_mtime=datetime.datetime.fromtimestamp(info).isoformat(timespec="seconds"),
           note="installed before the certificate runs of 1335 (2 Oct) and 1401 (3 Oct); a single python-flint distribution is installed")
json.dump(out, open(os.path.join(HERE, "env_versions.json"), "w", encoding="utf-8"), indent=1)
print(out)
