"""q1619_run_old_peak.py (10 Oct 2026): runs the OLD d20_order_n.py unchanged for a given NMAX and prints its peak working set at exit (the old script writes d20_out_<NMAX>.json in the current directory; use NMAX other than 5). Usage: py q1619_run_old_peak.py NMAX"""

import atexit, ctypes, runpy, sys, os


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak():
    pm = PMC()
    pm.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    ps = ctypes.WinDLL("psapi")
    ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    ps.GetProcessMemoryInfo.restype = ctypes.c_int
    ok = ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
    return pm.PeakWorkingSetSize / 1048576.0 if ok else -1.0


atexit.register(lambda: print("PEAK_WORKING_SET_MB %.0f" % peak(), flush=True))
assert int(sys.argv[1]) != 5, "NMAX 5 would overwrite d20_out_5.json"
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(here, "d20_order_n.py"), sys.argv[1]]
runpy.run_path(sys.argv[0], run_name="__main__")
