"""merge_1596.py: merged event lists for the 1590/1591/1593/1597 readouts, joining CHAIN entries that numpy printing wrapped over
several lines (continuation lines start with whitespace). Replaces merge_1596.sh (which cut wrapped entries).
"""

import re, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def entries(f, d, emax=None):
    t = re.sub(r"\n[ \t]+", " ", open(f, encoding="utf-8", errors="replace").read())
    out = []
    for line in t.split("\n"):
        g = re.match(r"CHAIN d %d e(\d+) " % d, line)
        if g and (emax is None or int(g.group(1)) <= emax):
            m = re.match(
                r"(CHAIN d \d+ e\d+ \w+: m [0-9.]+; support \[[^\]]*\]; weights \[[^\]]*\]; residual [0-9.e+-]+)",
                line,
            )
            if m:
                out.append(m.group(1) + "\n")
    return out


def write(name, parts):
    open(name, "w", encoding="utf-8").write("".join(parts))
    print(name, len(parts))


write(
    "chain_d3_merged_1595.log", entries("chain_d3_long.log", 3, emax=10) + entries("chain_d3_long_b.log", 3)
)
write("chain_d2_merged_1596.log", entries("chain_d2_long_part1.log", 2) + entries("chain_d2_long_b.log", 2))
write("chain_d4_merged_1596.log", entries("chain_d4_long.log", 4) + entries("chain_d4_long_b.log", 4))
write("chain_d5_merged_1597.log", entries("chain_d5_long.log", 5) + entries("chain_d5_long_b.log", 5))
