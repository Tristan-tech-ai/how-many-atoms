"""Build the ground-truth number index used by the paper gate.

Ground truth, in order of authority:
  PRIMARY-LOG   the append-only registration log PREREG_Q32.md, split into amendment blocks;
  PRIMARY-OUT   machine output files written by the scripts (results*.json, *.json, *.log, *_out.txt);
  SECONDARY     documents written as summaries (v0.10 text, READOUT/summary .md files). A number found only here does not pass.

Every number token in these files is stored with its value, its printed decimals and its location, so that the gate can ask
"does this printed number occur, to its printed precision, in a primary source?" without any judgement.

Run:  py truth_index.py            (writes truth_index.pkl next to this file)
"""
import os, re, pickle, glob, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
ROOT = os.path.dirname(TW)
LOG = os.path.join(TW, "PREREG_Q32.md")

NUM = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?(?![\w])")
SKIP_DIRS = {"review", "paper_onelaw", "paper_neuron", "public_log_tools", "gate_truth", "arxiv_src", "__pycache__", ".git",
             "node_modules"}
OUT_EXT = (".json", ".log", "_out.txt", ".out", "verdict.txt")
# amendments whose heading trips the writing filter but which record new computations; each has a stated reason
_ov = os.path.join(HERE, "computation_amendments.json")
COMPUTATION_OVERRIDE = set(json.load(open(_ov, encoding="utf-8"))) if os.path.exists(_ov) else set()
WRITING = (r"paper|referee|draft|v0\.\d|arxiv|abstract|e-?mail|letter|cold test|prose|wording|correspondence|"
           r"acknowledg|repository|readability|endorse")


def tokens(text):
    for m in NUM.finditer(text):
        s = m.group(0)
        try:
            v = float(s)
        except ValueError:
            continue
        mant = re.split(r"[eE]", s)[0]
        dec = len(mant.split(".")[1]) if "." in mant else 0
        exp = int(re.split(r"[eE]", s)[1]) if re.search(r"[eE]", s) else 0
        yield m.start(), s, abs(v), dec - exp


def split_log(text):
    """Return list of (amendment_number, start, end). Headers look like '## Amendment 117 (...'."""
    heads = [(m.start(), int(m.group(1)), m.group(1) + m.group(2)) for m in re.finditer(r"^## Amendment (\d+)([a-z]?)", text, re.M)]
    blocks = [(0, "0 (preamble)", 0, heads[0][0])] if heads and heads[0][0] > 0 else []
    for i, (pos, n, label) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        blocks.append((n, label, pos, end))
    return blocks


def output_files():
    roots = [TW, os.path.join(ROOT, "glm_contrib1")] + glob.glob(os.path.join(ROOT, "_archive", "lanes", "*"))
    seen = set()
    # tables built by code from primary files only (see their scripts in this folder) count as primary output
    for f in ("table2_primary.json", "table3_primary.json", "table5_primary.json", "paper2_derived.json"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            seen.add(p)
            yield p
    for r in roots:
        for dp, dns, fns in os.walk(r):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for f in fns:
                if f.endswith(OUT_EXT) and f != "truth_index.pkl":
                    p = os.path.join(dp, f)
                    if p not in seen and os.path.getsize(p) < 50_000_000:
                        seen.add(p)
                        yield p


def secondary_files():
    files = [os.path.join(TW, "paper_neuron", "v010_TIT.txt")]
    for r in [TW, os.path.join(ROOT, "glm_contrib1")] + glob.glob(os.path.join(ROOT, "_archive", "lanes", "*")):
        for dp, dns, fns in os.walk(r):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for f in fns:
                if f.endswith(".md") and os.path.join(dp, f) != LOG:
                    files.append(os.path.join(dp, f))
    return [f for f in files if os.path.exists(f)]


def main():
    sources = []          # (kind, label, path)
    vals, decs, src, pos, raw = [], [], [], [], []
    log = open(LOG, encoding="utf-8", errors="replace").read()
    blocks = split_log(log)
    amend = {}
    for n, label, a, b in blocks:
        sid = len(sources)
        head = log[a:log.find("\n", a)]
        # amendments about writing (drafts, referees, papers, correspondence) restate numbers; they are not computation records
        writing = re.search(WRITING, head, re.I) is not None and label not in COMPUTATION_OVERRIDE
        if not writing:
            amend.setdefault(n, []).append(sid)
        sources.append(("SECONDARY" if writing else "PRIMARY-LOG", f"Amendment {label}" + (" (writing)" if writing else ""), LOG, a))
        for p, s, v, d in tokens(log[a:b]):
            vals.append(v); decs.append(d); src.append(sid); pos.append(a + p); raw.append(s)
    nout = 0
    for p in output_files():
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        sid = len(sources)
        sources.append(("PRIMARY-OUT", os.path.relpath(p, ROOT), p, 0))
        nout += 1
        for q, s, v, d in tokens(t):
            vals.append(v); decs.append(d); src.append(sid); pos.append(q); raw.append(s)
    # CODE (scripts: method parameters) and LIT (extracted texts of cited works): they count only where a paragraph declares them
    code_lit = []
    for r in [TW, os.path.join(ROOT, "glm_contrib1")] + glob.glob(os.path.join(ROOT, "_archive", "lanes", "*")):
        for dp, dns, fns in os.walk(r):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for f in fns:
                if f.endswith(".py"):
                    code_lit.append(("CODE", os.path.join(dp, f)))
                elif f.endswith(".txt") and os.sep + "papers" in dp:
                    code_lit.append(("LIT", os.path.join(dp, f)))
    for kind, p in code_lit:
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        sid = len(sources)
        sources.append((kind, os.path.relpath(p, ROOT), p, 0))
        for q, s, v, d in tokens(t):
            vals.append(v); decs.append(d); src.append(sid); pos.append(q); raw.append(s)
    for p in secondary_files():
        t = open(p, encoding="utf-8", errors="replace").read()
        sid = len(sources)
        sources.append(("SECONDARY", os.path.relpath(p, ROOT), p, 0))
        for q, s, v, d in tokens(t):
            vals.append(v); decs.append(d); src.append(sid); pos.append(q); raw.append(s)
    vals = np.array(vals); order = np.argsort(vals, kind="stable")
    idx = dict(vals=vals[order], decs=np.array(decs)[order], src=np.array(src)[order], pos=np.array(pos)[order],
               raw=[raw[i] for i in order], sources=sources, amendments=amend)
    pickle.dump(idx, open(os.path.join(HERE, "truth_index.pkl"), "wb"))
    print(f"log blocks {len(blocks)}, output files {nout}, secondary files {sum(1 for s in sources if s[0] == 'SECONDARY')}, "
          f"numbers {len(vals)}")


if __name__ == "__main__":
    main()
