"""Number gate: every number printed in a paper must occur, to its printed precision, in a primary ground-truth source.

No judgement is involved. For each number in the body of the paper (bibliography excluded; it has its own gate) the code
  1. reads the LaTeX comment lines attached to the paragraph and collects the amendment numbers and file names they name;
  2. looks the value up in truth_index.pkl (registration log blocks, machine output files, secondary summaries);
  3. gives a verdict:
       PASS-ANCHORED   found in a primary source that the paragraph's comment names
       PASS-PRIMARY    found in a primary source (log or output file), but not in one the comment names
       DERIVED-OK      not printed in any source, but a declared formula over primary sources reproduces it (bindings.json)
       SECONDARY-ONLY  found only in summary documents (v0.10 text, README/READOUT .md): FAIL, no primary source
       PADDED          found only at lower precision than printed (e.g. 13.58 printed as 13.58200): FAIL
       NOT-FOUND       found nowhere: FAIL
       DERIVED-FAIL    the declared formula gives a different value: FAIL
     and, without a verdict, SMALL (integers up to 12, too common to test by occurrence), DATE, IDENT.
A number passes only on PASS-ANCHORED, PASS-PRIMARY or DERIVED-OK.

Run:  py gate_numbers.py ../paper_onelaw/paper1.tex ../paper_neuron/paper2.tex
Writes report_numbers.md and report_numbers.json next to this file; exit code 1 if any number fails.
"""
import os, re, sys, json, pickle, math, bisect, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
ROOT = os.path.dirname(TW)
IDX = pickle.load(open(os.path.join(HERE, "truth_index.pkl"), "rb"))
VALS, DECS, SRC, POS, RAW, SOURCES, AMEND = (IDX["vals"], IDX["decs"], IDX["src"], IDX["pos"], IDX["raw"], IDX["sources"],
                                             IDX["amendments"])

WORDS = {"zero": 0, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90, "hundred": 100}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
       "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
       "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40}
MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"


def split_comment(line):
    m = re.search(r"(?<!\\)%", line)
    return (line[:m.start()], line[m.start() + 1:]) if m else (line, "")


def body_lines(tex):
    lines = tex.split("\n")
    a = next(i for i, l in enumerate(lines) if "\\begin{document}" in l)
    b = next((i for i, l in enumerate(lines) if "\\begin{thebibliography}" in l), len(lines))
    return [(i + 1, lines[i]) for i in range(a + 1, b)]


def paragraphs(lines):
    """Group lines into paragraphs; comment-only lines attach to the paragraph above them."""
    paras, cur = [], None
    for no, line in lines:
        text, com = split_comment(line)
        if not text.strip() and not com.strip():
            cur = None
            continue
        if not text.strip():
            if paras:
                paras[-1]["comment"] += " " + com
            continue
        if cur is None:
            cur = {"lines": [], "comment": ""}
            paras.append(cur)
        cur["lines"].append((no, text))
        if com.strip():
            cur["comment"] += " " + com
    return paras


def comment_refs(com):
    amend, files = set(), set()
    c = re.sub(r"\d{1,2}:\d{2}|\d{1,2}:\dx|v\d+\.\d+|\d+\.\d+", " ", com)
    for m in re.finditer(r"(?<![\w.-])(\d{2,4})(?![\w.])", c):
        n = int(m.group(1))
        if n in AMEND:
            amend.add(n)
    for m in re.finditer(r"[\w./-]+\.(?:json|log|txt|out|py)", com):
        files.add(m.group(0).split("/")[-1])
    return amend, files


def clean_tex(s):
    s = re.sub(r"\\(cite|ref|eqref|label|url|includegraphics|setlength|begin|end|multicolumn|cline|hspace|vspace)\*?(\[[^\]]*\])?(\{[^{}]*\})+",
               " ", s)
    s = re.sub(r"p\{0\.\d+\\(column|text)width\}", " ", s)
    s = re.sub(r"width=\\(column|text)width", " ", s)
    s = re.sub(r"\\(section|subsection|title|author|thanks|emph|textbf|caption)\*?", " ", s)
    s = s.replace("~", " ").replace("\\,", " ").replace("\\;", " ").replace("{,}", "")
    return s


def scan_numbers(text):
    """Yield (kind, printed, value, eff_decimals, span) for numbers in cleaned LaTeX text."""
    out = []
    t = text
    sci = re.compile(r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{?\s*([-\u2212]?\d+)\s*\}?")
    for m in sci.finditer(t):
        mant, ex = m.group(1), int(m.group(2).replace("\u2212", "-"))
        md = len(mant.split(".")[1]) if "." in mant else 0
        out.append(("num", m.group(0), float(mant) * 10 ** ex, md - ex, m.span()))
    t = sci.sub(lambda m: " " * len(m.group(0)), t)
    p10 = re.compile(r"(?<![\d.])10\^\{\s*([-\u2212]?\d+)\s*\}")
    for m in p10.finditer(t):
        ex = int(m.group(1).replace("\u2212", "-"))
        out.append(("num", m.group(0), 10.0 ** ex, -ex, m.span()))
    t = p10.sub(lambda m: " " * len(m.group(0)), t)
    t = re.sub(r"[\^_]\{[^{}]*\}|[\^_]\\?\w", lambda m: " " * len(m.group(0)), t)
    for m in re.finditer(r"(?<![\w.\\])(\d+(?:\.\d+)?)(?:[eE]([-+]?\d+))?(?![\w])", t):
        s = m.group(1)
        d = len(s.split(".")[1]) if "." in s else 0
        ex = int(m.group(2)) if m.group(2) else 0
        out.append(("num", m.group(0), float(s) * 10 ** ex, d - ex, m.span()))
    for m in re.finditer(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(one|two|three|four|five|six|seven|eight|nine|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b", t, re.I):
        a = WORDS[m.group(1).lower()]
        u = m.group(2).lower()
        b = WORDS.get(u, ORD.get(u))
        out.append(("word", m.group(0), a + b, 0, m.span()))
    t2 = re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-\w+\b", lambda m: " " * len(m.group(0)), t, flags=re.I)
    for m in re.finditer(r"\b(" + "|".join(list(WORDS) + list(ORD)) + r")\b", t2, re.I):
        w = m.group(1).lower()
        out.append(("word", m.group(0), WORDS.get(w, ORD.get(w)), 0, m.span()))
    return sorted(out, key=lambda x: x[4][0])


def classify(kind, printed, val, dec, ctx_before, ctx_after):
    if re.search(MONTHS + r"\s*$", ctx_before) or re.match(r"\s*" + MONTHS, ctx_after):
        return "DATE"
    if kind == "num" and re.fullmatch(r"(19|20)\d\d", printed) and re.search(r"(et al\.|and [A-Z]\w+|[A-Z]\w+)\W*$", ctx_before):
        return "DATE"
    if kind == "num" and (re.fullmatch(r"0\d{3,}", printed) or re.search(r"(DANDI|Dryad|ORCID|arXiv)[\W\d-]*$", ctx_before)
                          or re.search(r"\bversion\s*$", ctx_before) or re.search(r"\d{4}-$", ctx_before)):
        return "IDENT"
    if re.match(r"\s*(September|October)\s+2026", ctx_after) or re.fullmatch(r"202[0-9]", printed):
        return "DATE"
    # numbers of log amendments cited in a table cell or after the word "amendment": identifiers, not results
    if kind == "num" and re.fullmatch(r"\d{2,4}", printed) and int(printed) in ALL_AMEND and \
            (re.search(r"&[^&]*$", ctx_before) and re.match(r"[\d,\s\-]*(\\\\|$)", ctx_after) or re.search(r"[Aa]mendments?\s*$", ctx_before)):
        return "IDENT"
    if dec <= 0 and val <= 12 and float(val).is_integer():
        return "SMALL"
    return None


def lookup(val, dec):
    tol = 0.5 * 10 ** (-dec) * (1 + 1e-9) + 1e-300
    lo = bisect.bisect_left(VALS, val - tol)
    hi = bisect.bisect_right(VALS, val + tol)
    return range(lo, hi), tol


def sig_digits(printed):
    m = re.search(r"\d+(?:\.\d+)?", printed)
    s = m.group(0).replace(".", "").lstrip("0") if m else ""
    return len(s)


SRC_SIZE = np.bincount(SRC, minlength=len(SOURCES))
ALL_AMEND = {int(m.group(1)) for s in SOURCES for m in [re.match(r"Amendment (\d+)", s[1])] if m}


def declared_sids(amend, files):
    out = set()
    for n in amend:
        out.update(AMEND.get(n, []))
    for j, s in enumerate(SOURCES):
        if s[0] in ("PRIMARY-OUT", "CODE", "LIT") and os.path.basename(s[2]) in files:
            out.add(j)
    return out


def primary_hits(val, dec):
    rng, _ = lookup(abs(val), dec)
    return [i for i in rng if DECS[i] >= dec and SOURCES[SRC[i]][0] in ("PRIMARY-LOG", "PRIMARY-OUT")]


def cooccur_sids(numbers, min_count=3, max_size=2000):
    """Sources that contain at least min_count different numbers of one paragraph, each with >= 5 significant digits.
    Restricted to small sources (<= max_size number tokens) so that a coincidence is implausible."""
    count = {}
    for val, dec, sig in numbers:
        if sig < 5:
            continue
        for sid in {int(SRC[i]) for i in primary_hits(val, dec)}:
            if SRC_SIZE[sid] <= max_size:
                count[sid] = count.get(sid, 0) + 1
    return {sid for sid, c in count.items() if c >= min_count}


def verdict(val, dec, declared, inferred, sig=0):
    rng, tol = lookup(abs(val), dec)
    anch, coo, prim, sec, padded = [], [], [], [], []
    for i in rng:
        s = SOURCES[SRC[i]]
        if s[0] in ("CODE", "LIT") and int(SRC[i]) not in declared:
            continue
        if DECS[i] < dec:
            padded.append(i)
            continue
        if s[0] == "SECONDARY":
            sec.append(i)
        elif int(SRC[i]) in declared:
            anch.append(i)
        elif int(SRC[i]) in inferred and sig >= 4:
            coo.append(i)
        else:
            prim.append(i)
    if anch:
        return "PASS-ANCHORED", anch
    if coo:
        return "PASS-COOCCUR", coo
    if prim:
        return ("PASS-UNIQUE" if sig >= 5 else "UNANCHORED"), prim
    if sec:
        return "SECONDARY-ONLY", sec
    if padded:
        return "PADDED", padded
    return "NOT-FOUND", []


def where(hits, k=3):
    seen, out = set(), []
    for i in hits:
        lab = SOURCES[SRC[i]][1]
        if lab not in seen:
            seen.add(lab)
            out.append(f"{lab} ({RAW[i]})")
        if len(out) >= k:
            break
    more = len({SOURCES[SRC[i]][1] for i in hits}) - len(out)
    return "; ".join(out) + (f"; +{more} more" if more > 0 else "")


def load_bindings():
    import glob as _g
    out = []
    for p in sorted(_g.glob(os.path.join(HERE, "bindings*.json"))):
        out += json.load(open(p, encoding="utf-8"))
    return out


def amend_text(n):
    log = open(os.path.join(TW, "PREREG_Q32.md"), encoding="utf-8", errors="replace").read()
    parts = []
    for sid in AMEND.get(n, []):
        a = SOURCES[sid][3]
        nxt = [SOURCES[j][3] for j in range(len(SOURCES)) if SOURCES[j][0] == "PRIMARY-LOG" and SOURCES[j][3] > a]
        parts.append(log[a:min(nxt) if nxt else len(log)])
    return "\n".join(parts)


def eval_binding(b):
    """Formula over primary sources. L(n, regex) = first captured number in amendment n; J(path, *keys) = JSON value."""
    def L(n, rx):
        m = re.search(rx, amend_text(n), re.S)
        if not m:
            raise ValueError(f"pattern not found in amendment {n}: {rx}")
        return float(m.group(1))

    def J(path, *keys):
        d = json.load(open(os.path.join(ROOT, path), encoding="utf-8"))
        for k in keys:
            d = d[k]
        return float(d)

    def T(path, rx):
        """first captured number of regex rx in a primary text output (log) file, path relative to the research folder"""
        m = re.search(rx, open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read(), re.S)
        if not m:
            raise ValueError(f"pattern not found in {path}: {rx}")
        return float(m.group(1))
    env = {"L": L, "J": J, "T": T, "math": math, "sqrt": math.sqrt, "log": math.log, "exp": math.exp, "abs": abs, "min": min,
           "max": max, "sum": sum}
    return eval(b["expr"], {"__builtins__": {}}, env)


def run(paths):
    bindings = load_bindings()
    rows = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        tex = open(path, encoding="utf-8").read()
        for para in paragraphs(body_lines(tex)):
            amend, files = comment_refs(para["comment"])
            declared = declared_sids(amend, files)
            nums = []
            for no, raw in para["lines"]:
                text = clean_tex(raw)
                for kind, printed, val, dec, (a, b) in scan_numbers(text):
                    if not classify(kind, printed, val, dec, text[max(0, a - 40):a], text[b:b + 40]):
                        nums.append((val, dec, sig_digits(printed)))
            inferred = cooccur_sids(nums, min_count=max(3, math.ceil(0.2 * sum(1 for x in nums if x[2] >= 5)))) - declared
            for no, raw in para["lines"]:
                text = clean_tex(raw)
                for kind, printed, val, dec, (a, b) in scan_numbers(text):
                    before, after = text[max(0, a - 40):a], text[b:b + 40]
                    ctx = re.sub(r"\s+", " ", text[max(0, a - 70):b + 50]).strip()
                    cls = classify(kind, printed, val, dec, before, after)
                    rid = hashlib.md5(f"{name}|{re.sub(r'\s+', ' ', raw.strip())}|{printed}|{a}".encode()).hexdigest()[:10]
                    row = dict(id=rid, paper=name, line=no, printed=printed.strip(), value=val, decimals=dec, context=ctx,
                               anchors=sorted(amend), files=sorted(files))
                    if cls:
                        row["verdict"] = cls
                        rows.append(row)
                        continue
                    v, hits = verdict(val, dec, declared, inferred, sig_digits(printed))
                    row["verdict"], row["where"] = v, where(hits)
                    row["weak"] = v == "PASS-ANCHORED" and sig_digits(printed) <= 2
                    veto = [bd for bd in bindings if bd.get("kind") == "unverified" and bd["paper"] == name
                            and bd["printed"] == row["printed"] and bd["context"] in raw]
                    if veto:
                        row["verdict"], row["where"] = "UNVERIFIED", veto[0]["why"]
                        rows.append(row)
                        continue
                    # a declared binding (an exact formula over primary sources) takes precedence over occurrence matching,
                    # so an accidental match of a low-precision number cannot stand in for it
                    if True:
                        for bd in bindings:
                            if bd.get("kind") != "unverified" and bd["paper"] == name and bd["printed"] == row["printed"] and bd["context"] in raw:
                                row["weak"] = False
                                try:
                                    got = eval_binding(bd)
                                    ok = abs(abs(got) - abs(val)) <= 0.5 * 10 ** (-dec) * (1 + 1e-9)
                                    row["verdict"] = (("DEFINED" if bd.get("kind") == "definition" else "DERIVED-OK") if ok else "DERIVED-FAIL")
                                    row["where"] = f"{bd['expr']} = {got:.10g}"
                                except Exception as e:
                                    row["verdict"], row["where"] = "DERIVED-FAIL", f"binding error: {e}"
                                break
                    rows.append(row)
    return rows


def report(rows):
    order = ["NOT-FOUND", "PADDED", "SECONDARY-ONLY", "DERIVED-FAIL", "UNVERIFIED", "UNANCHORED", "PASS-UNIQUE", "PASS-COOCCUR", "DERIVED-OK", "DEFINED",
             "PASS-ANCHORED", "SMALL", "DATE", "IDENT"]
    fail = {"NOT-FOUND", "PADDED", "SECONDARY-ONLY", "DERIVED-FAIL", "UNANCHORED", "UNVERIFIED"}
    papers = sorted({r["paper"] for r in rows})
    md = ["# Number gate report", "", "Verdicts are computed by gate_numbers.py against truth_index.pkl; no judgement is involved.", ""]
    md.append("| paper | " + " | ".join(order) + " |")
    md.append("|---|" + "---|" * len(order))
    for p in papers:
        md.append(f"| {p} | " + " | ".join(str(sum(1 for r in rows if r['paper'] == p and r['verdict'] == o)) for o in order) + " |")
    for o in order[:10]:
        sel = [r for r in rows if r["verdict"] == o]
        if not sel:
            continue
        md += ["", f"## {o} ({len(sel)})", "", "| paper | line | printed | context | comment anchors | found in |", "|---|---|---|---|---|---|"]
        for r in sel:
            ctx = r["context"].replace("|", "/")
            md.append(f"| {r['paper']} | {r['line']} | {r['printed']} | {ctx} | {', '.join(map(str, r['anchors']))} "
                      f"{' '.join(r['files'])} | {r.get('where', '').replace('|', '/')} |")
    open(os.path.join(HERE, "report_numbers" + os.environ.get("GATE_TAG", "") + ".md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    json.dump(rows, open(os.path.join(HERE, "report_numbers" + os.environ.get("GATE_TAG", "") + ".json"), "w", encoding="utf-8"), indent=1)
    weak = [r for r in rows if r.get("weak")]
    md += ["", f"## Weakly verified ({len(weak)}): passed by a declared source, but with two or fewer significant digits, so a "
           "coincidental match in that source cannot be excluded by code; read these", "", "| paper | line | printed | context | found in |",
           "|---|---|---|---|---|"]
    md += [f"| {r['paper']} | {r['line']} | {r['printed']} | {r['context'].replace('|', '/')} | {r.get('where', '').replace('|', '/')} |"
           for r in weak]
    open(os.path.join(HERE, "report_numbers" + os.environ.get("GATE_TAG", "") + ".md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    nfail = sum(1 for r in rows if r["verdict"] in fail)
    for p in papers:
        c = {o: sum(1 for r in rows if r["paper"] == p and r["verdict"] == o) for o in order}
        print(p, c)
    print("FAIL" if nfail else "PASS", nfail, "numbers without a primary source")
    # 4 Oct 2026: weak numbers fail the gate too. Rewording a sentence silently unbinds a number (its binding context no longer
    # matches the line) and it falls back to a low-precision occurrence match; the summary used to hide that (23 such numbers).
    print("FAIL" if weak else "PASS", len(weak), "weakly verified numbers (binding needed, or a binding context no longer matches)")
    return nfail + len(weak)


if __name__ == "__main__":
    sys.exit(1 if report(run(sys.argv[1:])) else 0)
