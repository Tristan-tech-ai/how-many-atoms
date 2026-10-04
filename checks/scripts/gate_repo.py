"""Repository gate (4 Oct 2026): the public repositories are checked by code against the gated papers.

Tristan asked whether the repositories pass the same gates as the papers. They had not been checked. Checks:
  R1  the paper PDFs in the repositories are byte-identical to the gated builds (papers and the supplement)
  R2  the title in each repository README equals the paper's title
  R3  every cross-reference in the repository docs (Table, Fig., Section, Theorem, Proposition, Conjecture, Lemma, Appendix)
      exists in the paper's .aux; each one is listed with what it points to, for a human read
  R4  stale wording: a 6-word phrase of a repository doc that occurs in an earlier version of the paper (the backups of 4 Oct)
      but no longer in the current paper
  R5  prose: ../../cek_en2.py on every README.md, registration-log/README.md and REDACTIONS.md, and on FILES.md outside its tables
      (the tables quote the scripts' own docstrings)
  R6  numbers in the repository docs: each number (integers up to 12, years and amendment numbers excepted) occurs in a primary
      source that the public material carries (R8 below), or in the current paper
  R7  file integrity: every data, log and result file of a repository equals a local original byte for byte (line endings
      normalised); every script has the same program (Python AST) as a local original of the same name
  R8  traceability: every number that the number gate passes in a paper rests on at least one source the public material carries:
      a repository file identical to that source, or an amendment present in the public copy of the registration log
  R9  every amendment cited in a paper's body text or in a repository doc is present in the public log copy
Writes report_repo.md; exit 1 if any check fails.
"""
import os, re, sys, json, glob, hashlib, ast, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
ROOT = os.path.dirname(TW)
sys.path.insert(0, HERE)
import gate_numbers as gn  # loads truth_index.pkl; no side effects beyond that

REPOS = {"paper1": os.path.join(ROOT, "how-many-atoms"), "paper2": os.path.join(ROOT, "how-many-firing-rates")}
PAPERS = {"paper1": os.path.join(TW, "paper_onelaw", "paper1"), "paper2": os.path.join(TW, "paper_neuron", "paper2")}
PDFS = {"paper1": [("paper/how-many-atoms.pdf", "paper_onelaw/paper1.pdf"),
                   ("paper/supplement-inner-rings.pdf", "paper_onelaw/supplement_inner_rings.pdf")],
        "paper2": [("paper/how-many-firing-rates.pdf", "paper_neuron/paper2.pdf")]}
PUBLIC_LOG = os.path.join(REPOS["paper1"], "registration-log", "PREREG_Q32.md")
DOC_RX = re.compile(r"(README|FILES|REDACTIONS)\.md$")
SKIP_SUFFIX = (".md", ".pdf", ".gitignore", "LICENSE", "requirements.txt")


def NP(p):
    return os.path.normcase(os.path.normpath(p))


def git_files(repo):
    out = subprocess.run(["git", "-C", repo, "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True,
                         text=True).stdout.split("\n")
    return [f for f in out if f]


def norm_bytes(p):
    b = open(p, "rb").read()
    return b.replace(b"\r\n", b"\n")


def sha(p):
    return hashlib.sha256(norm_bytes(p)).hexdigest()


def py_ast(p):
    try:
        return ast.dump(ast.parse(open(p, encoding="utf-8", errors="replace").read()), include_attributes=False)
    except SyntaxError:
        return None


def local_index():
    """basename -> local paths under the research folder, outside the repositories and archives of git metadata"""
    idx = {}
    skip = {os.path.normcase(r) for r in REPOS.values()}
    for d, subs, files in os.walk(ROOT):
        if os.path.normcase(d) in skip or "\\.git" in d or "/.git" in d or "node_modules" in d or "_repo_rebuild" in d:
            subs[:] = []
            continue
        for f in files:
            idx.setdefault(f, []).append(os.path.join(d, f))
    return idx


def aux_labels(base):
    labs = {}
    for m in re.finditer(r"\\newlabel\{(\w+):([^}]*)\}\{\{(?:\\mbox\s*\{)?([^{}]*)\}?\}", open(base + ".aux", encoding="utf-8", errors="replace").read()):
        labs[(m.group(1), m.group(3).strip())] = m.group(1) + ":" + m.group(2)
    return labs


def captions(base):
    tex = open(base + ".tex", encoding="utf-8").read()
    caps = {}
    for m in re.finditer(r"\\caption\{(.{0,160}?)\}\\label\{([^}]*)\}", tex, re.S):
        caps[m.group(2)] = re.sub(r"\s+", " ", m.group(1))
    for m in re.finditer(r"\\caption\{([^\n]{0,160})", tex):
        lab = re.search(r"\\label\{([^}]*)\}", tex[m.start():m.start() + 2000])
        if lab and lab.group(1) not in caps:
            caps[lab.group(1)] = re.sub(r"\s+", " ", m.group(1))
    for m in re.finditer(r"\\(section|subsection)\*?\{([^}]*)\}\\label\{([^}]*)\}", tex):
        caps[m.group(3)] = m.group(2)
    return caps


KIND = {"Table": "tab", "Tables": "tab", "Fig.": "fig", "Figure": "fig", "Section": "sec", "Sections": "sec", "Theorem": "thm",
        "Proposition": "prop", "Conjecture": "conj", "Lemma": "lem", "Appendix": "app"}
REF_RX = re.compile(r"\b(Tables?|Fig\.|Figure|Sections?|Theorem|Proposition|Conjecture|Lemma|Appendix)\s+([IVX]+|\d+|[A-Z])\b"
                    r"((?:\s*(?:,|and|to)\s*(?:[IVX]+|\d+))*)")


def main():
    rep, fails = ["# Repository gate report", "", "Computed by gate_repo.py; R3 and R4 lists are for a human read.", ""], {}
    idx = local_index()
    pub_amend = {int(m.group(1)) for m in re.finditer(r"^## Amendment (\d+)", open(PUBLIC_LOG, encoding="utf-8").read(), re.M)}
    repo_files = {k: git_files(r) for k, r in REPOS.items()}

    # ---- R1
    rows = []
    for p, pairs in PDFS.items():
        for rp, lp in pairs:
            a, b = os.path.join(REPOS[p], rp), os.path.join(TW, lp)
            same = os.path.exists(a) and os.path.exists(b) and open(a, "rb").read() == open(b, "rb").read()
            newer = os.path.exists(b) and os.path.getmtime(b) < os.path.getmtime(os.path.join(TW, lp.replace(".pdf", ".tex")))
            # the LaTeX log of the build: no error lines and no undefined reference (the published supplement had both
            # an error and a wrong table number, unseen because nonstop builds still write a PDF)
            lg = os.path.join(TW, lp.replace(".pdf", ".log"))
            lt = open(lg, encoding="utf-8", errors="replace").read() if os.path.exists(lg) else ""
            bad = [l.strip() for l in lt.split("\n") if l.startswith("!") or "undefined" in l or "Rerun to get" in l]
            note = "; ".join(x for x in ["tex newer than pdf" if newer else "", "LaTeX log: " + bad[0][:80] if bad else "",
                                         "" if lt else "no LaTeX log"] if x)
            rows.append((p, rp, "SAME" if same else "DIFFERENT", note))
    fails["R1"] = sum(1 for r in rows if r[2] != "SAME" or r[3])
    rep += ["## R1 paper PDFs", "", "| paper | repo file | vs local build | note |", "|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]

    # ---- R2, R3, R4, R6 on the docs
    r2, r3, r4, r6 = [], [], [], []
    for p, repo in REPOS.items():
        base = PAPERS[p]
        tex = open(base + ".tex", encoding="utf-8").read()
        title = re.sub(r"\s+", " ", re.search(r"\\title\{(.*?)\}\s*\n", tex, re.S).group(1)).strip()
        labs, caps = aux_labels(base), captions(base)
        cur_words = re.findall(r"[a-z]+", tex.lower())
        cur6 = {" ".join(cur_words[i:i + 6]) for i in range(len(cur_words) - 5)}
        old6 = set()
        for bk in glob.glob(base + "_before_*2026-10-04.tex"):
            w = re.findall(r"[a-z]+", open(bk, encoding="utf-8").read().lower())
            old6 |= {" ".join(w[i:i + 6]) for i in range(len(w) - 5)}
        stale6 = old6 - cur6
        docs = [f for f in repo_files[p] if DOC_RX.search(f)]
        for f in docs:
            txt = open(os.path.join(repo, f), encoding="utf-8").read()
            if f == "README.md":
                m = re.search(r"\*([^*]{20,300})\*", txt)
                got = re.sub(r"\s+", " ", m.group(1).replace(">", " ")).strip() if m else ""
                r2.append((p, f, "SAME" if got == title else "DIFFERENT", got[:80], title[:80]))
            body = txt if not f.endswith("FILES.md") else "\n".join(l for l in txt.split("\n") if not l.startswith("|"))
            for m in re.finditer(r"\b[Ee]quation\s+\((\d+)\)", body):
                lab = labs.get(("eq", m.group(1)))
                ctx = re.sub(r"\s+", " ", body[max(0, m.start() - 60):m.end() + 60]).replace("|", "/")
                r3.append((p, f, f"equation ({m.group(1)})", lab or "MISSING", "", ctx))
            for m in REF_RX.finditer(body):
                if re.match(r"\s+of\s+(?!the paper)[A-Z]", body[m.end():m.end() + 40]):
                    continue  # a result of another work ("Theorem 4 of Dytso, Barletta and Kramer")
                kind = KIND[m.group(1)]
                nums = [m.group(2)] + re.findall(r"[IVX]+|\d+", m.group(3) or "")
                for n in nums:
                    key = (kind, n)
                    lab = labs.get(key)
                    ctx = re.sub(r"\s+", " ", body[max(0, m.start() - 60):m.end() + 60]).replace("|", "/")
                    r3.append((p, f, f"{m.group(1)} {n}", lab or "MISSING", caps.get(lab, "") if lab else "", ctx))
            w = re.findall(r"[a-z]+", body.lower())
            hits = sorted({" ".join(w[i:i + 6]) for i in range(len(w) - 5)} & stale6)
            for h in hits:
                k = body.lower().find(h.split()[0] + " " + h.split()[1])
                r4.append((p, f, h))
            # numbers in the doc (outside code blocks too: expected outputs must be true as well)
            for line in body.split("\n"):
                if f == "README.md" and line.startswith(">"):
                    continue
                for kind_, printed, val, dec, (a, b) in gn.scan_numbers(line):
                    before, after = line[max(0, a - 40):a], line[b:b + 40]
                    cls = gn.classify(kind_, printed, val, dec, before, after)
                    if cls or re.search(r"(Amendment|amendment|amendments|registered in|ORCID|version|v)\s*$", before) \
                            or re.match(r"[-\w]*\.(py|json|log|txt|md|pdf)", after) or re.search(r"[\w/]$", before) \
                            or re.match(r"[A-Za-z_/]", after) or kind_ == "word" and val <= 12 \
                            or f.startswith("registration-log/") and printed.isdigit() and int(printed) in pub_amend:
                        continue
                    r6.append((p, f, printed, val, dec, line.strip()[:140]))
    fails["R2"] = sum(1 for r in r2 if r[2] != "SAME")
    fails["R3"] = sum(1 for r in r3 if r[3] == "MISSING")
    fails["R4"] = len(r4)
    rep += ["", "## R2 titles", "", "| paper | doc | verdict | README | paper |", "|---|---|---|---|---|"] + [f"| {' | '.join(map(str, r))} |" for r in r2]
    rep += ["", f"## R3 cross-references ({len(r3)}, {fails['R3']} missing)", "", "| paper | doc | reference | label | caption or title | context |",
            "|---|---|---|---|---|---|"] + [f"| {' | '.join(map(str, r))} |" for r in r3]
    rep += ["", f"## R4 stale wording ({len(r4)} phrases in the docs that the paper no longer has)", "", "| paper | doc | phrase |", "|---|---|---|"] + \
           [f"| {a} | {b} | {c} |" for a, b, c in r4]

    # ---- R7 file integrity, and the set of carried sources. The repositories are rebuilt from the originals with the
    # publication build scripts (temporal_within/repo_build/), which also write a manifest repo path -> original; a file that
    # was added by hand must equal a local original of the same name (bytes, or the program for Python).
    RB = os.path.join(HERE, "_repo_rebuild")
    BUILD = os.path.join(TW, "repo_build")
    if "--reuse" not in sys.argv or not os.path.isdir(RB):
        import shutil
        shutil.rmtree(RB, ignore_errors=True)
        for sub, scripts in (("paper1", ["build_repo.py", "build_cert.py"]), ("paper2", ["build_repo2.py"])):
            os.makedirs(os.path.join(RB, sub))
            for s in scripts:
                subprocess.run([sys.executable, os.path.join(BUILD, s), os.path.join(RB, sub)], cwd=BUILD, capture_output=True, check=True)
        subprocess.run([sys.executable, os.path.join(BUILD, "build_checks.py"), RB], cwd=BUILD, capture_output=True, check=True)
    r7 = []
    carried_paths = {}
    for p, repo in REPOS.items():
        man = json.load(open(os.path.join(RB, p, "_manifest.json"), encoding="utf-8"))
        for f in repo_files[p]:
            if f.endswith(SKIP_SUFFIX) or f.startswith("paper/") or f.startswith("registration-log/"):
                continue
            a = os.path.join(repo, f)
            b = os.path.join(RB, p, f)
            if os.path.exists(b):
                if norm_bytes(a) == norm_bytes(b):
                    if f in man:
                        carried_paths[NP(man[f])] = f
                    r7.append((p, f, "SAME-AS-REBUILD", os.path.relpath(man.get(f, b), ROOT)))
                else:
                    r7.append((p, f, "DIFFERENT", "differs from the rebuild of " + os.path.relpath(man.get(f, b), ROOT)))
                continue
            cands = idx.get(os.path.basename(f), [])
            if f.endswith(".py"):
                ra = py_ast(a)
                ok = [c for c in cands if py_ast(c) == ra]
            else:
                sa = sha(a)
                ok = [c for c in cands if os.path.getsize(c) < 300e6 and sha(c) == sa]
            if ok:
                for c in ok:
                    carried_paths[NP(c)] = f
                r7.append((p, f, "SAME-AS-ORIGINAL", os.path.relpath(ok[0], ROOT)))
            elif os.path.basename(f).startswith("rerun_"):
                r7.append((p, f, "GENERATED", "rerun in the repository"))
            elif cands:
                r7.append((p, f, "DIFFERENT", "; ".join(os.path.relpath(c, ROOT) for c in cands[:3])))
            else:
                r7.append((p, f, "NO-ORIGINAL", ""))
    fails["R7"] = sum(1 for r in r7 if r[2] in ("DIFFERENT", "NO-ORIGINAL"))
    json.dump(carried_paths, open(os.path.join(HERE, "report_repo_carried.json"), "w", encoding="utf-8"), indent=0)
    from collections import Counter as _C
    cnt = _C(r[2] for r in r7)
    rep += ["", f"## R7 file integrity ({dict(cnt)})", "", "| paper | repo file | verdict | original |", "|---|---|---|---|"] + \
           [f"| {' | '.join(r)} |" for r in r7 if r[2] not in ("SAME-AS-REBUILD", "SAME-AS-ORIGINAL")]

    carried_sids = set()
    pub_text = open(PUBLIC_LOG, encoding="utf-8").read()
    for j, s in enumerate(gn.SOURCES):
        if s[0] == "LIT" or "data_barletta_dytso" in s[2]:
            carried_sids.add(j)  # cited literature (carried by the citation) and public third-party data (named with its URL)
        elif s[0] == "PRIMARY-LOG":
            m = re.match(r"Amendment (\d+)", s[1])
            if m and int(m.group(1)) in pub_amend:
                carried_sids.add(j)
        elif NP(s[2]) in carried_paths:
            carried_sids.add(j)

    def carried_value(val, dec):
        rng, _ = gn.lookup(abs(val), dec)
        return any(int(gn.SRC[i]) in carried_sids and gn.DECS[i] >= dec for i in rng)

    # ---- R6 verdicts
    paper_text = {p: gn.clean_tex(open(PAPERS[p] + ".tex", encoding="utf-8").read()) for p in PAPERS}
    r6v = []
    for p, f, printed, val, dec, line in r6:
        if carried_value(val, dec):
            v = "CARRIED"
        elif re.search(r"(?<![\d.])" + re.escape(printed) + r"(?![\d])", paper_text[p]):
            v = "IN-PAPER"
        else:
            rng, _ = gn.lookup(abs(val), dec)
            v = "LOCAL-ONLY" if any(gn.SOURCES[gn.SRC[i]][0].startswith("PRIMARY") and gn.DECS[i] >= dec for i in rng) else "NOT-FOUND"
        r6v.append((p, f, printed, v, line))
    fails["R6"] = sum(1 for r in r6v if r[3] in ("NOT-FOUND", "LOCAL-ONLY"))
    rep += ["", f"## R6 numbers in the docs ({len(r6v)}; {fails['R6']} not carried by the public material nor printed in the paper)", "",
            "| paper | doc | printed | verdict | line |", "|---|---|---|---|---|"] + \
           [f"| {a} | {b} | {c} | {d} | {e.replace('|', '/')} |" for a, b, c, d, e in r6v if d in ("NOT-FOUND", "LOCAL-ONLY")]

    # ---- R8 traceability of the papers' numbers
    bindings = gn.load_bindings()
    r8, r8json = [], []
    for p in PAPERS:
        tex = open(PAPERS[p] + ".tex", encoding="utf-8").read()
        for para in gn.paragraphs(gn.body_lines(tex)):
            amend, files = gn.comment_refs(para["comment"])
            declared = gn.declared_sids(amend, files)
            for no, raw in para["lines"]:
                text = gn.clean_tex(raw)
                for kind_, printed, val, dec, (a, b) in gn.scan_numbers(text):
                    if gn.classify(kind_, printed, val, dec, text[max(0, a - 40):a], text[b:b + 40]):
                        continue
                    bd = next((x for x in bindings if x.get("kind") != "unverified" and x["paper"] == p and x["printed"] == printed.strip()
                               and x["context"] in raw), None)
                    if bd:
                        ams = [int(n) for n in re.findall(r"\bL\((\d+)", bd["expr"])]
                        fps = re.findall(r"\b[JT]\('([^']+)'", bd["expr"])
                        def fp_ok(fp, expr=bd["expr"]):
                            # cited literature: papers/ texts, and the Kostal-Shinomoto 2016 text kept with an archived lane
                            if NP(os.path.join(ROOT, fp)) in carried_paths or "/papers/" in fp or "data_barletta_dytso" in fp \
                                    or fp.endswith("occupancy_resolution/ks2016.txt"):
                                return True
                            if fp.endswith("PREREG_Q32.md"):  # the regexes must match the public copy of the log as well
                                return all(re.search(rx, pub_text, re.S) for rx in re.findall(r"T\('[^']*PREREG_Q32\.md', r'([^']*)'\)", expr))
                            return False
                        ok = any(n in pub_amend for n in ams) or any(fp_ok(fp) for fp in fps)
                        if not ams and not fps:
                            ok = True  # a literal definition (a label printed by the paper itself)
                        src = ", ".join([f"A{n}" for n in ams] + fps)[:150]
                        labels = [f"Amendment {n}" for n in ams] + fps
                    else:
                        v, hits = gn.verdict(val, dec, declared, set(), gn.sig_digits(printed))
                        sids = {int(gn.SRC[i]) for i in hits}
                        ok = bool(sids & carried_sids)
                        # a source the public material carries in an equivalent copy (e.g. the certificate log published
                        # under another name): accepted when the value has at least four significant digits
                        if not ok and gn.sig_digits(printed) >= 4 and carried_value(val, dec):
                            ok = True
                        src = gn.where(hits)[:150]
                        labels = sorted({gn.SOURCES[s][1] for s in sids})
                    if not ok:
                        r8.append((p, no, printed.strip(), src.replace("|", "/"), re.sub(r"\s+", " ", text[max(0, a - 50):b + 30]).replace("|", "/")))
                        r8json.append(dict(paper=p, line=no, printed=printed.strip(), sources=labels, binding=bool(bd)))
    fails["R8"] = len(r8)
    json.dump(r8json, open(os.path.join(HERE, "report_repo_r8.json"), "w", encoding="utf-8"), indent=0)
    rep += ["", f"## R8 numbers of the papers whose sources the public material does not carry ({len(r8)})", "",
            "| paper | line | printed | sources | context |", "|---|---|---|---|---|"] + [f"| {' | '.join(map(str, r))} |" for r in r8]

    # ---- R9 amendments cited
    r9 = []
    for p in PAPERS:
        tex = open(PAPERS[p] + ".tex", encoding="utf-8").read()
        for no, raw in gn.body_lines(tex):
            text = gn.clean_tex(gn.split_comment(raw)[0])
            for kind_, printed, val, dec, (a, b) in gn.scan_numbers(text):
                if gn.classify(kind_, printed, val, dec, text[max(0, a - 40):a], text[b:b + 40]) == "IDENT" and printed.isdigit() \
                        and int(printed) in gn.ALL_AMEND and int(printed) not in pub_amend:
                    r9.append((p, f"tex line {no}", printed))
    for p, repo in REPOS.items():
        for f in repo_files[p]:
            if DOC_RX.search(f):
                for m in re.finditer(r"\b[Aa]mendments?\s+(\d{2,4})", open(os.path.join(repo, f), encoding="utf-8").read()):
                    if int(m.group(1)) not in pub_amend:
                        r9.append((p, f, m.group(1)))
    fails["R9"] = len(r9)
    rep += ["", f"## R9 cited amendments missing from the public log (public copy ends at {max(pub_amend)})", "", "| paper | where | amendment |", "|---|---|---|"] + \
           [f"| {a} | {b} | {c} |" for a, b, c in r9]

    # ---- R5 prose
    r5 = []
    for p, repo in REPOS.items():
        for f in repo_files[p]:
            if DOC_RX.search(f):
                args = ["py", os.path.join(ROOT, "cek_en2.py"), os.path.join(repo, f), "--mode", "paper"] + (["--no-tables"] if f.endswith("FILES.md") else [])
                out = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
                if "FAIL" in out.split("\n")[-2:] or out.rstrip().endswith("FAIL"):
                    r5.append((p, f, " / ".join(l.strip() for l in out.split("\n") if l.strip().startswith("[H"))[:300]))
    fails["R5"] = len(r5)
    rep += ["", f"## R5 prose ({len(r5)} docs fail cek_en2)", "", "| paper | doc | findings |", "|---|---|---|"] + [f"| {a} | {b} | {c.replace('|', '/')} |" for a, b, c in r5]

    rep[3:3] = ["| check | failures |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(fails.items())] + [""]
    open(os.path.join(HERE, "report_repo.md"), "w", encoding="utf-8", newline="\n").write("\n".join(rep) + "\n")
    for k, v in sorted(fails.items()):
        print(k, "FAIL" if v else "PASS", v)
    return 1 if any(fails.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
