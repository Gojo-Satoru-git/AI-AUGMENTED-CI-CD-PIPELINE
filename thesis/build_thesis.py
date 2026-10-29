"""
Builds the thesis (.docx and .pdf) through Microsoft Word, matching the supplied format:
A4; Times New Roman 14 pt body (1.5 spacing, justified), black text; 16 pt bold chapter headings;
12 pt bold captions; tables 11 pt with grey (D9D9D9) header; algorithms in Courier New 9 pt on a light
grey box; front matter numbered in lower-case Roman, body from Chapter 1 in Arabic.

Run with the system Python (needs pywin32 + PyMuPDF):  python thesis/build_thesis.py
Two passes: the first fixes the layout, the second fills the contents / list pages from the PDF.
"""
import re
import sys
from pathlib import Path

import pymupdf
import win32com.client as wc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch1_2, ch3, ch4, ch5, ch6_refs                      # noqa: E402

FIG = HERE / "figures"
OUT_DOCX = HERE / "Thesis_Final_Draft.docx"
OUT_PDF = HERE / "Thesis_Final_Draft.pdf"
TNR = "Times New Roman"
CM = 28.3465

TITLE = "AUTONOMOUS AI-AUGMENTED CI/CD PIPELINE WITH ADAPTIVE TRUST-TIERING AND CHAOS ENGINEERING"
TITLE_MIXED = "Autonomous AI-Augmented CI/CD Pipeline with Adaptive Trust-Tiering and Chaos Engineering"
STUDENTS = [("Praveen Kumar C", "[Roll No.]"), ("Jaiprakash S", "[Roll No.]"), ("Dharshini T", "[Roll No.]")]
SUPERVISOR = "Dr. P. Anandha Kumar"
HOD = "[Name of the HOD]"
PANEL = "[Panel Member 1] and [Panel Member 2]"
DEAN = "Dr. P. Jayashree"
DATE_LINE = "[MONTH YEAR]"

ABSTRACT = [
    "Failed CI/CD runs are still triaged by hand, although the correct response depends on the cause: a "
    "transient fault should be retried, a data fault escalated, a failing host quarantined and a real defect "
    "blocked. This project builds an AI-augmented release gate that classifies the distributed trace of a "
    "failed run into five operational classes and decides how much of the resulting action the agent may "
    "take alone.",
    "Real HDFS traces from TraceBench are clock-aligned and split by scenario. An attention-pooled "
    "bidirectional LSTM reaches a macro-F1 of 0.7464 on 57 unseen scenarios and 0.6829 when REAL_BUG, whose "
    "perfect score is a workload confound, is excluded; an order-free baseline ties it. A trust engine grants "
    "four autonomy tiers from the Wilson lower bound of the agent's accuracy and demotes it at once when a "
    "real defect is merged. Supporting modules localise suspect spans, abstain outside the training domain, "
    "compare canary releases, test CVE reachability and inject faults.",
    "On a live four-service system the agent earned full autonomy, lost it on a defect with no trace "
    "signature, and regained trust only through new evidence. The limits are stated: about 40% of healthy "
    "traces are flagged, compound faults are mostly missed, and the workflows and monitoring stack were not "
    "executed.",
]
ACK = [
    f"We express our thankfulness to our project supervisor **{SUPERVISOR}**, Department of Information "
    "Technology, MIT Campus, for providing invaluable support and assistance with encouragement which aided "
    "us to complete this project.",
    f"We are thankful to the panel members **{PANEL}**, Department of Information Technology, MIT Campus, "
    "for their invaluable feedback in reviews.",
    f"We express our gratitude and sincere thanks to our respected Dean of MIT Campus, **{DEAN}**, for "
    "providing excellent computing facilities throughout the project.",
]

BLOCKS = ch1_2.BLOCKS + ch3.BLOCKS + ch4.BLOCKS + ch5.BLOCKS + ch6_refs.BLOCKS
REFS = ch6_refs.REFS
ABBR = [a for a in ch6_refs.ABBREVIATIONS if a[0] not in ("ROC", "DAG", "UI", "GBM")]


# ----------------------------------------------------------------------------- numbering
def roman(n):
    out, vals = "", [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    for v, s in vals:
        while n >= v:
            out, n = out + s, n - v
    return out


def prescan():
    chap = h2 = h3 = t = f = a = 0
    labels, toc, lot, lof = {}, [], [], []
    for b in BLOCKS:
        k = b[0]
        if k == "chapter":
            chap, h2, h3, t, f, a = chap + 1, 0, 0, 0, 0, 0
            toc.append((0, str(chap), b[1]))
        elif k == "h2":
            h2, h3 = h2 + 1, 0
            toc.append((1, f"{chap}.{h2}", b[1]))
        elif k == "h3":
            h3 += 1
            toc.append((2, f"{chap}.{h2}.{h3}", b[1]))
        elif k == "table":
            t += 1
            labels["T:" + b[1]] = f"Table {chap}.{t}"
            lot.append((f"{chap}.{t}", b[2]))
        elif k == "fig":
            f += 1
            labels["F:" + b[1]] = f"Figure {chap}.{f}"
            lof.append((f"{chap}.{f}", b[2]))
        elif k == "alg":
            a += 1
            labels["A:" + b[1]] = f"Algorithm {chap}.{a}"
    order = []
    for b in BLOCKS:
        txt = []
        if b[0] in ("p",):
            txt = [b[1]]
        elif b[0] == "bul":
            txt = b[1]
        for s in txt:
            for key in re.findall(r"\{R:(\w+)\}", s):
                if key not in order:
                    order.append(key)
    cite = {k: i + 1 for i, k in enumerate(order)}
    for k in REFS:
        if k not in cite:
            cite[k] = len(cite) + 1
    return labels, toc, lot, lof, cite


LABELS, TOC, LOT, LOF, CITE = prescan()


def resolve(s):
    s = re.sub(r"\{([TFA]):(\w+)\}", lambda m: LABELS[m.group(1) + ":" + m.group(2)], s)
    return re.sub(r"\{R:(\w+)\}", lambda m: f"[{CITE[m.group(1)]}]", s)


def parse_inline(s):
    """'a **b** *c*' -> ('a b c', [(start, end, 'b'|'i'), ...])"""
    plain, spans, pos = "", [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|\*(.+?)\*", s):
        plain += s[pos:m.start()]
        inner, kind = (m.group(1), "b") if m.group(1) is not None else (m.group(2), "i")
        spans.append((len(plain), len(plain) + len(inner), kind))
        plain += inner
        pos = m.end()
    plain += s[pos:]
    return plain, spans


# ----------------------------------------------------------------------------- Word helpers
class Builder:
    def __init__(self, word, pages):
        self.word = word
        self.pages = pages
        self.doc = word.Documents.Add()
        self.doc.TrackRevisions = False
        ps = self.doc.PageSetup
        ps.PaperSize = 7
        ps.LeftMargin, ps.RightMargin, ps.TopMargin, ps.BottomMargin = 85, 56.5, 71, 62
        ps.FooterDistance, ps.HeaderDistance = 40, 30
        n = self.doc.Styles("Normal")
        n.Font.Name, n.Font.Size, n.Font.Color = TNR, 14, 0
        n.ParagraphFormat.Alignment = 3
        self.cur_chapter = 0

    def end(self):
        return self.doc.Range(self.doc.Content.End - 1, self.doc.Content.End - 1)

    def para(self, text, size=14, bold=False, italic=False, align=3, left=0, first=0, before=0, after=6,
             spacing=1.5, keep=False, pbb=False, outline=10, font=TNR, right=0, tabs=()):
        plain, spans = parse_inline(text)
        r = self.end()
        start = r.Start
        r.InsertAfter(plain + "\r")
        pr = self.doc.Range(start, start + len(plain) + 1)
        f = pr.Font
        f.Name, f.Size, f.Bold, f.Italic, f.Color = font, size, bold, italic, 0
        pf = pr.ParagraphFormat
        pf.Alignment, pf.LeftIndent, pf.FirstLineIndent, pf.RightIndent = align, left, first, right
        pf.SpaceBefore, pf.SpaceAfter = before, after
        if spacing == 1:
            pf.LineSpacingRule = 0
        else:
            pf.LineSpacingRule = 5
            pf.LineSpacing = 12 * spacing
        pf.KeepWithNext, pf.PageBreakBefore, pf.OutlineLevel = keep, pbb, outline
        pf.TabStops.ClearAll()
        for pos, kind in tabs:                       # kind: 0 left, 1 center, 2 right
            pf.TabStops.Add(pos, kind)
        for a, b, kind in spans:
            sub = self.doc.Range(start + a, start + b)
            if kind == "b":
                sub.Font.Bold = True
            else:
                sub.Font.Italic = True
        return pr

    def section_break(self):
        self.end().InsertBreak(2)

    # ---------------------------------------------------------------- body blocks
    def chapter(self, title):
        self.cur_chapter += 1
        self.para(f"CHAPTER {self.cur_chapter}", size=16, bold=True, align=1, spacing=1.5, after=18, pbb=True,
                  keep=True, outline=1)
        self.para(title, size=16, bold=True, align=1, spacing=1.5, after=12, keep=True)
        self.h2n = self.h3n = 0
        self.tn = self.fn = self.an = 0

    def h2(self, title):
        self.h2n += 1
        self.h3n = 0
        self.para(f"{self.cur_chapter}.{self.h2n} {title}", bold=True, align=0, before=12, after=6, keep=True,
                  outline=2)

    def h3(self, title):
        self.h3n += 1
        self.para(f"{self.cur_chapter}.{self.h2n}.{self.h3n} {title}", bold=True, italic=True, align=0,
                  before=10, after=6, keep=True, outline=3)

    def p(self, text):
        self.para(resolve(text))

    def bullets(self, items):
        for it in items:
            self.para("•\t" + resolve(it), left=36, first=-24, after=6, tabs=[(36, 0)])

    def caption(self, text, keep):
        self.para(text, size=12, bold=True, align=1, spacing=1, before=6, after=6, keep=keep)

    def spacer(self, pt=6):
        self.para("", size=2, spacing=1, after=pt, align=0)

    def table(self, key, caption, header, rows, widths):
        self.tn += 1
        self.caption(f"Table {self.cur_chapter}.{self.tn}: {caption}", True)
        tbl = self.doc.Tables.Add(self.end(), len(rows) + 1, len(header))
        tbl.AllowAutoFit = False
        tbl.Borders.Enable = True
        for b in (-1, -2, -3, -4, -5, -6):
            tbl.Borders(b).LineStyle = 1
            tbl.Borders(b).LineWidth = 4
            tbl.Borders(b).Color = 0
        tbl.TopPadding = tbl.BottomPadding = 1.5
        tbl.LeftPadding = tbl.RightPadding = 4
        sc = min(1.0, 15.8 / sum(widths))
        for j, w in enumerate(widths, 1):
            tbl.Columns(j).Width = w * CM * sc
        for j, h in enumerate(header, 1):
            tbl.Cell(1, j).Range.Text = h
        for i, row in enumerate(rows, 2):
            for j, c in enumerate(row, 1):
                tbl.Cell(i, j).Range.Text = resolve(str(c))
        rng = tbl.Range
        rng.Font.Name, rng.Font.Size, rng.Font.Bold, rng.Font.Italic, rng.Font.Color = TNR, 11, False, False, 0
        rng.ParagraphFormat.Alignment = 0
        rng.ParagraphFormat.SpaceAfter = rng.ParagraphFormat.SpaceBefore = 0
        rng.ParagraphFormat.LineSpacingRule = 0
        rng.ParagraphFormat.LeftIndent = rng.ParagraphFormat.FirstLineIndent = 0
        rng.ParagraphFormat.KeepWithNext = False
        tbl.Rows.AllowBreakAcrossPages = False
        if len(rows) <= 12:
            for ri in range(1, len(rows) + 1):
                tbl.Rows(ri).Range.ParagraphFormat.KeepWithNext = True
        hdr = tbl.Rows(1)
        hdr.Range.Font.Bold = True
        hdr.Shading.BackgroundPatternColor = 217 + 217 * 256 + 217 * 65536
        hdr.HeadingFormat = True
        self.spacer(8)

    def figure(self, key, caption, file, w_cm):
        self.fn += 1
        pr = self.para("", align=1, spacing=1, before=6, after=4, keep=True)
        pic = self.doc.InlineShapes.AddPicture(str(FIG / file), False, True, self.doc.Range(pr.Start, pr.Start))
        pic.LockAspectRatio = True
        ratio = pic.Height / pic.Width
        pic.Width = w_cm * CM
        pic.Height = w_cm * CM * ratio
        self.caption(f"Figure {self.cur_chapter}.{self.fn}: {caption}", False)
        self.spacer(6)

    def algorithm(self, key, caption, code):
        self.an += 1
        self.caption(f"Algorithm {self.cur_chapter}.{self.an}: {caption}", True)
        tbl = self.doc.Tables.Add(self.end(), 1, 1)
        tbl.Borders.Enable = False
        tbl.AllowAutoFit = False
        tbl.Columns(1).Width = 15.8 * CM
        tbl.TopPadding = tbl.BottomPadding = 5
        tbl.LeftPadding = tbl.RightPadding = 6
        tbl.Cell(1, 1).Range.Text = code.replace("\n", "\v")
        tbl.Cell(1, 1).Shading.BackgroundPatternColor = 242 + 242 * 256 + 242 * 65536
        rng = tbl.Range
        rng.Font.Name, rng.Font.Size, rng.Font.Bold, rng.Font.Color = "Courier New", 9, False, 0
        rng.ParagraphFormat.Alignment = 0
        rng.ParagraphFormat.SpaceAfter = rng.ParagraphFormat.SpaceBefore = 0
        rng.ParagraphFormat.LineSpacingRule = 0
        tbl.Rows.AllowBreakAcrossPages = False
        self.spacer(8)

    def references(self):
        self.para("REFERENCES", size=16, bold=True, align=1, spacing=1.5, after=12, pbb=True, outline=1)
        order = sorted(CITE.items(), key=lambda kv: kv[1])
        for key, n in order:
            self.para(f"[{n}]\t" + REFS[key], size=13, left=28, first=-28, after=6, tabs=[(28, 0)])

    # ---------------------------------------------------------------- front matter
    def cover(self):
        t = TITLE.split(" WITH ")
        self.para(t[0] + " WITH", size=16, bold=True, align=1, before=13, after=0)
        self.para(t[1], size=16, bold=True, align=1, after=18)
        self.para("A PROJECT REPORT", bold=True, align=1, after=6)
        self.para("Submitted by", italic=True, align=1, after=6)
        for n, r in STUDENTS:
            self.para(f"{n} ({r})", align=1, after=4)
        self.para("in partial fulfilment for the award of the degree of", italic=True, align=1, before=8, after=6)
        self.para("BACHELOR OF TECHNOLOGY", bold=True, align=1, after=6)
        self.para("in", italic=True, align=1, after=6)
        self.para("ARTIFICIAL INTELLIGENCE AND DATA SCIENCE", bold=True, align=1, after=10)
        pr = self.para("", align=1, spacing=1, after=10)
        pic = self.doc.InlineShapes.AddPicture(str(FIG / "logo.png"), False, True, self.doc.Range(pr.Start, pr.Start))
        pic.LockAspectRatio = True
        pic.Width, pic.Height = 99, 87.7
        for line in ("DEPARTMENT OF INFORMATION TECHNOLOGY", "MADRAS INSTITUTE OF TECHNOLOGY", "ANNA UNIVERSITY",
                     "CHENNAI – 600 044", DATE_LINE):
            self.para(line, bold=True, align=1, after=6)
        self.section_break()

    def bonafide(self):
        self.para("ANNA UNIVERSITY : CHENNAI 60025", size=18, bold=True, align=1, spacing=1.5, after=8)
        self.para("BONAFIDE CERTIFICATE", bold=True, align=1, spacing=1.5, after=18)
        names = ", ".join(f"{n} {r}" for n, r in STUDENTS)
        self.para(f"Certified that this project report “{TITLE}” is the bonafide work of {names} who "
                  "carried out the project work under my supervision.", after=48)
        tbl = self.doc.Tables.Add(self.end(), 1, 2)
        tbl.Borders.Enable = False
        tbl.AllowAutoFit = False
        tbl.Columns(1).Width = 8.0 * CM
        tbl.Columns(2).Width = 8.0 * CM
        left = ["SIGNATURE", "", HOD, "HEAD OF THE DEPARTMENT", "Department of Information Technology",
                "Madras Institute of Technology,", "Chennai – 44"]
        right = ["SIGNATURE", "", SUPERVISOR, "SUPERVISOR", "Department of Information Technology",
                 "Madras Institute of Technology,", "Chennai - 44"]
        tbl.Cell(1, 1).Range.Text = "\v".join(left)
        tbl.Cell(1, 2).Range.Text = "\v".join(right)
        rng = tbl.Range
        rng.Font.Name, rng.Font.Size, rng.Font.Color = TNR, 14, 0
        rng.ParagraphFormat.Alignment = 0
        rng.ParagraphFormat.LineSpacingRule = 0
        rng.ParagraphFormat.SpaceAfter = 0
        for c in (1, 2):
            cell_rng = tbl.Cell(1, c).Range
            txt = cell_rng.Text
            for token in ("SIGNATURE", "HEAD OF THE DEPARTMENT", "SUPERVISOR"):
                i = txt.find(token)
                if i >= 0:
                    self.doc.Range(cell_rng.Start + i, cell_rng.Start + i + len(token)).Font.Bold = True
        self.spacer(0)

    def acknowledgement(self):
        self.para("ACKNOWLEDGEMENT", size=16, bold=True, align=1, spacing=1.5, after=12, pbb=True)
        for t in ACK:
            self.para(t, first=36, after=6)
        self.spacer(14)
        tbl = self.doc.Tables.Add(self.end(), len(STUDENTS), 2)
        tbl.Borders.Enable = False
        tbl.AllowAutoFit = False
        tbl.Columns(1).Width = 7.0 * CM
        tbl.Columns(2).Width = 4.5 * CM
        tbl.Rows.LeftIndent = 6.0 * CM
        for i, (n, r) in enumerate(STUDENTS, 1):
            tbl.Cell(i, 1).Range.Text = n.upper()
            tbl.Cell(i, 2).Range.Text = r
        rng = tbl.Range
        rng.Font.Name, rng.Font.Size, rng.Font.Bold, rng.Font.Color = TNR, 14, True, 0
        rng.ParagraphFormat.Alignment = 0
        rng.ParagraphFormat.SpaceAfter = 8
        self.spacer(0)

    def abstract(self):
        self.para("ABSTRACT", size=16, bold=True, align=1, spacing=1.5, after=12, pbb=True)
        for t in ABSTRACT:
            self.para(t, spacing=2.0, after=0)

    def front_page(self, label):
        return self.pages.get(label, "0")

    def toc(self):
        self.para("TABLE OF CONTENTS", size=16, bold=True, align=1, spacing=1.5, after=12, pbb=True)
        self.para("CH. NO\tTITLE\tPAGE NO", size=12, bold=True, align=0, spacing=1, after=8, left=80,
                  first=-80, right=0, tabs=[(80, 0), (430, 2)])
        for title in ("ABSTRACT", "LIST OF TABLES", "LIST OF FIGURES", "LIST OF ABBREVIATIONS"):
            self.para(f"\t{title}\t{self.front_page(title)}", size=12, bold=True, align=0, spacing=1, after=5,
                      left=80, first=-80, right=40, tabs=[(80, 0), (432, 2)])
        for level, num, title in TOC:
            pg = self.pages.get(("toc", num), "0")
            if level == 0:
                self.para(f"{num}\t{title}\t{pg}", size=12, bold=True, align=0, spacing=1, before=5, after=5,
                          left=80, first=-80, right=40, tabs=[(80, 0), (432, 2)])
            else:
                ind = 80 if level == 1 else 100
                self.para(f"{num}  {title}\t{pg}", size=12, align=0, spacing=1, after=4, left=ind, first=0,
                          right=40, tabs=[(432, 2)])
        self.para(f"\tREFERENCES\t{self.pages.get('REFERENCES', '0')}", size=12, bold=True, align=0, spacing=1,
                  before=5, after=4, left=80, first=-80, right=40, tabs=[(80, 0), (432, 2)])

    def list_page(self, heading, head_cols, entries, key):
        self.para(heading, size=14, bold=True, align=1, spacing=1.5, after=10, pbb=True)
        self.para(head_cols, size=12, bold=True, align=0, spacing=1, after=8, left=78, first=-78, right=0,
                  tabs=[(78, 0), (429, 2)])
        for num, title in entries:
            self.para(f"{num}\t{title}\t{self.pages.get((key, num), '0')}", size=12, align=0, spacing=1, after=5,
                      left=78, first=-53, right=40, tabs=[(78, 0), (431, 2)])

    def abbreviations(self):
        self.para("LIST OF ABBREVIATIONS", size=14, bold=True, align=1, spacing=1.5, after=10, pbb=True)
        tbl = self.doc.Tables.Add(self.end(), len(ABBR) + 1, 2)
        tbl.AllowAutoFit = False
        tbl.Borders.Enable = True
        for b in (-1, -2, -3, -4, -5, -6):
            tbl.Borders(b).LineStyle = 1
            tbl.Borders(b).LineWidth = 4
            tbl.Borders(b).Color = 0
        tbl.Columns(1).Width = 4.2 * CM
        tbl.Columns(2).Width = 11.6 * CM
        tbl.Cell(1, 1).Range.Text, tbl.Cell(1, 2).Range.Text = "Abbreviation", "Expansion"
        for i, (a, e) in enumerate(ABBR, 2):
            tbl.Cell(i, 1).Range.Text, tbl.Cell(i, 2).Range.Text = a, e
        rng = tbl.Range
        rng.Font.Name, rng.Font.Size, rng.Font.Bold, rng.Font.Color = TNR, 11, False, 0
        rng.ParagraphFormat.Alignment = 0
        rng.ParagraphFormat.SpaceAfter = rng.ParagraphFormat.SpaceBefore = 0
        rng.ParagraphFormat.LineSpacingRule = 0
        tbl.TopPadding = tbl.BottomPadding = 2
        tbl.Rows(1).Range.Font.Bold = True
        tbl.Rows(1).Shading.BackgroundPatternColor = 217 + 217 * 256 + 217 * 65536
        tbl.Rows.AllowBreakAcrossPages = False
        self.spacer(0)

    # ---------------------------------------------------------------- footers
    def footers(self):
        for i in (1, 2, 3):
            f = self.doc.Sections(i).Footers(1)
            f.LinkToPrevious = False
        for i in (1, 2, 3):
            f = self.doc.Sections(i).Footers(1)
            f.Range.Text = ""
            if i > 1:
                f.Range.ParagraphFormat.Alignment = 1
                f.Range.Font.Name, f.Range.Font.Size, f.Range.Font.Color = TNR, 12, 0
                self.doc.Fields.Add(f.Range, 33)
                f.Range.ParagraphFormat.Alignment = 1
                f.Range.Font.Name, f.Range.Font.Size, f.Range.Font.Color = TNR, 12, 0
                f.PageNumbers.NumberStyle = 2 if i == 2 else 0
                f.PageNumbers.RestartNumberingAtSection = True
                f.PageNumbers.StartingNumber = 1

    # ---------------------------------------------------------------- whole document
    def build(self):
        self.cover()
        self.bonafide()
        self.acknowledgement()
        self.abstract()
        self.toc()
        self.list_page("LIST OF TABLES", "TABLE NO\tTITLE\tPAGE NO", LOT, "T")
        self.list_page("LIST OF FIGURES", "FIGURE NO\tTITLE\tPAGE NO", LOF, "F")
        self.abbreviations()
        self.section_break()
        for b in BLOCKS:
            k = b[0]
            if k == "chapter":
                self.chapter(b[1])
            elif k == "h2":
                self.h2(b[1])
            elif k == "h3":
                self.h3(b[1])
            elif k == "p":
                self.p(b[1])
            elif k == "bul":
                self.bullets(b[1])
            elif k == "table":
                self.table(*b[1:])
            elif k == "fig":
                self.figure(*b[1:])
            elif k == "alg":
                self.algorithm(*b[1:])
        self.references()
        self.footers()


# ----------------------------------------------------------------------------- pass 2 helpers
def norm(s):
    return re.sub(r"\s+", " ", s)


def locate_pages(pdf_path):
    d = pymupdf.open(str(pdf_path))
    txt = [norm(p.get_text()) for p in d]
    n = len(d)
    ch1 = next(i for i, t in enumerate(txt) if re.search(r"(^| )CHAPTER 1 INTRODUCTION", t))
    lab = lambda i: str(i - ch1 + 1)
    pages = {}

    def find(needle, start=ch1):
        for i in range(start, n):
            if needle in txt[i]:
                return i
        return None

    for level, num, title in TOC:
        needle = f"CHAPTER {num} " if level == 0 else f"{num} {title}"
        i = find(needle)
        pages[("toc", num)] = lab(i) if i is not None else "?"
    for num, cap in LOT:
        i = find(f"Table {num}: ")
        pages[("T", num)] = lab(i) if i is not None else "?"
    for num, cap in LOF:
        i = find(f"Figure {num}: ")
        pages[("F", num)] = lab(i) if i is not None else "?"
    i = find("REFERENCES [1]")
    pages["REFERENCES"] = lab(i) if i is not None else "?"
    for title in ("ABSTRACT", "LIST OF TABLES", "LIST OF FIGURES", "LIST OF ABBREVIATIONS"):
        for i in range(1, ch1):
            if title in txt[i][:40]:
                pages[title] = roman(i)       # pdf page index i is roman numeral i (cover is unnumbered)
                break
    return pages, n, ch1


def run(pages):
    word = wc.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        b = Builder(word, pages)
        b.build()
        b.doc.SaveAs2(str(OUT_DOCX))
        b.doc.SaveAs2(str(OUT_PDF), FileFormat=17)
        b.doc.Close(False)
    finally:
        word.Quit()


if __name__ == "__main__":
    run({})
    pages, n, ch1 = locate_pages(OUT_PDF)
    print("pass 1:", n, "pages; chapter 1 starts on pdf page", ch1 + 1)
    run(pages)
    pages2, n2, ch1b = locate_pages(OUT_PDF)
    print("pass 2:", n2, "pages;", "stable" if pages2 == pages else "PAGES SHIFTED")
    missing = [k for k, v in pages2.items() if v == "?"]
    print("unlocated:", missing)
