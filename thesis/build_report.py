"""
Builds the Final Review Report (.docx + .pdf) with Microsoft Word.
Style: easy-to-read project report (A4, Times New Roman 12 pt, 1.15 spacing, numbered headings,
grey table headers, shaded code boxes and call-outs). Content: thesis/report_content.py.

    python thesis/build_report.py        (system Python: pywin32 + PyMuPDF)
"""
import re
import sys
from pathlib import Path

import win32com.client as wc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_content as RC                                    # noqa: E402

FIG = HERE / "figures"
OUT_DOCX = HERE / "Final_Review_Report.docx"
OUT_PDF = HERE / "Final_Review_Report.pdf"
TNR, CM = "Times New Roman", 28.3465
GREY = 217 + 217 * 256 + 217 * 65536
LIGHT = 242 + 242 * 256 + 242 * 65536
BLUE = 247 + 238 * 256 + 222 * 65536          # light call-out fill (BGR int)


def parse_inline(s):
    plain, spans, pos = "", [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|\*(.+?)\*", s):
        plain += s[pos:m.start()]
        inner, kind = (m.group(1), "b") if m.group(1) is not None else (m.group(2), "i")
        spans.append((len(plain), len(plain) + len(inner), kind))
        plain += inner
        pos = m.end()
    return plain + s[pos:], spans


class B:
    def __init__(self, word):
        self.doc = word.Documents.Add()
        ps = self.doc.PageSetup
        ps.PaperSize = 7
        ps.LeftMargin = ps.RightMargin = 71
        ps.TopMargin, ps.BottomMargin = 65, 60
        n = self.doc.Styles("Normal")
        n.Font.Name, n.Font.Size, n.Font.Color = TNR, 12, 0
        n.ParagraphFormat.Alignment = 3
        self.h1n = self.h2n = 0
        self.tn = self.fn = 0

    def end(self):
        return self.doc.Range(self.doc.Content.End - 1, self.doc.Content.End - 1)

    def para(self, text, size=12, bold=False, italic=False, align=3, left=0, first=0, before=0, after=6,
             spacing=1.15, keep=False, pbb=False, outline=10, font=TNR, color=0):
        plain, spans = parse_inline(text)
        r = self.end()
        start = r.Start
        r.InsertAfter(plain + "\r")
        pr = self.doc.Range(start, start + len(plain) + 1)
        f = pr.Font
        f.Name, f.Size, f.Bold, f.Italic, f.Color = font, size, bold, italic, color
        pf = pr.ParagraphFormat
        pf.Alignment, pf.LeftIndent, pf.FirstLineIndent = align, left, first
        pf.SpaceBefore, pf.SpaceAfter = before, after
        pf.LineSpacingRule = 5
        pf.LineSpacing = 12 * spacing
        pf.KeepWithNext, pf.PageBreakBefore, pf.OutlineLevel = keep, pbb, outline
        pf.TabStops.ClearAll()
        if left:
            pf.TabStops.Add(left, 0)
        for a, b, k in spans:
            sub = self.doc.Range(start + a, start + b)
            if k == "b":
                sub.Font.Bold = True
            else:
                sub.Font.Italic = True
        return pr

    def h1(self, t):
        self.h1n += 1
        self.h2n = 0
        self.para(f"{self.h1n}. {t}", size=16, bold=True, align=0, before=14, after=8, keep=True, outline=1,
                  pbb=(self.h1n > 1 and self.h1n in RC.PAGE_BREAK_BEFORE))

    def h2(self, t):
        self.h2n += 1
        self.para(f"{self.h1n}.{self.h2n} {t}", size=13, bold=True, align=0, before=10, after=5, keep=True,
                  outline=2)

    def p(self, t):
        self.para(RC.res(t))

    def bul(self, items):
        for it in items:
            self.para("•\t" + RC.res(it), left=22, first=-14, after=4)

    def spacer(self, pt=6):
        self.para("", size=2, spacing=1, after=pt, align=0)

    def caption(self, t, keep):
        self.para(t, size=11, bold=True, align=1, spacing=1, before=4, after=5, keep=keep)

    def table(self, cap, header, rows, widths):
        self.tn += 1
        self.caption(f"Table {self.tn}: {cap}", True)
        tbl = self.doc.Tables.Add(self.end(), len(rows) + 1, len(header))
        tbl.AllowAutoFit = False
        for b in (-1, -2, -3, -4, -5, -6):
            tbl.Borders(b).LineStyle = 1
            tbl.Borders(b).LineWidth = 4
            tbl.Borders(b).Color = 0
        tbl.TopPadding = tbl.BottomPadding = 1.5
        tbl.LeftPadding = tbl.RightPadding = 4
        sc = min(1.0, 16.0 / sum(widths))
        for j, w in enumerate(widths, 1):
            tbl.Columns(j).Width = w * CM * sc
        for j, h in enumerate(header, 1):
            tbl.Cell(1, j).Range.Text = h
        for i, row in enumerate(rows, 2):
            for j, c in enumerate(row, 1):
                tbl.Cell(i, j).Range.Text = RC.res(str(c))
        rng = tbl.Range
        rng.Font.Name, rng.Font.Size, rng.Font.Bold, rng.Font.Italic, rng.Font.Color = TNR, 10.5, False, False, 0
        rng.ParagraphFormat.Alignment = 0
        rng.ParagraphFormat.SpaceAfter = rng.ParagraphFormat.SpaceBefore = 0
        rng.ParagraphFormat.LineSpacingRule = 0
        rng.ParagraphFormat.LeftIndent = rng.ParagraphFormat.FirstLineIndent = 0
        tbl.Rows.AllowBreakAcrossPages = False
        if len(rows) <= 12:
            for ri in range(1, len(rows) + 1):
                tbl.Rows(ri).Range.ParagraphFormat.KeepWithNext = True
        tbl.Rows(1).Range.Font.Bold = True
        tbl.Rows(1).Shading.BackgroundPatternColor = GREY
        tbl.Rows(1).HeadingFormat = True
        self.spacer(8)

    def fig(self, cap, file, w_cm):
        self.fn += 1
        pr = self.para("", align=1, spacing=1, before=4, after=3, keep=True)
        pic = self.doc.InlineShapes.AddPicture(str(FIG / file), False, True, self.doc.Range(pr.Start, pr.Start))
        pic.LockAspectRatio = True
        ratio = pic.Height / pic.Width
        pic.Width, pic.Height = w_cm * CM, w_cm * CM * ratio
        self.caption(f"Figure {self.fn}: {cap}", False)
        self.spacer(4)

    def box(self, text, fill, font=TNR, size=12, lead=None):
        tbl = self.doc.Tables.Add(self.end(), 1, 1)
        tbl.Borders.Enable = False
        tbl.AllowAutoFit = False
        tbl.Columns(1).Width = 16.0 * CM
        tbl.TopPadding = tbl.BottomPadding = 5
        tbl.LeftPadding = tbl.RightPadding = 7
        txt = RC.res(text)
        plain, spans = parse_inline(txt) if font == TNR else (txt.replace("\n", "\v"), [])
        tbl.Cell(1, 1).Range.Text = plain
        tbl.Cell(1, 1).Shading.BackgroundPatternColor = fill
        rng = tbl.Range
        rng.Font.Name, rng.Font.Size, rng.Font.Bold, rng.Font.Color = font, size, False, 0
        rng.ParagraphFormat.Alignment = 0 if font != TNR else 3
        rng.ParagraphFormat.SpaceAfter = rng.ParagraphFormat.SpaceBefore = 0
        rng.ParagraphFormat.LineSpacingRule = 0 if font != TNR else 5
        if font == TNR:
            rng.ParagraphFormat.LineSpacing = 12 * 1.1
        base = tbl.Cell(1, 1).Range.Start
        for a, b, k in spans:
            sub = self.doc.Range(base + a, base + b)
            if k == "b":
                sub.Font.Bold = True
            else:
                sub.Font.Italic = True
        tbl.Rows.AllowBreakAcrossPages = False
        self.spacer(8)

    def code(self, cap, code):
        self.caption(cap, True)
        self.box(code.replace("\n", "\v"), LIGHT, font="Courier New", size=8.5)

    # ------------------------------------------------------------------ cover
    def cover(self):
        for t, sz, b, aft in RC.COVER:
            self.para(t, size=sz, bold=b, align=1, after=aft, spacing=1.2)
        self.para("", pbb=False)
        self.doc.Paragraphs(self.doc.Paragraphs.Count).Range.InsertBreak(7)   # page break

    def footers(self):
        f = self.doc.Sections(1).Footers(1)
        f.Range.Text = ""
        self.doc.Fields.Add(f.Range, 33)
        f.Range.ParagraphFormat.Alignment = 1
        f.Range.Font.Name, f.Range.Font.Size, f.Range.Font.Color = TNR, 10, 0

    def build(self):
        self.cover()
        for b in RC.BLOCKS:
            k = b[0]
            if k == "h1":
                self.h1(b[1])
            elif k == "h2":
                self.h2(b[1])
            elif k == "p":
                self.p(b[1])
            elif k == "bul":
                self.bul(b[1])
            elif k == "table":
                self.table(*b[1:])
            elif k == "fig":
                self.fig(*b[1:])
            elif k == "code":
                self.code(*b[1:])
            elif k == "note":
                self.box(b[1], BLUE)
        self.footers()


def run():
    word = wc.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        b = B(word)
        b.build()
        b.doc.SaveAs2(str(OUT_DOCX))
        b.doc.SaveAs2(str(OUT_PDF), FileFormat=17)
        b.doc.Close(False)
    finally:
        word.Quit()


if __name__ == "__main__":
    run()
    import pymupdf
    print("pages:", len(pymupdf.open(str(OUT_PDF))))
