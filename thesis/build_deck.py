"""
Builds the Final Review presentation (.pptx) with Microsoft PowerPoint (COM).
    python thesis/build_deck.py        (system Python: pywin32 + PyMuPDF)
Numbers come from reports/*.json and reports/cmp (measured), via thesis/cmp_data.py and live_state.json.
"""
import json
import sys
from pathlib import Path

import pymupdf
import win32com.client as wc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cmp_data                                               # noqa: E402

FIG = HERE / "figures"
OUT = HERE / "Final_Review_Presentation.pptx"
NAVY, TEAL, GREY, RED, GREEN, WHITE, INK = (92 * 65536 + 50 * 256 + 22, 128 * 65536 + 128 * 256 + 0, 120 * 65536 + 120 * 256 + 120,
                                            60 * 65536 + 60 * 256 + 190, 60 * 65536 + 150 * 256 + 40, 16777215, 40 * 65536 + 40 * 256 + 40)
PALE = 247 * 65536 + 244 * 256 + 240
ROWS = cmp_data.load()
OURS, OTH = ROWS[0], ROWS[1:]
LIVE = json.loads((HERE / "live_state.json").read_text(encoding="utf-8"))
f4 = lambda x: f"{x:.4f}"
W, H = 960, 540
FONT = "Calibri"


def img_size(name):
    pm = pymupdf.Pixmap(str(FIG / name))
    return pm.width, pm.height


class Deck:
    def __init__(self):
        self.app = wc.Dispatch("PowerPoint.Application")
        self.pres = self.app.Presentations.Add(WithWindow=False)
        self.pres.PageSetup.SlideWidth, self.pres.PageSetup.SlideHeight = W, H
        self.n = 0

    def slide(self, title, notes=""):
        self.n += 1
        s = self.pres.Slides.Add(self.n, 12)
        if title:
            bar = s.Shapes.AddShape(1, 0, 0, W, 64)
            bar.Fill.ForeColor.RGB, bar.Line.Visible = NAVY, 0
            self.text(s, title, 36, 10, W - 72, 44, size=26, bold=True, color=WHITE, anchor=3)
            acc = s.Shapes.AddShape(1, 0, 64, W, 4)
            acc.Fill.ForeColor.RGB, acc.Line.Visible = TEAL, 0
            self.text(s, str(self.n), W - 60, H - 30, 40, 20, size=11, color=GREY, align=3)
        s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = notes
        return s

    def text(self, s, txt, x, y, w, h, size=18, bold=False, color=INK, align=1, anchor=1, italic=False, font=FONT):
        tb = s.Shapes.AddTextbox(1, x, y, w, h)
        tf = tb.TextFrame
        tf.WordWrap = True
        tf.VerticalAnchor = anchor
        tf.MarginLeft = tf.MarginRight = 4
        r = tf.TextRange
        r.Text = txt
        r.Font.Name, r.Font.Size, r.Font.Bold, r.Font.Italic, r.Font.Color.RGB = font, size, bold, italic, color
        r.ParagraphFormat.Alignment = align
        return tb

    def bullets(self, s, items, x, y, w, h, size=18, gap=6, color=INK):
        tb = s.Shapes.AddTextbox(1, x, y, w, h)
        tf = tb.TextFrame
        tf.WordWrap = True
        r = tf.TextRange
        r.Text = "\r".join(items)
        r.Font.Name, r.Font.Size, r.Font.Color.RGB = FONT, size, color
        pf = r.ParagraphFormat
        pf.Bullet.Visible, pf.Bullet.Character, pf.Bullet.Font.Color.RGB = True, 8226, TEAL
        pf.SpaceAfter, pf.Alignment = gap, 1
        tb.TextFrame.Ruler.Levels(1).FirstMargin = 0
        tb.TextFrame.Ruler.Levels(1).LeftMargin = 18
        return tb

    def pic(self, s, name, x, y, w, h):
        iw, ih = img_size(name)
        sc = min(w / iw, h / ih)
        pw, ph = iw * sc, ih * sc
        return s.Shapes.AddPicture(str(FIG / name), False, True, x + (w - pw) / 2, y + (h - ph) / 2, pw, ph)

    def kpi(self, s, x, y, w, big, small, color=NAVY):
        box = s.Shapes.AddShape(5, x, y, w, 86)
        box.Fill.ForeColor.RGB, box.Line.ForeColor.RGB = PALE, 214 * 65536 + 205 * 256 + 195
        self.text(s, big, x, y + 6, w, 44, size=30, bold=True, color=color, align=2)
        self.text(s, small, x + 4, y + 50, w - 8, 34, size=12, color=GREY, align=2)

    def table(self, s, header, rows, x, y, w, colw, size=12, rowh=26, hi=None):
        shp = s.Shapes.AddTable(len(rows) + 1, len(header), x, y, w, rowh * (len(rows) + 1))
        t = shp.Table
        tot = sum(colw)
        for j, cw in enumerate(colw, 1):
            t.Columns(j).Width = w * cw / tot
        for i in range(len(rows) + 1):
            t.Rows(i + 1).Height = rowh
            for j in range(len(header)):
                c = t.Cell(i + 1, j + 1)
                c.Shape.TextFrame.TextRange.Text = header[j] if i == 0 else str(rows[i - 1][j])
                tr = c.Shape.TextFrame.TextRange
                tr.Font.Name, tr.Font.Size = FONT, size
                tr.Font.Bold = (i == 0) or (hi is not None and i - 1 == hi)
                c.Shape.TextFrame.MarginTop = c.Shape.TextFrame.MarginBottom = 2
                if i == 0:
                    c.Shape.Fill.ForeColor.RGB, tr.Font.Color.RGB = NAVY, WHITE
                else:
                    c.Shape.Fill.ForeColor.RGB = (230 * 65536 + 245 * 256 + 235) if (hi is not None and i - 1 == hi) else (WHITE if i % 2 else PALE)
                    tr.Font.Color.RGB = INK
        return shp

    def save(self):
        self.pres.SaveAs(str(OUT))
        self.pres.Close()
        self.app.Quit()


def concl():
    b = max(OTH, key=lambda r: r["macro_f1"])
    dd = OURS["macro_f1"] - b["macro_f1"]
    if dd > 0:
        return (f"Macro-F1 {f4(OURS['macro_f1'])} on unseen scenarios: the highest of the five models compared on the same data "
                f"(next best {f4(b['macro_f1'])}, {b['name'].split(':')[0]})")
    return (f"Macro-F1 {f4(OURS['macro_f1'])} on unseen scenarios; the best re-implemented competitor "
            f"({b['name'].split(':')[0]}) reaches {f4(b['macro_f1'])}")


CONCL = None


def build():
    global CONCL
    CONCL = concl()
    d = Deck()
    best = max(OTH, key=lambda r: r["macro_f1"])
    diff = OURS["macro_f1"] - best["macro_f1"]
    tc = sum(s[2] for s in LIVE["series"][:390])

    # 1 title ------------------------------------------------------------------------------------
    s = d.slide("", "Welcome. This is the final review of our project: an AI agent that triages failed CI/CD runs and decides how much it may do on its own.")
    bg = s.Shapes.AddShape(1, 0, 0, W, H)
    bg.Fill.ForeColor.RGB, bg.Line.Visible = NAVY, 0
    acc = s.Shapes.AddShape(1, 60, 300, 120, 5)
    acc.Fill.ForeColor.RGB, acc.Line.Visible = TEAL, 0
    d.text(s, "Autonomous AI-Augmented CI/CD Pipeline with Adaptive Trust-Tiering and Chaos Engineering", 60, 110, 840, 170, size=38, bold=True, color=WHITE, anchor=4)
    d.text(s, "Final Review  ·  Team 18", 60, 320, 840, 36, size=24, color=WHITE)
    d.text(s, "Praveen Kumar C  ·  Jaiprakash S  ·  Dharshini T", 60, 362, 840, 30, size=18, color=WHITE)
    d.text(s, "Guide: Dr. P. Anandha Kumar  ·  Department of Information Technology, MIT, Anna University", 60, 396, 840, 30, size=16, color=WHITE)

    # 2 journey -----------------------------------------------------------------------------------
    s = d.slide("Where we started, and what the final phase added",
                "Review 1 reported 88 percent, which turned out to be a one-column rule. Review 2 rebuilt the foundation. The final phase made it a runnable, measured, honest system.")
    d.table(s, ["Review", "What we showed", "Honest limitation then"],
            [["Review 1", "Phase 1 LSTM, reported 88.18%", "It equalled a one-column rule: counts fed in as token ids"],
             ["Review 2", "Clock-aligned data, attention BiLSTM, trust tiers, root-cause ranking", "Offline only; one data set; no live system"],
             ["Final", "Release gate (drift, canary, security, chaos, actuator), live system, dashboard, comparison with published models", "Workflows and Grafana stack written, not executed"]],
            40, 100, 880, [1.1, 4.2, 3.6], size=16, rowh=70)
    d.text(s, "Rule we followed: report weak numbers as plainly as strong ones.", 40, 440, 880, 40, size=20, bold=True, color=TEAL)

    # 3 problem -----------------------------------------------------------------------------------
    s = d.slide("The problem: four failures that look the same", "On a dashboard they are all a red run, but the right response differs completely. Retrying a corruption wastes time. Merging a bug ships it.")
    d.table(s, ["The run failed because of...", "Class", "Right response", "Needs trust tier"],
            [["A transient network or process fault", "INFRA_FLAKE", "Retry (budget 2)", "T2"],
             ["Corruption or data loss", "DATA_FAULT", "Escalate, never retry", "T1"],
             ["A failing host", "ENV_FAULT", "Quarantine the node", "T3"],
             ["A genuine defect", "REAL_BUG", "Block merge, alert developer", "T1"],
             ["Nothing wrong", "PASS", "Allow merge", "T2"]],
            40, 110, 880, [4, 2, 3.4, 1.8], size=17, rowh=48)
    d.text(s, "Second question: how much may the agent do alone? We answer it from its measured track record.", 40, 440, 880, 50, size=20, bold=True, color=TEAL)

    # 4 architecture ------------------------------------------------------------------------------
    s = d.slide("System architecture: five layers", "Traces come in, are normalised and checked, classified, combined with security and canary signals, and the decision is executed in dry-run mode. The true outcome flows back to the trust engine.")
    d.pic(s, "fig_arch.png", 40, 84, 560, 430)
    d.bullets(s, ["Deciding and executing are separate", "Prefers to abstain: unknown trace, thin canary, low confidence all go to a human", "One function serves the API, CI gate and dashboard"], 620, 130, 320, 330, size=18, gap=12)

    # 5 data --------------------------------------------------------------------------------------
    s = d.slide("Data foundation: real traces, clocks fixed, no leakage", "TraceBench HDFS. The hosts disagree on time, so we recover each host's offset from the parent-child edge records. We split by scenario so the test set measures generalisation.")
    d.pic(s, "fig_align.png", 40, 88, 520, 220)
    d.kpi(s, 590, 92, 160, "81,777", "traces, all align")
    d.kpi(s, 770, 92, 160, "363", "scenario folders, 3.0 GB")
    d.kpi(s, 590, 192, 160, "57", "held-out test scenarios")
    d.kpi(s, 770, 192, 160, "10,513", "test traces")
    d.bullets(s, ["Offset of a host = parent's clock reading minus child's first reading, propagated down the call tree", "Class from the scenario family; compound faults (20,010 traces) held out as a probe", "Task title is not a model input: it leaks the scenario"], 40, 330, 880, 190, size=18, gap=8)

    # 6 model -------------------------------------------------------------------------------------
    s = d.slide("The model: attention-pooled BiLSTM", "Instead of keeping only the last state, attention lets every span contribute with a learned weight. The weights also tell us which spans drove the decision.")
    d.pic(s, "fig_model.png", 30, 84, 900, 300)
    d.bullets(s, ["Inputs per span: operation|status, host role, agent type, depth, duration; plus 9 trace-level features", "125,430 parameters, trained on CPU in 24 minutes", "Evidence spans are a hint, not proof: two attempts to regularise attention were rejected"], 40, 390, 880, 140, size=17, gap=6)

    # 7 comparison chart --------------------------------------------------------------------------
    s = d.slide("Comparison with published architectures (same data, same split)",
                f"We re-implemented four published architectures and trained them on our split with the same recipe. Our model reaches {f4(OURS['macro_f1'])}; the best competitor is {best['name']} at {f4(best['macro_f1'])}. These are re-implementations, one run each, and the papers report on other data, so this is a like-for-like comparison on our data, not a comparison with the papers' own numbers.")
    d.pic(s, "fig_cmp.png", 30, 80, 520, 420)
    rows = [[r["name"].split(":")[0].split(" (")[0], f4(r["macro_f1"]), f4(r["excl"]) if r["excl"] is not None else "-", f"{r['params']:,}"]
            for r in sorted(ROWS, key=lambda r: -r["macro_f1"])]
    hi = [i for i, r in enumerate(sorted(ROWS, key=lambda r: -r["macro_f1"])) if r["key"] == "triage"][0]
    d.table(s, ["Model", "Macro-F1", "Excl. REAL_BUG", "Params"], rows, 560, 100, 380, [3.2, 1.3, 1.6, 1.4], size=11, rowh=34, hi=hi)
    d.text(s, f"Proposed vs best competitor: {diff:+.4f} macro-F1", 560, 360, 380, 28, size=16, bold=True, color=TEAL)
    d.text(s, "Competitors are re-implementations adapted to our 5-class task; single run each. With identical inputs, a plain 2-layer LSTM matches ours: inputs matter more than architecture here.", 560, 392, 380, 100, size=13, color=GREY)

    # 8 comparison details ------------------------------------------------------------------------
    s = d.slide("How the comparison was made, and how to read it", "Fairness points: same split, seed, optimiser, class weights, early stopping. The span-and-timing model is included so we can see how much of any advantage is just richer input.")
    d.table(s, ["Model", "Based on", "Reads", "Classifies with"],
            [["DeepLog-style", "Du et al., CCS 2017", "tokens", "2-layer LSTM, last state"],
             ["LogRobust-style", "Zhang et al., FSE 2019", "tokens", "attention BiLSTM"],
             ["Span + timing LSTM", "after Nedelkoski et al., 2019", "tokens + duration", "LSTM, last state"],
             ["TextCNN", "Kim, EMNLP 2014", "tokens", "conv 2/3/4 + max-pool"],
             ["DeepLog-style, our inputs", "Du et al. backbone", "same inputs as ours", "2-layer LSTM, last state"],
             ["Proposed", "this work", "tokens + host, agent, depth, duration + 9 features", "attention BiLSTM + max-pool"]],
            40, 88, 880, [2.2, 2.6, 3.0, 2.4], size=13, rowh=34, hi=5)
    d.bullets(s, ["Same split, seed, optimiser, class weights and early stopping for every model", "Three competitors see only operation tokens; one adds timing; one gets exactly our inputs, to separate input from architecture", "The order-free gradient-boosting baseline (0.7037) already shows this data rewards counts over sequence structure", "One run each: small differences are within seed noise"], 40, 335, 880, 190, size=16, gap=5)

    # 9 results -----------------------------------------------------------------------------------
    s = d.slide("Results on 10,513 unseen traces", "Our model beats the count-based representation Phase 1 used by a wide margin, and the strongest order-free baseline by about four points.")
    d.pic(s, "fig_baselines.png", 30, 80, 560, 260)
    d.kpi(s, 620, 92, 150, f4(OURS["macro_f1"]), "macro-F1")
    d.kpi(s, 785, 92, 150, f4(OURS["acc"]), "accuracy")
    d.kpi(s, 620, 192, 150, "0.6829", "macro-F1 excl. REAL_BUG", color=TEAL)
    d.kpi(s, 785, 192, 150, "0", "real bugs missed", color=GREEN)
    d.pic(s, "fig_cm.png", 30, 345, 270, 190)
    d.bullets(s, ["DATA_FAULT recall 0.913; INFRA_FLAKE 0.792", "PASS recall 0.603: about 40% of healthy traces flagged", "ENV_FAULT is the weak class (F1 0.481)"], 320, 345, 620, 170, size=17, gap=6)

    # 10 honest ----------------------------------------------------------------------------------
    s = d.slide("What the numbers really say", "These are the three things we do not want the panel to discover on their own.")
    d.bullets(s, ["REAL_BUG = 1.000 is a workload effect: those traces are short (median 7 spans, 2 hosts vs 88–134 spans, 9–12 hosts). We quote 0.6829 excluding it.",
                  "Order carries little signal: an order-free baseline ties the BiLSTM on the other four classes; shuffling tokens did not hurt.",
                  "Compound faults mostly fail: 77.8% of 20,010 multi-fault traces were called PASS.",
                  "Architecture is not what wins: a plain 2-layer LSTM given our exact inputs scores 0.7672, slightly above our 0.7464. The gain comes from the inputs (timing, host, trace features).",
                  "Stock Jaeger exports of our own traces were misjudged 4 times in 5 (20% accuracy, 90% called REAL_BUG), so the drift guard also demands an attested feature schema.",
                  "Root-cause ranking is weak for data faults (top-3 47.6% vs random 38.1%) and is measured against a proxy label."], 50, 92, 860, 420, size=18, gap=10)

    # 11 trust ------------------------------------------------------------------------------------
    s = d.slide("Trust engine: autonomy is earned and can be lost", "Promotion needs the Wilson lower bound of accuracy to clear a threshold on fresh evidence. A single missed real defect that was acted on autonomously costs a tier immediately.")
    d.pic(s, "fig_trust_fsm.png", 30, 80, 600, 230)
    d.pic(s, "fig_trust_replay.png", 30, 315, 560, 215)
    d.bullets(s, ["Wilson bound: 17/20 correct is 85%, but the bound is 0.64", "Replay on the test stream: T1 at decision 14, T2 at 906, T3 at 1,130", "Injected faults: demoted twice at once", "Bug found by the live run: tiers flipped 53 times in 50 decisions; fixed by counting only evidence since the last change"], 620, 100, 320, 420, size=15, gap=8)

    # 12 gate -------------------------------------------------------------------------------------
    s = d.slide("Release gate: the most conservative signal wins", "Security first, then canary, then abstain-if-unfamiliar, then the triage decision. The tier then decides who may execute it.")
    d.pic(s, "fig_gate.png", 30, 84, 440, 330)
    d.table(s, ["Module", "Measured"],
            [["Drift guard", "foreign trace: OOV 1.0, routed to a human"],
             ["Canary", "slow DB 16.7–20.9× p95, rolled back"],
             ["Security", "27 flagged, 0 reachable"],
             ["Chaos", "silent bug: 0% correct, 2 unsafe merges, T3→T0"],
             ["Actuator", "dry-run default, argv-only, validated names"]],
            490, 100, 450, [1.4, 4.2], size=13, rowh=44)

    # 13 live -------------------------------------------------------------------------------------
    s = d.slide("Live closed loop: 390 requests, faults injected on a schedule", f"The agent starts with no autonomy, earns T3, is wrong on the silent bug, loses autonomy, and regains it only through new evidence. {tc} of 390 correct.")
    d.pic(s, "fig_live_traj.png", 30, 80, 620, 330)
    d.kpi(s, 670, 92, 120, f"{tc / 390:.1%}", "correct decisions")
    d.kpi(s, 805, 92, 120, "2", "unsafe actions", color=RED)
    d.bullets(s, ["Promoted at requests 15, 45, 97 → T3", "Silent bug (looks healthy): 3 of 30 right; demoted at 342, 344 and by the retention check at 374", "Regained T1 at 389 after 15 correct decisions", "A toy system: the scalar baseline also scores 1.000 here"], 40, 415, 890, 120, size=15, gap=3)

    # 14 dashboard -------------------------------------------------------------------------------
    s = d.slide("Live demo: the dashboard", "Start with python scripts/serve_ui.py. The strip along the top is the whole pipeline. Fault buttons inject real faults; the tier chart moves live.")
    d.pic(s, "ui_live.png", 30, 80, 470, 340)
    d.pic(s, "ui_gate.png", 510, 80, 420, 250)
    d.bullets(s, ["Inject faults, run the scripted demo, switch on the chaos monkey", "Canary: deploy a new version and watch the gate roll it back", "Security, root-cause, real-data and metrics tabs"], 510, 340, 430, 180, size=15, gap=6)
    d.text(s, "python scripts/serve_ui.py  →  http://127.0.0.1:8765", 40, 440, 460, 30, size=15, bold=True, color=TEAL, font="Consolas")

    # 15 drift ------------------------------------------------------------------------------------
    s = d.slide("Knowing what it does not know", "The model only understands HDFS traces in the training schema. Rather than guess elsewhere, the gate abstains.")
    d.pic(s, "fig_fidelity.png", 30, 84, 520, 200)
    d.pic(s, "ui_hdfs.png", 560, 84, 380, 290)
    d.bullets(s, ["Corpus records 0.74 → attested export 0.63 → stock export 0.20", "Unknown vocabulary (OOV > 0.5) or unattested schema: ask a human", "The real Jaeger 1.76.0 round trip works: traces pushed over OTLP, read back, judged in-distribution"], 40, 300, 500, 220, size=16, gap=8)

    # 16 problems --------------------------------------------------------------------------------
    s = d.slide("Problems that measurement exposed", "Several of our best results were bugs we found because we measured.")
    d.table(s, ["Finding", "Fix"],
            [["88.18% was a one-column rule", "Rebuilt from sequences; per-class metrics and baselines"],
             ["Live test accuracy 1.0 with classes missing", "Scenario assigned once by dominant fault class"],
             ["53 tier flips in 50 decisions", "Only evidence since the last tier change counts"],
             ["Stock exports judged at 20% accuracy", "Feature-schema condition in the drift guard"],
             ["Attention made uniform by regularisation", "Rejected; spans reported as a hint"],
             ["localhost cost 2.1 s per call on Windows", "Scripts use 127.0.0.1 (26 ms)"]],
            40, 100, 880, [4, 4.4], size=16, rowh=54)

    # 17 limits ----------------------------------------------------------------------------------
    s = d.slide("Limitations and future work", "We state these ourselves. The future work follows directly from the limitations.")
    d.text(s, "Limitations", 50, 90, 420, 30, size=22, bold=True, color=RED)
    d.bullets(s, ["Valid only for HDFS traces in the training schema", "Credible accuracy is 0.6829, not 1.000", "No measured gain from attention over a plain LSTM on the same inputs", "Root-cause numbers use a proxy label", "Workflows, Grafana/Prometheus and real-infrastructure chaos not executed", "Comparison models are re-implementations, one run each"], 50, 124, 430, 380, size=16, gap=8)
    d.text(s, "Next steps", 500, 90, 420, 30, size=22, bold=True, color=GREEN)
    d.bullets(s, ["Train on a real microservice system with several scenarios per class", "Cut false alarms (40% of healthy traces)", "Multi-label output for compound faults", "Signals outside the trace for silent defects", "Run the unexecuted infrastructure; keep the gate advisory until T2"], 500, 124, 430, 380, size=16, gap=8)

    # 18 conclusion ------------------------------------------------------------------------------
    s = d.slide("Conclusion", "Close on the main idea: autonomy should be earned from evidence, and every limit should be measured.")
    d.bullets(s, ["A trace-based triage agent that maps five failure classes to pipeline actions", CONCL, "A trust engine that grants autonomy from a statistical bound and withdraws it at once on a missed defect", "A release gate, live system and dashboard that show the whole loop, including where it fails", "183 automated tests; every limit stated"], 50, 100, 860, 330, size=22, gap=14)
    d.text(s, "Thank you  ·  Questions?", 50, 450, 860, 50, size=30, bold=True, color=TEAL)
    d.save()
    print("saved", OUT, "slides:", d.n)


if __name__ == "__main__":
    build()
