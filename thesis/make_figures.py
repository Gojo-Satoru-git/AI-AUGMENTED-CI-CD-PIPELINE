"""Figures for the thesis: white background, black text, Times New Roman, grey boxes.\nEvery figure is drawn at the size it is printed (width in inches = printed width), with 8-9 pt text.\nNumbers are read from reports/*.json, never typed in."""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
R = lambda p: json.loads((ROOT / "reports" / p).read_text(encoding="utf-8"))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.edgecolor": "black", "axes.labelcolor": "black",
                     "text.color": "black", "xtick.color": "black", "ytick.color": "black",
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})
G1, G2, G3 = "#f2f2f2", "#dcdcdc", "#bdbdbd"
K = 0.80          # diagram font factor: the drawing is made at print size, so text ~8 pt


def canvas(w_in, aspect, xmax=100):
    """aspect = height / width. Axes units are 0..xmax horizontally, 0..xmax*aspect vertically."""
    fig, ax = plt.subplots(figsize=(w_in, w_in * aspect))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, xmax * aspect)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def box(ax, x, y, w, h, text, fc=G1, fs=10, bold=False, lw=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.0",
                                fc=fc, ec="black", lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs * K,
            fontweight="bold" if bold else "normal", linespacing=1.2)


def txt(ax, x, y, s, fs=10, **kw):
    ax.text(x, y, s, fontsize=fs * K, **kw)


def arrow(ax, x1, y1, x2, y2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                 color="black", shrinkA=0, shrinkB=0, ls=ls))


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ------------------------------------------------------------------ 3.1 architecture
def fig_arch():
    fig, ax = canvas(5.6, 0.72)
    layers = [("LAYER 1\nTrace\nsources", ["Minishop live\nsystem (4 services)", "TraceBench HDFS\n(363 scenarios)", "Jaeger / OTLP\nexport"]),
              ("LAYER 2\nAdapter and\nguards", ["Jaeger adapter\n(clock alignment)", "Feature schema\nattestation", "Drift guard\n(vocabulary, schema)"]),
              ("LAYER 3\nDecision\ncomponents", ["Triage model\n(attention BiLSTM)", "Root-cause\nlocalisation", "Trust engine\n(tiers T0 - T3)"]),
              ("LAYER 4\nRelease\ngate", ["Security gate\n(CVE reachability)", "Canary health\n(Wilson intervals)", "Release gate\n(precedence rules)"]),
              ("LAYER 5\nOutputs", ["Actuator\n(dry-run default)", "Prometheus metrics\nand dashboard", "Ground-truth feedback\nto the trust engine"])]
    top = 70.5
    h, gap = 11.6, 2.5
    y = top - h
    for title, items in layers:
        box(ax, 1, y, 15, h, title, fc=G3, fs=9, bold=True)
        for i, t in enumerate(items):
            box(ax, 19 + i * 27, y, 25, h, t, fc=G1, fs=9.5)
        y -= h + gap
    for k in range(4):
        yy = top - h - k * (h + gap)
        arrow(ax, 58, yy, 58, yy - gap)
    save(fig, "fig_arch.png")


# ------------------------------------------------------------------ 3.2 clock alignment
def fig_align():
    fig, ax = canvas(5.1, 0.44)
    txt(ax, 1, 41, "(a) As recorded: each host stamps events with its own clock", fs=10, fontweight="bold")
    txt(ax, 1, 33, "host A (root)", fs=10, va="center")
    box(ax, 22, 29.5, 42, 7, "parent operation", fc=G2, fs=10)
    txt(ax, 1, 24, "host B (+6.0 s)", fs=10, va="center")
    box(ax, 72, 20.5, 25, 7, "child operation", fc=G1, fs=10)
    txt(ax, 22, 16.5, "sorting by raw timestamps puts the child far after its parent", fs=9.5, style="italic")
    txt(ax, 1, 10.5, "(b) Aligned: the host offset comes from the edge record and is", fs=10, fontweight="bold")
    txt(ax, 1, 5.5, "propagated down the call tree, ties broken by depth", fs=10, fontweight="bold")
    txt(ax, 1, 0.8, "offset(B) = father_start + offset(A) - first_start(B)", fs=10)
    save(fig, "fig_align.png")


# ------------------------------------------------------------------ 3.3 model
def fig_model():
    fig, ax = canvas(5.9, 0.50)
    stages = [("Span\ntokens\n\"op|status\"", 1), ("Embeddings\n+ side\nfeatures", 17.5), ("BiLSTM\n64 units\nper direction", 34), ("Attention\npooling\n(+ max pool)", 50.5),
              ("Concat with\n9 trace-level\nfeatures", 67), ("Dense layers\nsoftmax\n5 classes", 83.5)]
    for t, x in stages:
        box(ax, x, 15, 15.5, 17, t, fc=G3 if x == 50.5 else G1, fs=9, bold=(x == 50.5))
    for (a, xa), (b, xb) in zip(stages, stages[1:]):
        arrow(ax, xa + 15.5, 23.5, xb, 23.5)
    box(ax, 47.5, 38.5, 21.5, 8, "attention weights = evidence spans", fc="white", fs=8.5, ls="--")
    arrow(ax, 58.2, 32, 58.2, 38.5)
    txt(ax, 50, 9, r"$\alpha_t = \mathrm{softmax}(w^{\top}\tanh(Wh_t))$,   $c = \sum_t \alpha_t h_t$", fs=11, ha="center")
    txt(ax, 50, 2.5, "every span contributes in proportion to its learned weight, not only the last hidden state",
        fs=9, ha="center", style="italic")
    ax.set_ylim(0, 50)
    save(fig, "fig_model.png")


# ------------------------------------------------------------------ 3.4 trust FSM
def fig_trust():
    fig, ax = canvas(5.6, 0.50)
    xs = [1, 28, 55, 82]
    names = ["T0\nobserve", "T1\nrecommend", "T2\nact + rollback", "T3\nautonomous"]
    for x, n in zip(xs, names):
        box(ax, x, 27, 15, 14, n, fc=G1, fs=10, bold=True)
    th = ["n >= 15\nbound >= 0.70", "n >= 30\nbound >= 0.85", "n >= 50\nbound >= 0.93"]
    for i in range(3):
        arrow(ax, xs[i] + 15, 36, xs[i + 1], 36)
        txt(ax, xs[i] + 21, 37.5, th[i], fs=8.5, ha="center", va="bottom")
        arrow(ax, xs[i + 1], 30, xs[i] + 15, 30)
        txt(ax, xs[i] + 21, 28.5, "demotion", fs=8.5, ha="center", va="top")
    txt(ax, 50, 15.5, "promotion: Wilson lower bound over evidence earned since the last tier change", fs=9.5, ha="center")
    txt(ax, 50, 6, "demotion: one tier at once when a missed REAL_BUG was acted on autonomously,\nor, optionally, when the accuracy floor of the current tier fails",
        fs=9.5, ha="center")
    ax.set_ylim(1, 50)
    save(fig, "fig_trust_fsm.png")


# ------------------------------------------------------------------ 3.5 gate
def fig_gate():
    fig, ax = canvas(5.1, 0.62)
    steps = [("Security verdict = BLOCK ?", "BLOCK_MERGE", 47), ("Canary verdict = ROLLBACK ?", "ROLLBACK_CANARY", 33),
             ("Trace out of distribution ?", "REVIEW (abstain)", 19), ("Triage decision\n(policy.decide)", "ALLOW / RETRY / ESCALATE\n/ QUARANTINE / BLOCK", 5)]
    for t, out, y in steps:
        box(ax, 3, y, 40, 10, t, fc=G1, fs=10)
        box(ax, 58, y, 39, 10, out, fc=G2, fs=9.5)
        arrow(ax, 43, y + 5, 58, y + 5)
        if y > 5:
            txt(ax, 50, y + 6, "yes", fs=9, ha="center")
    for y in (47, 33, 19):
        arrow(ax, 23, y, 23, y - 4)
        txt(ax, 25, y - 3.4, "no", fs=9)
    txt(ax, 50, 60.5, "The most conservative signal wins; the trust tier decides who may execute it", fs=10, ha="center", fontweight="bold")
    ax.set_ylim(2, 63)
    save(fig, "fig_gate.png")


# ------------------------------------------------------------------ 3.6 closed loop
def fig_loop():
    fig, ax = canvas(5.6, 0.44)
    n = [("Request served\nby the system", 1, 28), ("Trace\n(Jaeger / OTLP)", 26, 28), ("Adapter and\ndrift guard", 51, 28), ("Triage, policy\nand release gate", 76, 28)]
    for t, x, y in n:
        box(ax, x, y, 21, 12, t, fc=G1, fs=9.5)
    for (_, xa, ya), (_, xb, yb) in zip(n, n[1:]):
        arrow(ax, xa + 21, ya + 6, xb, yb + 6)
    box(ax, 76, 4, 21, 12, "Decision and\ndry-run actuation", fc=G2, fs=9.5)
    arrow(ax, 86.5, 28, 86.5, 16)
    box(ax, 40, 4, 29, 12, "True outcome fed back\n(exact label)", fc=G2, fs=9.5)
    arrow(ax, 76, 10, 69, 10)
    box(ax, 1, 4, 29, 12, "Trust engine updates\nthe tier (T0 - T3)", fc=G3, fs=9.5, bold=True)
    arrow(ax, 40, 10, 30, 10)
    arrow(ax, 15.5, 16, 15.5, 28)
    txt(ax, 17.5, 21.5, "next request is governed\nby the new tier", fs=8.5, va="center")
    ax.set_ylim(0, 44)
    save(fig, "fig_loop.png")


# ------------------------------------------------------------------ results charts
def bars(ax, labels, vals, hi=None, fmt="{:.3f}"):
    cols = ["black" if i == hi else "#9e9e9e" for i in range(len(vals))]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.5, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, fmt.format(v), va="center", fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


def fig_baselines():
    b = R("baselines.json")
    b5 = R("baseline_B5.json")["macro_f1"]
    ev = R("evaluation.json")["model"]["macro_f1"]
    rows = [("B1 majority class", b["B1_majority"]["macro_f1"]),
            ("B3 bag-of-ops counts, LR", b["B3_bag_of_ops_counts_lr"]["macro_f1"]),
            ("B4 bag-of-ops TF-IDF, SVM", b["B4_bag_of_ops_tfidf_svm"]["macro_f1"]),
            ("B2 task scalars, GBM", b["B2_task_scalars_gbm"]["macro_f1"]),
            ("B5 scalars + bag-of-ops, GBM", b5),
            ("Attention BiLSTM (proposed)", ev)]
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    bars(ax, [r[0] for r in rows], [r[1] for r in rows], hi=len(rows) - 1)
    ax.set_xlabel("Macro-F1 on the 10,513 held-out test traces", fontsize=9)
    save(fig, "fig_baselines.png")


def fig_cm():
    ev = R("evaluation.json")["model"]
    L, M = ev["labels"], np.array(ev["confusion_matrix"])
    P = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    ax.imshow(P, cmap="Greys", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:,}\n({P[i, j]:.0%})", ha="center", va="center", fontsize=7.5,
                    color="white" if P[i, j] > 0.55 else "black")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([l.replace("_", "\n") for l in L], fontsize=7.5)
    ax.set_yticklabels(L, fontsize=7.5)
    ax.set_xlabel("Predicted class", fontsize=9); ax.set_ylabel("True class", fontsize=9)
    save(fig, "fig_cm.png")


def fig_rca():
    r = R("rca_evaluation.json")
    classes = ["INFRA_FLAKE", "ENV_FAULT", "DATA_FAULT", "COMPOUND"]
    x = np.arange(len(classes)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    for k, (key, lab, col) in enumerate([("ours", "Proposed", "black"), ("slowest_raw", "Slowest span", "#8c8c8c"), ("random", "Random span", "#d0d0d0")]):
        ax.bar(x + (k - 1) * w, [r[key][c]["top3"] for c in classes], w, label=lab, color=col, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=8.5)
    ax.set_ylabel("Top-3 localisation rate", fontsize=9); ax.set_ylim(0, 1.18)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    save(fig, "fig_rca.png")


def fig_fidelity():
    f = R("adapter_fidelity.json")
    rows = [("Corpus records (training format)", f["corpus"]["accuracy"]),
            ("Jaeger export, training schema", f["rich_export"]["accuracy"]),
            ("Stock Jaeger export", f["bare_export"]["accuracy"])]
    fig, ax = plt.subplots(figsize=(5.0, 1.7))
    bars(ax, [r[0] for r in rows], [r[1] for r in rows], hi=2, fmt="{:.2f}")
    ax.set_xlabel("Accuracy on 400 held-out traces", fontsize=9)
    save(fig, "fig_fidelity.png")


def fig_chaos():
    c = R("chaos_report.json")["by_fault"]
    names = list(c)
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.bar(x - w / 2, [c[n]["flagged_as_fault_rate"] for n in names], w, label="flagged as a fault", color="#bdbdbd", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, [c[n]["class_match_rate"] for n in names], w, label="assigned the correct class", color="black", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7.5)
    ax.set_ylim(0, 1.2); ax.set_ylabel("Rate over 30 injections", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    save(fig, "fig_chaos.png")


def fig_live():
    sys.path.insert(0, str(ROOT))
    from src.live.plot import plot_trajectory
    from src.live.loop import SCHEDULE
    st = json.loads((OUT.parent / "live_state.json").read_text(encoding="utf-8"))
    ev = {e["step"]: e for e in st["events"]}
    steps = [{"step": i + 1, "tier": t, "ok": bool(ok), "unsafe": ("x" if un else None),
              "tier_action": (ev[i + 1]["action"] if i + 1 in ev else None)}
             for i, (t, f, ok, un) in enumerate(st["series"])][:390]
    ph, first = [], 1
    for kind, n in SCHEDULE:
        ph.append({"fault": kind, "n": n, "first_step": first})
        first += n
    plot_trajectory(steps, ph, OUT / "fig_live_traj.png", figsize=(6.2, 2.9),
                    title="Agent trust tier during the scripted live run (390 requests)")


def fig_replay():
    tt = R("trust_trajectory.json")
    steps = tt["steps"]
    x = list(range(len(steps)))
    y = [s["tier_after"] for s in steps]
    inj = next(i for i, s in enumerate(steps) if s["note"] != "honest")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.4), gridspec_kw={"width_ratios": [2.3, 1]})
    for ax, lo, hi in ((a1, 0, len(x)), (a2, inj - 12, len(x))):
        ax.step(x[lo:hi], y[lo:hi], where="post", color="black", lw=1.3)
        ax.axvline(inj, color="#7a7a7a", ls="--", lw=0.8)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["T0", "T1", "T2", "T3"], fontsize=8)
        ax.set_ylim(-0.2, 3.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("decision #", fontsize=8.5)
    for i, s in enumerate(steps):
        if s.get("tier_action") in ("promoted", "demoted"):
            m = "^" if s["tier_action"] == "promoted" else "v"
            a1.plot(i, s["tier_after"], marker=m, color="black", ms=4)
            if i >= inj - 12:
                a2.plot(i, s["tier_after"], marker=m, color="black", ms=5)
    a1.set_title("Honest test stream; dashed line: fault injection", fontsize=8.5)
    a2.set_title("Zoom on the injected faults", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_trust_replay.png")


def fig_cmp():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cmp_data
    rows = sorted(cmp_data.load(), key=lambda r: r["macro_f1"])
    fig, ax = plt.subplots(figsize=(5.4, 0.55 * len(rows) + 0.9))
    names = [r["name"].replace("Proposed: ", "").replace(": ", "\n", 1) if r["key"] != "triage" else "Proposed:\nattention BiLSTM + side features" for r in rows]
    cols = ["black" if r["key"] == "triage" else "#9e9e9e" for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, [r["macro_f1"] for r in rows], color=cols, edgecolor="black", linewidth=0.5, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    for yi, r in zip(y, rows):
        ax.text(r["macro_f1"] + 0.005, yi, f"{r['macro_f1']:.4f}", va="center", fontsize=8.5)
    lo = min(r["macro_f1"] for r in rows)
    ax.set_xlim(max(0, lo - 0.08), max(r["macro_f1"] for r in rows) + 0.06)
    ax.set_xlabel("Macro-F1 on the 10,513 held-out test traces (axis does not start at 0)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    save(fig, "fig_cmp.png")


if __name__ == "__main__":
    for f in (fig_arch, fig_align, fig_model, fig_trust, fig_gate, fig_loop, fig_baselines, fig_cm,
              fig_rca, fig_fidelity, fig_chaos, fig_live, fig_replay, fig_cmp):
        f()
    print(sorted(p.name for p in OUT.iterdir()))
