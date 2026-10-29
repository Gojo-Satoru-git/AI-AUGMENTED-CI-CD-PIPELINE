"""
One command, one narrated run of the whole project.

    python scripts/showcase.py                 # ~1 min, no pauses
    python scripts/showcase.py --pause         # press Enter between sections (live presentation)
    python scripts/showcase.py --otlp http://127.0.0.1:4318/v1/traces    # also push to a real Jaeger

Sections: (1) the system and what a fault looks like, (2) the credible accuracy
numbers on real data, (3) the live model, (4) the closed loop where the agent
earns and loses autonomy, (5) scorecard and honest limits.

Writes showcase_output/{transcript.txt, summary.md, live_trajectory.png}.
Builds the live model first if it does not exist (~2 min, one time).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_peek import show                       # noqa: E402
from src.live.loop import DEFAULT_MS, run_loop           # noqa: E402
from src.live.plot import LABELS, plot_trajectory        # noqa: E402

OUT = ROOT / "showcase_output"
MODEL_DIR = ROOT / "src" / "models" / "saved_live"
PY = sys.executable


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def banner(n, title, pause):
    if pause:
        input("\n[Enter] ")
    print(f"\n{'=' * 78}\n  {n}. {title}\n{'=' * 78}")


def build_live_model():
    steps = [
        [PY, "scripts/live_collect.py"],
        [PY, "scripts/jaeger_to_corpus.py", "--manifest", "live_data/manifest.json",
         "--out", "data/processed/corpus_live.jsonl.gz"],
        [PY, "src/models/train.py", "--corpus", "data/processed/corpus_live.jsonl.gz",
         "--out", "src/models/saved_live", "--reports", "reports/live",
         "--feature-schema", "jaeger_adapter_v1", "--weight-power", "0.5", "--epochs", "40",
         "--patience", "8", "--batch-size", "32", "--lr", "0.005", "--hidden", "32",
         "--emb-dim", "24", "--max-len", "32"],
    ]
    for c in steps:
        print("  building:", " ".join(Path(x).name if i == 0 else x for i, x in enumerate(c))[:100])
        subprocess.run(c, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def jaeger_services(base):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/api/services", timeout=5) as r:
            return json.loads(r.read()).get("data") or []
    except Exception:
        return None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", action="store_true", help="wait for Enter between sections")
    ap.add_argument("--otlp", help="OTLP/HTTP traces URL of a real Jaeger")
    ap.add_argument("--jaeger", default="http://127.0.0.1:16686", help="Jaeger query URL")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    sys.stdout = Tee(OUT / "transcript.txt")

    print("AI-augmented CI/CD triage agent - live showcase")
    print("Team 18, MIT Anna University. Everything below runs on this machine.")

    # ---------------------------------------------------------------- 1
    banner(1, "THE SYSTEM: a 4-service shop with real faults injected into it", a.pause)
    print("frontend -> orders -> inventory -> db. Every request is traced; the fault injector\n"
          "knows exactly whether the fault bit, so every trace has a true label.\n")
    for kind in ("none", "flaky_net", "host_down", "silent_bug"):
        show(kind, 3)

    # ---------------------------------------------------------------- 2
    banner(2, "CREDIBLE NUMBERS: the model on real HDFS traces (TraceBench, 3 GB)", a.pause)
    ev = json.loads((ROOT / "reports/evaluation.json").read_text())
    base = json.loads((ROOT / "reports/baselines.json").read_text())
    b5 = json.loads((ROOT / "reports/baseline_B5.json").read_text())
    m, op = ev["model"], ev["operational"]
    rows = [("B1 majority class", base["B1_majority"]["macro_f1"]),
            ("B3 bag-of-ops counts (Phase 1's representation)",
             base["B3_bag_of_ops_counts_lr"]["macro_f1"]),
            ("B2 task scalars, GBM", base["B2_task_scalars_gbm"]["macro_f1"]),
            ("B5 scalars + bag-of-ops, GBM", b5["macro_f1"]),
            ("BiLSTM + attention (ours)", m["macro_f1"])]
    print(f"held-out test: {op['test_traces']:,} traces from scenarios never seen in training\n")
    for name, v in rows:
        print(f"  {name:<50} macro-F1 {v:.3f}  {'#' * int(v * 40)}")
    print(f"\n  accuracy {m['accuracy']:.3f} | REAL_BUG missed {op['real_bugs_missed']} | "
          f"healthy-looking failures wrongly retried {op['wrongly_retried']} "
          f"({op['wrongly_retried_rate']:.1%})")
    print("  Phase 1 reported 88.18%; that was a one-column rule. This is the honest figure.")

    # ---------------------------------------------------------------- 3
    banner(3, "THE LIVE MODEL: trained on traces from the shop above", a.pause)
    if not (MODEL_DIR / "triage_lstm.pth").exists():
        print("  live model not found - building it once (collect -> corpus -> train)...")
        build_live_model()
    from src.models.predict import TriagePredictor
    predictor = TriagePredictor(ckpt=MODEL_DIR / "triage_lstm.pth",
                                vocab_path=MODEL_DIR / "vocab.json")
    lev = json.loads((ROOT / "reports/live/evaluation.json").read_text())
    print(f"  schema {predictor.feature_schema} | classes {predictor.labels}")
    print(f"  held-out test macro-F1 {lev['model']['macro_f1']:.3f}")
    print("  CAVEAT: the injected faults are simple - a plain scalar baseline scores "
          f"{lev['baselines']['B2_task_scalars_gbm']['macro_f1']:.3f} too.\n"
          "  So this proves the PIPELINE works; it is not the accuracy claim (section 2 is).")

    # ---------------------------------------------------------------- 4
    banner(4, "THE CLOSED LOOP: the agent earns autonomy, then loses it", a.pause)
    print("The agent starts at T0 (observe only). Every request: trace -> model -> policy ->\n"
          "gate -> decision; then the TRUE outcome is fed back and the trust tier may move.\n"
          "Promotion needs a Wilson lower bound over fresh evidence; a missed real bug demotes.\n")
    steps, phases, done, cur = [], [], None, None
    for e in run_loop(predictor=predictor, ms=DEFAULT_MS, seed=a.seed, scale=a.scale,
                      otlp=a.otlp):
        if e["type"] == "phase":
            if cur:
                print(f"    -> agent right {cur['ok']}/{cur['n']}")
            cur = {"ok": 0, "n": 0}
            phases.append({"fault": e["fault"], "n": e["n"], "first_step": e["first_step"]})
            print(f"  [{e['first_step']:>3}] {LABELS.get(e['fault'], e['fault']).upper():<12}"
                  f"({e['n']} requests)")
        elif e["type"] == "step":
            steps.append(e)
            cur["n"] += 1
            cur["ok"] += e["ok"]
            if e["tier_action"]:
                arrow = "UP  " if e["tier_action"] == "promoted" else "DOWN"
                print(f"      step {e['step']:>3}: tier {arrow} T{e['from_tier']} -> "
                      f"T{e['to_tier']}  {e['reason'][:60]}")
            if e["unsafe"]:
                print(f"      step {e['step']:>3}: UNSAFE autonomous action ({e['unsafe']}) "
                      f"truth={e['truth']} agent said {e['pred']}")
        else:
            done = e
    print(f"    -> agent right {cur['ok']}/{cur['n']}")

    chart = plot_trajectory(steps, phases, OUT / "live_trajectory.png", start_tier=0)

    # ---------------------------------------------------------------- 5
    banner(5, "SCORECARD AND HONEST LIMITS", a.pause)
    promos = sum(1 for t in done["tier_events"] if t["action"] == "promoted")
    demos = sum(1 for t in done["tier_events"] if t["action"] == "demoted")
    print(f"  requests served ............ {done['requests']}")
    print(f"  agent correct .............. {done['agent_correct']:.1%}")
    print(f"  trust tier ................. T{done['start_tier']} -> T{done['end_tier']} "
          f"({promos} promotions, {demos} demotions)")
    print(f"  autonomous actuations ...... {done['autonomous_actuations']} (dry-run)")
    print(f"  unsafe autonomous actions .. {done['unsafe_autonomous_actions']}")
    jaeger_line = None
    if a.otlp:
        svcs = jaeger_services(a.jaeger)
        jaeger_line = ("Jaeger push failed: " + str(done["push_err"]) if done["push_err"]
                       else f"traces visible in real Jaeger, services: {svcs}")
        print(f"  Jaeger ..................... {jaeger_line}")
    print(f"\n  chart: {chart}")
    limits = [
        "the live shop is a toy; real-data quality is the HDFS macro-F1 "
        f"{m['macro_f1']:.3f} (about 40% of healthy HDFS traces are flagged)",
        "the 'silent bug' is invisible in traces by construction - the agent is wrong on it, "
        "and the trust engine demoting it is the point",
        "the model is valid only inside its training schema; the gate abstains outside it",
        "GitHub workflows and the Grafana/Prometheus compose stack have not been run",
    ]
    print("\n  Limits we state ourselves:")
    for x in limits:
        print("   -", x)

    md = [
        "# Showcase run summary", "",
        f"- Requests: {done['requests']}; agent correct {done['agent_correct']:.1%}",
        f"- Trust tier T{done['start_tier']} -> T{done['end_tier']} "
        f"({promos} promotions, {demos} demotions)",
        f"- Unsafe autonomous actions: {done['unsafe_autonomous_actions']}",
        f"- HDFS held-out macro-F1: {m['macro_f1']:.3f} (best baseline {b5['macro_f1']:.3f})",
        *( [f"- {jaeger_line}"] if jaeger_line else [] ), "",
        "![trajectory](live_trajectory.png)", "", "## Limits", *[f"- {x}" for x in limits], "",
        "## Tier changes", *[f"- step {t['step']}: {t['action']} T{t['from']}->T{t['to']} "
                             f"{t['reason'][:90]}" for t in done["tier_events"]],
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    sys.stdout = sys.stdout.o
    print(f"\nsaved: {OUT / 'transcript.txt'}, {OUT / 'summary.md'}, {chart}")


if __name__ == "__main__":
    main()
