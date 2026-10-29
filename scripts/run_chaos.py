"""
S7: run the chaos experiments against the trained triage agent.

    python scripts/run_chaos.py                    # 30 injections per fault, start at T3
    python scripts/run_chaos.py --n 60 --severity 0.5
    python scripts/run_chaos.py --sweep            # detection vs severity curve

Faults are injected into HEALTHY held-out test traces (src/chaos/experiments.py),
streamed through triage -> policy -> trust engine, and the result is written to
reports/chaos_report.json and reports/chaos_report.md.

The agent starts at T3 with accuracy-retention demotion ON, so you can watch
the trust engine react when chaos degrades the agent's decisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chaos.experiments import FAULTS, run_chaos, severity_sweep   # noqa: E402
from src.data.dataset import prepare                                   # noqa: E402
from src.models.predict import TriagePredictor                         # noqa: E402

CORPUS = ROOT / "data" / "processed" / "corpus.jsonl.gz"


def to_markdown(rep):
    L = ["# Chaos experiment report", "",
         f"severity {rep['severity']} | {rep['n_per_fault']} injections per fault "
         f"| seed {rep['seed']} | agent started at T{rep['start_tier']} "
         f"(retention demotion {'ON' if rep['retention_enabled'] else 'OFF'})", "",
         "Faults are SYNTHETIC injections built from signatures seen in the real "
         "TraceBench fault traces, applied to healthy held-out traces. They stress-"
         "test the agent's sensitivity; they do not replace reports/evaluation.json.",
         "",
         "| fault | TraceBench family | should be | n | flagged as fault | "
         "class match | mean conf | unsafe autonomous actions |",
         "|---|---|---|---|---|---|---|---|"]
    for n, v in rep["by_fault"].items():
        L.append(f"| {n} | {v['family']} | {v['truth']} | {v['n']} | "
                 f"{v['flagged_as_fault_rate']:.0%} | {v['class_match_rate']:.0%} | "
                 f"{v['mean_confidence']:.2f} | {v['unsafe_autonomous_actions']} |")
    L += ["", f"**Trust engine:** started T{rep['start_tier']}, ended "
          f"T{rep['final_tier']}, {len(rep['demotions'])} demotion(s).", ""]
    for d in rep["demotions"]:
        L.append(f"- experiment #{d['experiment']} ({d['fault']}): "
                 f"T{d['from']} -> T{d['to']} - {d['reason']}")
    L += ["", f"**Unsafe autonomous actions:** {rep['unsafe_actions'] or 'none'}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="injections per fault")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--start-tier", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-retention", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    predictor = TriagePredictor()
    ds, *_ = prepare(str(CORPUS), max_len=predictor.max_len)
    healthy = [r for r in ds["test"].recs if r["cls"] == "PASS"]
    print(f"{len(healthy)} healthy held-out traces available; "
          f"{len(FAULTS)} fault types")

    rep_dir = Path(args.reports)
    rep_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        sweep = severity_sweep(predictor, healthy, n_per_fault=args.n,
                               seed=args.seed)
        (rep_dir / "chaos_severity_sweep.json").write_text(json.dumps(sweep, indent=2))
        print("\nflagged-as-fault rate by severity")
        cols = list(next(iter(sweep.values())))
        print(f"{'severity':<10}" + "".join(f"{c[:18]:<20}" for c in cols))
        for sev, row in sweep.items():
            print(f"{sev:<10}" + "".join(f"{row[c]:<20.0%}" for c in cols))
        return

    rep = run_chaos(predictor, healthy, n_per_fault=args.n,
                    severity=args.severity, seed=args.seed,
                    start_tier=args.start_tier,
                    retention=not args.no_retention)
    (rep_dir / "chaos_report.json").write_text(json.dumps(rep, indent=2))
    md = to_markdown(rep)
    (rep_dir / "chaos_report.md").write_text(md)
    print(md)
    print(f"wrote {rep_dir/'chaos_report.json'} and {rep_dir/'chaos_report.md'}")


if __name__ == "__main__":
    main()
