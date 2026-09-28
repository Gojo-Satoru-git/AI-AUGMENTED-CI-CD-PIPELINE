"""
S4 exit-criterion driver: build the PASS-latency baseline, run top-k
localization accuracy vs. two baselines, and emit one example postmortem.

    python scripts/run_rca_eval.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rca.localize import (build_baseline, evaluate_localization,  # noqa: E402
                              generate_postmortem, save_baseline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl.gz")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out-baseline",
                    default="src/models/saved/op_latency_baseline.json")
    args = ap.parse_args()

    rep = Path(args.reports); rep.mkdir(parents=True, exist_ok=True)

    print("building per-op PASS-latency baseline from corpus ...")
    baseline = build_baseline(args.corpus)
    save_baseline(baseline, args.out_baseline)
    n_ops = len(baseline) - 1  # exclude __global__
    print(f"  {n_ops} distinct ops baselined "
          f"(global n={baseline['__global__']['n']})")

    print("evaluating top-k localization accuracy ...")
    metrics = evaluate_localization(args.corpus, baseline, ks=(1, 3, 5))
    (rep / "rca_evaluation.json").write_text(json.dumps(metrics, indent=2))
    for name, per_cls in metrics.items():
        n = per_cls.pop("__n_traces_scored__", None)
        print(f"\n{name}  (n={n} traces with a locatable error span)")
        for cls, ks in sorted(per_cls.items()):
            print(f"  {cls:<12} " + "  ".join(
                f"{k}={v}" for k, v in ks.items()))

    # one worked example for the report
    with gzip.open(args.corpus, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["cls"] == "DATA_FAULT":
                example = r
                break
    from src.policy_engine.policy import decide
    d = decide(example["cls"], 0.9, trust_tier=3)  # ground-truth class here
    # only for a readable worked example; the postmortem itself never needs
    # ground truth in production, only the model's own prediction.
    md = generate_postmortem(example, baseline, example["cls"], d.action)
    (rep / "example_postmortem.md").write_text(md)
    print(f"\nwrote {rep/'rca_evaluation.json'} and "
          f"{rep/'example_postmortem.md'} (task {example['task']})")


if __name__ == "__main__":
    main()
