"""
How much accuracy does the Jaeger route cost?

The reported test metrics (reports/evaluation.json) feed the model corpus
records: real host-role / agent codes and task-level counts computed by
src/data/build_sequences.py. A live trace arrives as Jaeger JSON, and an
ordinary tracer cannot supply all of that. This script re-exports held-out test
traces as Jaeger JSON two ways and measures what the model does with each:

  corpus path   record -> model                          (the reported numbers)
  rich export   Jaeger JSON incl. triage.agent tags and
                triage_extra task counts -> adapter -> model
                (an exporter that knows the training schema)
  bare export   plain Jaeger JSON, nothing extra -> adapter -> model
                (what a stock tracer gives you)

    python scripts/adapter_fidelity.py            # 400 traces
    python scripts/adapter_fidelity.py --n 1000

Writes reports/adapter_fidelity.json. These numbers are the honest answer to
"how good is the agent on a live trace?" - for HDFS-origin traces, an UPPER
bound, since a different system also brings an unseen vocabulary (the drift
guard's job).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from make_samples import record_to_jaeger                        # noqa: E402
from src.data.dataset import prepare                              # noqa: E402
from src.models.predict import TriagePredictor                    # noqa: E402
from src.observability.jaeger_adapter import load_jaeger          # noqa: E402


def macro_f1(y, p, labels):
    f1s = []
    for c in labels:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / len(f1s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="reports/adapter_fidelity.json")
    args = ap.parse_args()

    pred = TriagePredictor()
    ds, *_ = prepare(str(ROOT / "data/processed/corpus.jsonl.gz"), max_len=pred.max_len)
    # exclude very short traces only to keep exports meaningful (>=3 spans)
    pool = [r for r in ds["test"].recs if r["toks"].count(" ") + 1 >= 3]
    random.Random(args.seed).shuffle(pool)
    sample = pool[: args.n]

    truth, out = [], collections.defaultdict(list)
    for r in sample:
        truth.append(r["cls"])
        out["corpus"].append(pred.predict_record(r)["classification"])
        for name, rich in (("rich_export", True), ("bare_export", False)):
            doc = record_to_jaeger(r, rich=rich)
            _t, spans, extra = next(iter(load_jaeger({"data": [doc]})))
            out[name].append(pred.predict(spans=spans, extra=extra)["classification"])

    labels = sorted(set(truth))
    res = {"n_traces": len(sample), "classes": dict(collections.Counter(truth))}
    for name, preds in out.items():
        res[name] = {
            "accuracy": round(sum(a == b for a, b in zip(truth, preds)) / len(truth), 4),
            "macro_f1": round(macro_f1(truth, preds, labels), 4),
            "agreement_with_corpus_path": round(
                sum(a == b for a, b in zip(out["corpus"], preds)) / len(truth), 4),
            "predicted": dict(collections.Counter(preds)),
        }
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"{'path':<14}{'accuracy':>10}{'macro-F1':>10}{'agrees w/ corpus':>19}")
    for name in ("corpus", "rich_export", "bare_export"):
        r = res[name]
        print(f"{name:<14}{r['accuracy']:>10.3f}{r['macro_f1']:>10.3f}"
              f"{r['agreement_with_corpus_path']:>19.3f}")
    print(f"\n({len(sample)} held-out test traces)   wrote {args.out}")


if __name__ == "__main__":
    main()
