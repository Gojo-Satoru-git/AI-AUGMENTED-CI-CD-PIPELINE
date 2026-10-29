"""
Labelled Jaeger exports -> training corpus, then train a model for YOUR system.

    python scripts/jaeger_to_corpus.py --manifest experiments/manifest.json \\
        --out data/processed/corpus_live.jsonl.gz

    python src/models/train.py --corpus data/processed/corpus_live.jsonl.gz \\
        --out src/models/saved_live --reports reports/live \\
        --feature-schema jaeger_adapter_v1 --weight-power 0.5 --epochs 20 --patience 4

    # serve it
    TRIAGE_MODEL_DIR=src/models/saved_live python run_api.py

See src/data/jaeger_corpus.py for the manifest format and why the split needs
>= 3 independent scenarios per class. USER_TODO.md walks through collecting the
labelled experiments from a live TrainTicket / HotROD deployment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.jaeger_corpus import build_corpus   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/processed/corpus_live.jsonl.gz")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    info = build_corpus(args.manifest, args.out)
    print(json.dumps(info, indent=2))
    thin = [c for c, n in info["scenarios_per_class"].items() if n < 3]
    if thin:
        print(f"\nWARNING: fewer than 3 scenarios for {thin} - the grouped split "
              "cannot keep them out of the test set. Run more independent "
              "experiments for these classes.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
