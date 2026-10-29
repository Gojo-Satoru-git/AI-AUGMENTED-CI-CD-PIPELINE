"""
Quantify "attention collapse": does the model spread attention over the spans
that matter, or dump nearly all weight on a single span?

    python scripts/attention_stats.py                       # current default checkpoint
    python scripts/attention_stats.py --ckpt src/models/saved/attn_lstm.pth

Reported on held-out test traces with at least 10 spans (shorter traces cannot
meaningfully "spread"):
    mean_top1_weight    average weight of the single most-attended span
    collapsed_frac      share of traces whose top span holds > 90% of the weight
    norm_entropy        entropy / log(n): 0 = one span takes all, 1 = uniform
    eff_spans           exp(entropy): how many spans' worth of attention there is
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import N_DEPTHS, UNK, prepare, task_features   # noqa: E402
from src.models.predict import TriagePredictor                         # noqa: E402


@torch.no_grad()
def attention_of(pred: TriagePredictor, r):
    toks = r["toks"].split(" ")[: pred.max_len]
    n = len(toks)
    batch = {
        "ids": torch.tensor([[pred.vocab.get(t, UNK) for t in toks]]),
        "hr": torch.tensor([[int(c) for c in r["hr"]][:n]]),
        "ag": torch.tensor([[int(c) for c in r["ag"]][:n]]),
        "dep": torch.tensor([[min(int(x), N_DEPTHS - 1) for x in r["dep"].split(",")][:n]]),
        "dur": torch.tensor([[math.log1p(max(int(x), 0)) for x in r["dur"].split(",")][:n]],
                            dtype=torch.float32),
        "mask": torch.ones(1, n, dtype=torch.bool),
        "lens": torch.tensor([n]),
        "feat": ((torch.tensor([task_features(r)], dtype=torch.float32)
                  - pred.feat_mean) / pred.feat_std),
    }
    _, w = pred.model(batch, return_attention=True)
    return w[0][:n]


def stats(pred, recs):
    top1, collapsed, ent, eff = [], 0, [], []
    for r in recs:
        w = attention_of(pred, r)
        n = len(w)
        h = float(-(w * torch.log(w.clamp_min(1e-9))).sum())
        top1.append(float(w.max()))
        collapsed += float(w.max()) > 0.9
        ent.append(h / math.log(n))
        eff.append(math.exp(h))
    k = len(recs)
    return {"n_traces": k,
            "mean_top1_weight": round(sum(top1) / k, 4),
            "collapsed_frac": round(collapsed / k, 4),
            "norm_entropy": round(sum(ent) / k, 4),
            "eff_spans": round(sum(eff) / k, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="src/models/saved/triage_lstm.pth")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    pred = TriagePredictor(ckpt=args.ckpt)
    ds, *_ = prepare(str(ROOT / "data/processed/corpus.jsonl.gz"), max_len=pred.max_len)
    pool = [r for r in ds["test"].recs if r["toks"].count(" ") + 1 >= 10]
    random.Random(args.seed).shuffle(pool)
    out = stats(pred, pool[: args.n])
    print(json.dumps({"checkpoint": args.ckpt, **out}, indent=2))


if __name__ == "__main__":
    main()
