"""Train the Test-Triage Agent. Seeded, early-stopped on validation macro-F1."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import (collate, prepare, N_TASK_FEATS,  # noqa: E402
                              BucketBatchSampler, seq_lengths)
from src.features.triage_lstm import TriageLSTM               # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / len(f1s), f1s


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    ys, ps = [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        ps.append(model(b).argmax(-1).cpu())
        ys.append(b["y"].cpu())
    y, p = torch.cat(ys), torch.cat(ps)
    mf1, per = macro_f1(y, p, n_classes)
    return mf1, per, float((y == p).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl.gz")
    ap.add_argument("--out", default="src/models/saved")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-tokens", action="store_true",
                    help="ablation: destroy order to test if sequence matters")
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="dampen class weights: weight_c = "
                         "(total/(K*count_c))**power. 1.0 = full inverse-"
                         "frequency (default); < 1.0 softens it. ENV_FAULT's "
                         "raw weight (3.18x) makes it a false-positive magnet "
                         "(precision 0.28 despite recall 0.71) - a fractional "
                         "power keeps the minority-class boost without letting "
                         "it dominate the decision boundary.")
    ap.add_argument("--tag", default=None,
                    help="override the output filename tag (default: "
                         "'shuffled' or 'triage'); use this to run an "
                         "experiment without overwriting the current best "
                         "checkpoint")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, vocab, itos, meta = prepare(args.corpus, seed=args.seed,
                                    max_len=args.max_len)

    if args.shuffle_tokens:
        rng = random.Random(args.seed)
        for split in ds.values():
            for r in split.recs:
                t = r["toks"].split(" "); rng.shuffle(t)
                r["toks"] = " ".join(t)

    labels = meta["labels"]
    print(json.dumps({k: meta[k] for k in
                      ("labels", "vocab_size", "split_sizes",
                       "scenarios_per_split", "class_counts")}, indent=2))

    loaders = {
        k: DataLoader(
            v, collate_fn=collate, num_workers=0,
            batch_sampler=BucketBatchSampler(
                seq_lengths(v), args.batch_size,
                shuffle=(k == "train"), seed=args.seed))
        for k, v in ds.items()}

    counts = torch.zeros(len(labels))
    for r in ds["train"].recs:
        counts[ds["train"].label_ix[r["cls"]]] += 1
    raw_weights = counts.sum() / (len(labels) * counts.clamp_min(1))
    weights = raw_weights.pow(args.weight_power).to(device)
    print(f"class weights (power={args.weight_power}):",
          {l: round(float(w), 3) for l, w in zip(labels, weights)})

    model = TriageLSTM(len(vocab), len(labels), n_task_feats=N_TASK_FEATS,
                       emb_dim=args.emb_dim, hidden=args.hidden,
                       layers=args.layers, dropout=args.dropout).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5,
                                                       patience=2)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(args.reports); rep_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("shuffled" if args.shuffle_tokens else "triage")
    ckpt = out_dir / f"{tag}_lstm.pth"

    best, bad, history = -1.0, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = n = 0
        for b in loaders["train"]:
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad()
            loss = crit(model(b), b["y"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b["y"]); n += len(b["y"])
        vf1, vper, vacc = evaluate(model, loaders["val"], device, len(labels))
        sched.step(vf1)
        history.append({"epoch": ep, "train_loss": tot / n,
                        "val_macro_f1": vf1, "val_acc": vacc,
                        "val_per_class_f1": dict(zip(labels, vper))})
        star = ""
        if vf1 > best:
            best, bad = vf1, 0
            torch.save(model.state_dict(), ckpt)
            star = "  *"
        else:
            bad += 1
        print(f"ep {ep:02d}/{args.epochs} loss {tot/n:.4f} "
              f"val_macroF1 {vf1:.4f} val_acc {vacc:.4f}{star}", flush=True)
        if bad >= args.patience:
            print(f"early stop at epoch {ep}")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    tf1, tper, tacc = evaluate(model, loaders["test"], device, len(labels))
    print(f"\nTEST macro-F1 {tf1:.4f}  acc {tacc:.4f}")
    for l, f in zip(labels, tper):
        print(f"  {l:<12} F1 {f:.4f}")

    config = {**vars(args), "device": device, "labels": labels,
              "vocab_size": len(vocab),
              "params": sum(p.numel() for p in model.parameters()),
              "best_val_macro_f1": best, "test_macro_f1": tf1,
              "test_acc": tacc, "train_seconds": round(time.time() - t0, 1),
              "meta": meta, "history": history}
    (rep_dir / f"{tag}_train.json").write_text(json.dumps(config, indent=2))
    (out_dir / "vocab.json").write_text(json.dumps(
        {"vocab": vocab, "labels": labels,
         "feat_mean": ds["train"].feat_mean.tolist(),
         "feat_std": ds["train"].feat_std.tolist(),
         "max_len": args.max_len,
         "model": {"emb_dim": args.emb_dim, "hidden": args.hidden,
                   "layers": args.layers, "dropout": args.dropout}}, indent=2))
    print(f"saved {ckpt} and {out_dir/'vocab.json'}")


if __name__ == "__main__":
    main()
