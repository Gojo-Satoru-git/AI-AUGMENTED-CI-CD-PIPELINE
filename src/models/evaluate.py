"""
Evaluate the Test-Triage Agent against baselines that actually test something.

Phase 1 reported a single accuracy number (88.18%) that turned out to equal a
one-feature rule. Every number here is therefore reported per class, on a
scenario-grouped test split, next to baselines chosen to falsify a specific
claim:

  B1 majority          - is the class prior already enough?
  B2 task scalars      - do you need the sequence at all, or just trace shape?
  B3 bag-of-ops counts - this IS Phase 1's representation (event counts),
                         built correctly. Beating it is the burden of proof.
  B4 bag-of-ops TF-IDF - the strongest order-free text baseline.

If the model cannot beat B3 and B4, sequence order does not carry signal on
this dataset and we should say so rather than ship an LSTM for appearances.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import (collate, prepare, task_features,  # noqa: E402
                              N_TASK_FEATS)
from src.features.triage_lstm import TriageLSTM                 # noqa: E402


# ---------------------------------------------------------------------------
def metrics(y_true, y_pred, labels):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    K = len(labels)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per, f1s = {}, []
    for c, name in enumerate(labels):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        per[name] = {"precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(f1, 4), "support": int(cm[c, :].sum())}
    sup = cm.sum(1)
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "weighted_f1": round(float((np.array(f1s) * sup).sum() / sup.sum()), 4),
        "per_class": per,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def bag_of_ops(recs, order_free=True):
    """Order-free document per trace - Phase 1's representation, done right."""
    return [" ".join(r["toks"].split(" ")) if not order_free
            else " ".join(sorted(r["toks"].split(" "))) for r in recs]


def run_baselines(ds, labels, out_dir):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.svm import LinearSVC

    ytr = np.array([ds["train"].label_ix[r["cls"]] for r in ds["train"].recs])
    yte = np.array([ds["test"].label_ix[r["cls"]] for r in ds["test"].recs])
    results = {}

    # B1 - majority class
    maj = int(np.bincount(ytr).argmax())
    results["B1_majority"] = metrics(yte, np.full_like(yte, maj), labels)

    # B2 - task-level scalars only, no sequence
    Xtr = np.array([task_features(r) for r in ds["train"].recs])
    Xte = np.array([task_features(r) for r in ds["test"].recs])
    gb = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    gb.fit(Xtr, ytr)
    results["B2_task_scalars_gbm"] = metrics(yte, gb.predict(Xte), labels)

    # B3 - bag-of-ops COUNTS (Phase 1's event-count representation)
    cv = CountVectorizer(token_pattern=r"\S+", min_df=2)
    Ctr = cv.fit_transform(bag_of_ops(ds["train"].recs))
    Cte = cv.transform(bag_of_ops(ds["test"].recs))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(Ctr, ytr)
    results["B3_bag_of_ops_counts_lr"] = metrics(yte, lr.predict(Cte), labels)

    # B4 - bag-of-ops TF-IDF
    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True)
    Ttr = tv.fit_transform(bag_of_ops(ds["train"].recs))
    Tte = tv.transform(bag_of_ops(ds["test"].recs))
    sv = LinearSVC(class_weight="balanced")
    sv.fit(Ttr, ytr)
    results["B4_bag_of_ops_tfidf_svm"] = metrics(yte, sv.predict(Tte), labels)
    return results


@torch.no_grad()
def predict(model, ds, device, batch_size=256):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate)
    ys, ps, probs = [], [], []
    model.eval()
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(b)
        probs.append(torch.softmax(logits, -1).cpu())
        ps.append(logits.argmax(-1).cpu())
        ys.append(b["y"].cpu())
    return (torch.cat(ys).numpy(), torch.cat(ps).numpy(),
            torch.cat(probs).numpy())


def plot_confusion(cm, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm, dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]*100:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "#222")
    fig.colorbar(im, ax=ax, shrink=0.8, label="row-normalised")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl.gz")
    ap.add_argument("--ckpt", default="src/models/saved/triage_lstm.pth")
    ap.add_argument("--vocab", default="src/models/saved/vocab.json")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    rep = Path(args.reports); rep.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.vocab).read_text())
    labels = cfg["labels"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, vocab, itos, meta = prepare(args.corpus, seed=args.seed,
                                    max_len=cfg["max_len"])
    model = TriageLSTM(len(vocab), len(labels), n_task_feats=N_TASK_FEATS,
                       emb_dim=cfg["model"]["emb_dim"],
                       hidden=cfg["model"]["hidden"],
                       layers=cfg["model"]["layers"],
                       dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    y, p, probs = predict(model, ds["test"], device)
    model_metrics = metrics(y, p, labels)
    print(f"MODEL  macro-F1 {model_metrics['macro_f1']}  "
          f"acc {model_metrics['accuracy']}")

    out = {"model": model_metrics, "split_meta": {
        k: meta[k] for k in ("split_sizes", "scenarios_per_split",
                             "class_counts", "vocab_size")}}

    if not args.skip_baselines:
        out["baselines"] = run_baselines(ds, labels, rep)
        for name, m in out["baselines"].items():
            print(f"{name:<28} macro-F1 {m['macro_f1']:.4f}  "
                  f"acc {m['accuracy']:.4f}")

    # ---- operational view: what does the pipeline actually decide? ----------
    # A retry on a DATA_FAULT/REAL_BUG burns CI minutes for nothing; missing a
    # REAL_BUG ships a defect. Report those two error types explicitly.
    ix = {l: i for i, l in enumerate(labels)}
    retryable = {ix["INFRA_FLAKE"]}
    never_retry = {ix[l] for l in ("DATA_FAULT", "REAL_BUG") if l in ix}
    wrong_retry = int(sum((pi in retryable) and (yi in never_retry)
                          for yi, pi in zip(y, p)))
    missed_bug = int(sum((yi == ix["REAL_BUG"]) and (pi != ix["REAL_BUG"])
                         for yi, pi in zip(y, p))) if "REAL_BUG" in ix else 0
    out["operational"] = {
        "test_traces": int(len(y)),
        "wrongly_retried": wrong_retry,
        "wrongly_retried_rate": round(wrong_retry / len(y), 5),
        "real_bugs_missed": missed_bug,
        "real_bug_recall": model_metrics["per_class"]
                           .get("REAL_BUG", {}).get("recall"),
    }
    print(f"\noperational: wrongly retried {wrong_retry}/{len(y)}  "
          f"REAL_BUG missed {missed_bug}")

    # ---- COMPOUND generalisation probe (never trained on) ------------------
    ds_c, _, _, _ = prepare(args.corpus, seed=args.seed,
                            max_len=cfg["max_len"], include_compound=True)
    comp = [r for split in ds_c.values() for r in split.recs
            if r["cls"] == "COMPOUND"]
    if comp:
        from src.data.dataset import TraceDataset
        # COMPOUND has no true index among the 5 trained classes; map it to 0
        # purely so the Dataset can build - only the PREDICTIONS are read.
        probe_ix = dict(ds["train"].label_ix)
        probe_ix["COMPOUND"] = 0
        probe = TraceDataset(comp, vocab, probe_ix, ds["train"].feat_mean,
                             ds["train"].feat_std, cfg["max_len"])
        _, pc, _ = predict(model, probe, device)
        dist = {labels[i]: int((pc == i).sum()) for i in range(len(labels))}
        out["compound_probe"] = {
            "n": len(comp), "predicted_distribution": dist,
            "note": "COMPOUND scenarios inject multiple simultaneous faults "
                    "and were never trained on. A sane model should route "
                    "them to fault classes, not PASS."}
        print(f"COMPOUND probe (n={len(comp)}): {dist}")

    plot_confusion(model_metrics["confusion_matrix"], labels,
                   rep / "confusion_matrix.png",
                   "Test-Triage Agent - test set (scenario-grouped)")
    (rep / "evaluation.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {rep/'evaluation.json'} and {rep/'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
