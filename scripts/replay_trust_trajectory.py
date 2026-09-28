"""
S3 exit-criterion demo: replay the held-out test set through the trust-tier
state machine and show the agent climb T0 -> T3, then demonstrate a
demotion.

Two phases:
  1. HONEST STREAM - every test-set trace, in the model's actual predicted
     order, scored against its real label. This is what naturally happens in
     production: right answers accumulate, the Wilson lower-bound clears each
     tier's threshold, and the agent is promoted (see
     src/policy_engine/database.py:evaluate_promotion).
  2. INJECTED FAULT - a handful of synthetic decisions where the agent is
     shown a REAL_BUG trace, misclassifies it, and acts on that
     misclassification autonomously (simulating what a Chaos-Engineering
     experiment, S7, would provoke). This is clearly a synthetic stress
     segment, not a claim that the model failed organically - REAL_BUG had
     0 misses on the real test set (F1 1.0, see reports/evaluation.json), so
     demonstrating the demotion path at all requires forcing the case.

Writes reports/trust_trajectory.json (tier at every step) and
reports/trust_trajectory.png (plot), using a dedicated agent_id so it never
touches the real "triage_agent" row in trust_tiers.db.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import N_TASK_FEATS, prepare              # noqa: E402
from src.features.triage_lstm import TriageLSTM                 # noqa: E402
from src.models.evaluate import predict as batch_predict        # noqa: E402
from src.policy_engine import database as db                    # noqa: E402
from src.policy_engine import policy as P                       # noqa: E402

AGENT_ID = "replay_demo_agent"


def run_stream(agent_id, steps, out):
    """steps: iterable of (classification, confidence, ground_truth,
    label_for_plot). Logs each through the real decide()+log_decision()+
    record_ground_truth() path so promotion/demotion fire exactly as the API
    would trigger them."""
    for i, (cls, conf, truth, note) in enumerate(steps):
        tier_before = db.get_agent_tier(agent_id)
        d = P.decide(cls, conf, tier_before)
        did = db.log_decision(agent_id, f"replay-{i}", d.classification,
                              d.confidence, d.action, d.rule, d.trust_tier,
                              d.autonomous)
        result = db.record_ground_truth(did, truth)
        tier_after = db.get_agent_tier(agent_id)
        out.append({
            "step": i, "note": note, "predicted": cls, "truth": truth,
            "correct": cls == truth, "tier_before": tier_before,
            "tier_after": tier_after, "tier_action": result["tier_action"],
        })
        if tier_after != tier_before:
            print(f"  step {i:>5} [{note}] tier {tier_before} -> "
                  f"{tier_after}  ({result.get('reason', '')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl.gz")
    ap.add_argument("--ckpt", default="src/models/saved/triage_lstm.pth")
    ap.add_argument("--vocab", default="src/models/saved/vocab.json")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-fault-injections", type=int, default=5,
                    help="synthetic missed-REAL_BUG decisions to force a "
                         "demotion demo (see module docstring, phase 2)")
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
    y, p, probs = batch_predict(model, ds["test"], device)

    db.init_db()
    # start clean: wipe any prior replay for this agent_id
    with db._conn() as c:
        c.execute("DELETE FROM decisions WHERE agent_id=?", (AGENT_ID,))
        c.execute("DELETE FROM tier_history WHERE agent_id=?", (AGENT_ID,))
        c.execute("DELETE FROM agent_trust WHERE agent_id=?", (AGENT_ID,))
    db.set_agent_tier(AGENT_ID, 0, "replay start")
    with db._conn() as c:  # set_agent_tier logs a 0->0 no-op transition; clear it
        c.execute("DELETE FROM tier_history WHERE agent_id=?", (AGENT_ID,))

    trace = []
    print(f"Phase 1: honest stream over {len(y)} real test-set decisions")
    honest_steps = [
        (labels[pi], float(probs[i, pi]), labels[yi], "honest")
        for i, (yi, pi) in enumerate(zip(y, p))
    ]
    run_stream(AGENT_ID, honest_steps, trace)
    tier_after_honest = db.get_agent_tier(AGENT_ID)
    print(f"  tier after honest stream: T{tier_after_honest}")

    print(f"\nPhase 2: {args.n_fault_injections} injected missed-REAL_BUG "
          f"decisions (simulated chaos fault, see docstring)")
    fault_steps = [("PASS", 0.75, "REAL_BUG", "injected-fault")
                  for _ in range(args.n_fault_injections)]
    run_stream(AGENT_ID, fault_steps, trace)
    tier_after_fault = db.get_agent_tier(AGENT_ID)
    print(f"  tier after fault injection: T{tier_after_fault}")

    (rep / "trust_trajectory.json").write_text(json.dumps({
        "agent_id": AGENT_ID, "tier_after_honest_stream": tier_after_honest,
        "tier_after_fault_injection": tier_after_fault, "steps": trace,
    }, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tiers = [s["tier_after"] for s in trace]
    fault_start = len(honest_steps)
    tier_names = ["T0 observe", "T1 recommend", "T2 rollback", "T3 autonomous"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 1]})

    ax1.step(range(len(tiers)), tiers, where="post", color="#1e824c")
    ax1.axvline(fault_start, color="#c0392b", linestyle="--",
               label="synthetic fault injection starts")
    ax1.set_yticks([0, 1, 2, 3]); ax1.set_yticklabels(tier_names)
    ax1.set_xlabel("decision #")
    ax1.set_title("Full replay: honest test stream, T0 -> T3")
    ax1.legend(loc="lower right")

    zoom_start = max(0, fault_start - 20)
    xs = range(zoom_start, len(tiers))
    ys = tiers[zoom_start:]
    ax2.step(xs, ys, where="post", color="#1e824c", marker="o", markersize=3)
    ax2.axvline(fault_start, color="#c0392b", linestyle="--")
    demote_n = 0
    for i, s in enumerate(trace[zoom_start:], start=zoom_start):
        if s["tier_action"] == "demoted":
            demote_n += 1
            dy = -14 if demote_n % 2 else -26
            ax2.annotate("demoted", (i, s["tier_after"]),
                        textcoords="offset points", xytext=(0, dy),
                        fontsize=8, color="#c0392b", ha="center")
    ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(tier_names)
    ax2.set_xlabel("decision #")
    ax2.set_title("Zoom: fault-injection segment")
    fig.tight_layout()
    fig.savefig(rep / "trust_trajectory.png", dpi=150)
    print(f"\nwrote {rep/'trust_trajectory.json'} and "
          f"{rep/'trust_trajectory.png'}")


if __name__ == "__main__":
    main()
