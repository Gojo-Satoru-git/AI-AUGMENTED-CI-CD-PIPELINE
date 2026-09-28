"""
S4: root-cause span localization, replacing the Phase-1
policy_engine/graph_generator.py defect (hardcoded to one scenario folder,
drew a spring-layout picture, never actually ranked anything - see
PHASE2_PLAN.md's "HIGH" defect list).

Method (the one specced in PHASE2_PLAN.md S4, minus the subtree-impact term
- see the note in `score_spans` for why that term is dropped here):

  1. From PASS-class traces only, learn a per-operation latency baseline:
     mean and stdev of duration for every distinct OpName.
  2. For a failing trace, score every span by how surprising it is against
     that baseline: a latency z-score (how much slower than normal PASS
     traffic doing the same op) plus a flat bonus if the span's own status
     was non-OK.
  3. Rank spans by that score, descending. The top of the list is the
     model's best guess at "where the fault actually is."

This reuses the corpus already built by build_sequences.py (toks/dur/hr/ag
in lock-step arrays) instead of re-parsing raw TraceBench CSVs, so building
the baseline is a single pass over corpus.jsonl.gz rather than 363 folders
of event.csv.
"""
from __future__ import annotations

import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path

NON_OK_STATUSES = None  # populated below from status suffix, see _status_of


def _op_and_status(tok: str) -> tuple[str, str]:
    op, _, status = tok.rpartition("|")
    return op, status


# ----------------------------------------------------------------------------
# Baseline: per-op latency distribution over PASS-class traces
# ----------------------------------------------------------------------------
def build_baseline(corpus_path: str | Path) -> dict:
    """Returns {op: {"mean": float, "std": float, "n": int}} plus a
    "__global__" fallback entry for ops never seen in a PASS trace (e.g. an
    op that only ever appears mid-fault, such as a retry handler)."""
    sums = defaultdict(lambda: [0.0, 0.0, 0])  # [sum, sumsq, n]
    g_sum = g_sumsq = g_n = 0.0

    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["cls"] != "PASS":
                continue
            toks = r["toks"].split(" ")
            durs = [int(x) for x in r["dur"].split(",")]
            for tok, d in zip(toks, durs):
                op, _ = _op_and_status(tok)
                s = sums[op]
                s[0] += d; s[1] += d * d; s[2] += 1
                g_sum += d; g_sumsq += d * d; g_n += 1

    baseline = {}
    for op, (s, sq, n) in sums.items():
        mean = s / n
        var = max(sq / n - mean * mean, 0.0)
        baseline[op] = {"mean": mean, "std": math.sqrt(var) or 1.0, "n": n}
    g_mean = g_sum / g_n if g_n else 0.0
    g_var = max(g_sumsq / g_n - g_mean * g_mean, 0.0) if g_n else 1.0
    baseline["__global__"] = {"mean": g_mean, "std": math.sqrt(g_var) or 1.0,
                              "n": int(g_n)}
    return baseline


def save_baseline(baseline: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(baseline, indent=2))


def load_baseline(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------
# Per-trace suspect scoring
# ----------------------------------------------------------------------------
ERROR_BONUS = 2.0   # flat score added to any span whose own status != OK/USER
Z_CLIP = 0.0        # only "slower than normal" counts as suspicious


def score_spans(record: dict, baseline: dict) -> list[dict]:
    """Rank every span in one corpus record by suspicion. Does NOT use
    subtree/descendant impact (PHASE2_PLAN's third term) - the built corpus
    keeps per-span duration/host/depth but not the edge list, so "how many
    descendants does this span dominate" isn't reconstructable without
    re-reading edge.csv per task. Depth is used as a much weaker proxy
    instead: a fault near the root of the call tree explains more of the
    trace than one in a leaf, all else equal."""
    toks = record["toks"].split(" ")
    durs = [int(x) for x in record["dur"].split(",")]
    deps = [int(x) for x in record["dep"].split(",")]
    g = baseline["__global__"]

    scored = []
    for i, (tok, d, dep) in enumerate(zip(toks, durs, deps)):
        op, status = _op_and_status(tok)
        b = baseline.get(op, g)
        z = (d - b["mean"]) / b["std"]
        z = max(z, Z_CLIP)
        err = status not in ("OK", "USER")
        depth_bonus = 1.0 / (1 + dep)  # root-ish spans score slightly higher
        score = z + (ERROR_BONUS if err else 0.0) + 0.25 * depth_bonus
        scored.append({"index": i, "token": tok, "op": op, "status": status,
                       "duration": d, "depth": dep, "z_score": round(z, 3),
                       "error": err, "score": round(score, 3)})
    scored.sort(key=lambda s: s["score"], reverse=True)
    return scored


def top_k(record: dict, baseline: dict, k: int = 5) -> list[dict]:
    return score_spans(record, baseline)[:k]


# ----------------------------------------------------------------------------
# Postmortem generation
# ----------------------------------------------------------------------------
def generate_postmortem(record: dict, baseline: dict, classification: str,
                        action: str, top_n: int = 5) -> str:
    suspects = top_k(record, baseline, top_n)
    lines = [
        f"# Postmortem: task `{record['task']}`",
        "",
        f"**Classification:** {classification}  **Action taken:** {action}",
        f"**Scenario:** `{record['scen']}`  **Spans:** {len(record['toks'].split(' '))}",
        "",
        "## Top suspect spans",
        "",
        "| rank | op | status | duration (us) | z-score | depth | score |",
        "|---|---|---|---|---|---|---|",
    ]
    for rank, s in enumerate(suspects, 1):
        flag = " ⚠" if s["error"] else ""
        lines.append(f"| {rank} | `{s['op']}`{flag} | {s['status']} | "
                     f"{s['duration']} | {s['z_score']} | {s['depth']} | "
                     f"{s['score']} |")
    top1 = suspects[0]
    why = (f"status {top1['status']}" if top1["error"]
          else f"z-score {top1['z_score']}")
    lines += [
        "",
        "## Summary",
        f"The highest-ranked suspect is `{top1['op']}` ({why}), "
        f"{top1['duration']}us against the PASS-baseline for that "
        "operation. Investigate this span first.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Evaluation: top-k localization accuracy vs. two baselines
# ----------------------------------------------------------------------------
def _true_error_indices(record: dict) -> list[int]:
    toks = record["toks"].split(" ")
    return [i for i, t in enumerate(toks)
            if _op_and_status(t)[1] not in ("OK", "USER")]


def evaluate_localization(corpus_path: str | Path, baseline: dict,
                          ks=(1, 3, 5), seed=42, max_per_class=3000) -> dict:
    """Proxy-ground-truth evaluation: TraceBench does not label a "true root
    cause span" for a task, so the observable proxy used here is "a span
    whose own status is not OK/USER" - i.e. a span that itself recorded a
    failure. A trace with zero such spans (the fault only shows up as a
    global effect, no local error status) is skipped - there is nothing to
    localize against. This is a real limitation, stated plainly rather than
    hidden: it measures "can the ranking surface the failing span itself,"
    not "can it find the true upstream cause of that failure." """
    rng = random.Random(seed)
    per_class = defaultdict(lambda: {k: [0, 0] for k in ks})  # [hits, n]
    baselines = {
        "ours": lambda rec: [s["index"] for s in score_spans(rec, baseline)],
        "random": lambda rec: rng.sample(
            range(len(rec["toks"].split(" "))),
            len(rec["toks"].split(" "))),
        "slowest_raw": lambda rec: [
            i for i, _ in sorted(
                enumerate(int(x) for x in rec["dur"].split(",")),
                key=lambda p: p[1], reverse=True)],
    }
    results = {name: defaultdict(lambda: {k: [0, 0] for k in ks})
              for name in baselines}
    seen_by_class = defaultdict(int)

    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["cls"] in ("PASS",):
                continue
            truth = set(_true_error_indices(r))
            if not truth:
                continue
            if seen_by_class[r["cls"]] >= max_per_class:
                continue
            seen_by_class[r["cls"]] += 1
            for name, fn in baselines.items():
                order = fn(r)
                for k in ks:
                    hit = any(i in truth for i in order[:k])
                    results[name][r["cls"]][k][0] += int(hit)
                    results[name][r["cls"]][k][1] += 1

    out = {}
    for name, per_cls in results.items():
        out[name] = {
            cls: {f"top{k}": round(hits / n, 4) if n else None
                 for k, (hits, n) in ks_dict.items()}
            for cls, ks_dict in per_cls.items()
        }
        n_total = sum(v[ks[0]][1] for v in per_cls.values())
        out[name]["__n_traces_scored__"] = n_total
    return out
