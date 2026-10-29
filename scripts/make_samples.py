"""
Generate the sample inputs under samples/ used by the CI gate, the docs and the
demo. Nothing here needs a live system:

  jaeger_hdfs_<class>.json   REAL held-out test traces re-exported in Jaeger's
                             JSON shape (in-distribution: the model understands
                             these operation names)
  jaeger_foreign_checkout.json  a hand-built microservice trace (the shape a
                             TrainTicket / HotROD trace has) - OUT of the
                             training distribution, so the gate must abstain
  canary_baseline.json / canary_bad.json / canary_good.json
                             synthetic canary comparison sets (200 / 60 / 60 traces)
  advisories_example.json    example enrichment file for the security gate

    python scripts/make_samples.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.observability.jaeger_adapter import load_jaeger   # noqa: E402

OUT = ROOT / "samples"
HR_SERVICE = {"0": "namenode", "1": "datanode01", "2": "client01", "3": "other"}
PROCS = {c: {"serviceName": s} for c, s in HR_SERVICE.items()}
BASE_US = 1_700_000_000_000_000


AG_NAME = ["Namenode", "Datanode", "RPC Client", "DFSClient", "User", "OTH"]


def record_to_jaeger(rec, rich=True) -> dict:
    """Corpus record -> Jaeger trace.

    rich=True adds what only an exporter that knows the training schema could
    supply: the agent of each span (`triage.agent` tag) and the task-level
    counts (`triage_extra`). rich=False is what a stock tracer would emit - used
    by scripts/adapter_fidelity.py to measure what that costs.

    Children share their parent's start time so service clocks are consistent
    and the alignment step is a no-op."""
    toks = rec["toks"].split(" ")
    durs = [int(x) for x in rec["dur"].split(",")]
    deps = [int(x) for x in rec["dep"].split(",")]
    spans, stack = [], {}
    for i, (tok, dur, dep) in enumerate(zip(toks, durs, deps)):
        op, status = tok.rsplit("|", 1)
        parent = stack.get(dep - 1) if dep > 0 else None
        start = (spans[parent]["startTime"] if parent is not None
                 else BASE_US + i * 10)
        stack[dep] = i
        tags = []
        if rich:
            tags += [{"key": "triage.status", "value": status},
                     {"key": "triage.agent", "value": AG_NAME[int(rec["ag"][i])]}]
        if status not in ("OK", "USER"):
            tags.append({"key": "error", "value": True})
        spans.append({
            "traceID": rec["task"].lower(), "spanID": f"s{i}",
            "operationName": op, "startTime": start, "duration": dur,
            "references": ([{"refType": "CHILD_OF", "traceID": rec["task"].lower(),
                             "spanID": f"s{parent}"}] if parent is not None else []),
            "tags": tags, "logs": [], "processID": rec["hr"][i]})
    out = {"traceID": rec["task"].lower(), "spans": spans,
           "processes": {c: PROCS[c] for c in set(rec["hr"])}}
    if rich:
        f = rec["f"]
        out["triage_extra"] = {k: f[k] for k in ("n_tid", "n_host", "n_edge",
                                                 "n_rep", "lat")}
    return out


def foreign_trace(trace_id, error=False, dur=120_000):
    procs = {"p1": {"serviceName": "frontend"}, "p2": {"serviceName": "route"},
             "p3": {"serviceName": "driver"}}
    t = BASE_US
    spans = [
        {"traceID": trace_id, "spanID": "a", "operationName": "HTTP GET /dispatch",
         "references": [], "startTime": t, "duration": dur, "tags": [], "logs": [],
         "processID": "p1"},
        {"traceID": trace_id, "spanID": "b", "operationName": "FindDriver",
         "references": [{"refType": "CHILD_OF", "traceID": trace_id, "spanID": "a"}],
         "startTime": t + 1000, "duration": dur // 2, "tags": [], "logs": [],
         "processID": "p3"},
        {"traceID": trace_id, "spanID": "c", "operationName": "Route.ComputeETA",
         "references": [{"refType": "CHILD_OF", "traceID": trace_id, "spanID": "a"}],
         "startTime": t + 2000, "duration": dur // 3,
         "tags": ([{"key": "error", "value": True},
                   {"key": "http.status_code", "value": 503}] if error else []),
         "logs": [], "processID": "p2"},
    ]
    return {"traceID": trace_id, "spans": spans, "processes": procs}


def main():
    OUT.mkdir(exist_ok=True)
    from src.data.dataset import prepare
    from src.models.predict import TriagePredictor
    predictor = TriagePredictor()
    ds, *_ = prepare(str(ROOT / "data" / "processed" / "corpus.jsonl.gz"),
                     max_len=predictor.max_len)
    rng = random.Random(7)

    for cls in ("PASS", "DATA_FAULT", "ENV_FAULT", "INFRA_FLAKE"):
        pool = [r for r in ds["test"].recs if r["cls"] == cls and 8 <= r["toks"].count(" ") < 200]
        rng.shuffle(pool)
        for rec in pool[:400]:
            doc = record_to_jaeger(rec)
            _tid, spans, extra = next(iter(load_jaeger({"data": [doc]})))
            via_adapter = predictor.predict(spans=spans, extra=extra)
            # pick by what the ADAPTER path says, not the corpus path: the
            # sample is there to show the Jaeger -> agent route working
            if (via_adapter["classification"] == cls
                    and via_adapter["confidence"] >= 0.6):
                (OUT / f"jaeger_hdfs_{cls.lower()}.json").write_text(
                    json.dumps({"data": [doc]}, indent=1))
                print(f"wrote jaeger_hdfs_{cls.lower()}.json  (task {rec['task']}, "
                      f"adapter path says {cls} @{via_adapter['confidence']:.2f})")
                break
        else:
            print(f"!! no {cls} sample classified correctly through the adapter "
                  "in 400 candidates - not written")

    (OUT / "jaeger_foreign_checkout.json").write_text(
        json.dumps({"data": [foreign_trace("foreign1", error=True)]}, indent=1))

    base = [foreign_trace(f"b{i}", error=(i % 100 == 0)) for i in range(200)]
    good = [foreign_trace(f"g{i}", error=(i % 60 == 0)) for i in range(60)]
    bad = [foreign_trace(f"x{i}", error=(i % 3 == 0), dur=400_000) for i in range(60)]
    for name, traces in (("canary_baseline", base), ("canary_good", good),
                         ("canary_bad", bad)):
        (OUT / f"{name}.json").write_text(json.dumps({"data": traces}))
    print("wrote canary_baseline.json / canary_good.json / canary_bad.json")

    (OUT / "advisories_example.json").write_text(json.dumps({
        "_comment": "keys are a vuln id or any alias; symbols are the vulnerable "
                    "functions as dotted names. pip-audit reports neither "
                    "severity nor symbols - fill these in from the advisory text "
                    "to let the gate narrow 'imported' down to 'actually called'.",
        "CVE-0000-00000": {"severity": "high", "symbols": ["yaml.load"]}}, indent=1))
    print("wrote advisories_example.json")


if __name__ == "__main__":
    main()
