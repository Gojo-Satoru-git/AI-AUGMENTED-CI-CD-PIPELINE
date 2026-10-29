"""
S5: Jaeger / OpenTelemetry adapter.

Turns a Jaeger trace (the JSON shape served by `GET /api/traces/<id>` or saved
from the Jaeger UI's "JSON" download) into the span-dict schema the triage API
already accepts:

    {op, status, duration, host, agent, depth}

so a live trace can flow: app -> Jaeger -> this adapter -> /api/v1/triage.

Two things are carried over from the HDFS pipeline (src/data/build_sequences.py):

1. CLOCK SKEW. Services in a microservice system have their own clocks, exactly
   like HDFS hosts do. `align_service_clocks` is the Jaeger-shaped twin of
   `align_clocks`: walk the span tree from the root and, the first time a call
   crosses into a new service, shift that service so its span starts inside its
   parent's window. If the clocks already agree, the offset is 0.
2. THE DISTRIBUTION GAP. The triage model was trained on HDFS operation names.
   A TrainTicket/HotROD trace has different operation names, so most tokens are
   out-of-vocabulary. The predictor already reports `oov_rate`; `drift_check`
   below turns that into an explicit "do not trust this classification" flag
   that the release gate (src/policy_engine/gate.py) routes to a human. We do
   NOT pretend an HDFS-trained model understands another system.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

OOV_ABSTAIN_THRESHOLD = 0.50


# ----------------------------------------------------------------------------
def _tags(span) -> dict:
    return {t.get("key"): t.get("value") for t in span.get("tags", []) or []}


def span_status(span) -> str:
    """Map Jaeger/OTel error signals onto the model's status vocabulary
    (OK, IOEXC, EXC, ERR, USER, OTH)."""
    tags = _tags(span)
    explicit = tags.get("triage.status")      # lets instrumented apps be exact
    if explicit in ("OK", "IOEXC", "EXC", "ERR", "USER", "OTH"):
        return explicit
    blob = " ".join(str(v).lower() for v in tags.values()
                    if isinstance(v, (str, bool)))
    for log in span.get("logs", []) or []:
        for f in log.get("fields", []) or []:
            blob += " " + str(f.get("value", "")).lower()
            if f.get("key") == "event" and str(f.get("value")).lower() in (
                    "error", "exception"):
                blob += " exception"

    is_error = (str(tags.get("error")).lower() == "true"
                or str(tags.get("otel.status_code", "")).upper() == "ERROR")
    code = tags.get("http.status_code") or tags.get("http.response.status_code")
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = None

    if is_error or (code is not None and code >= 500):
        if any(w in blob for w in ("timeout", "timed out", "connection",
                                   "ioexception", "refused", "reset")):
            return "IOEXC"
        return "ERR"
    if code is not None and code >= 400:
        return "EXC"
    if "exception" in blob:
        return "EXC"
    return "OK"


def _parent_of(span):
    for ref in span.get("references", []) or []:
        if ref.get("refType") in ("CHILD_OF", "FOLLOWS_FROM"):
            return ref.get("spanID")
    return None


def align_service_clocks(spans, processes):
    """Return {service: offset_us} putting every service on the root's clock.

    First-reach rule, mirroring build_sequences.align_clocks: when the span
    tree first crosses into a service, if the child already starts inside its
    parent's [start, end] window the clocks agree (offset 0); otherwise shift
    the service so the child starts at the parent's start."""
    by_id = {s["spanID"]: s for s in spans}
    service = {s["spanID"]: processes.get(s.get("processID"), {}).get(
        "serviceName", s.get("processID", "?")) for s in spans}
    kids = collections.defaultdict(list)
    roots = []
    for s in spans:
        p = _parent_of(s)
        if p and p in by_id:
            kids[p].append(s["spanID"])
        else:
            roots.append(s["spanID"])

    offset = {}
    for r in roots:
        offset.setdefault(service[r], 0)
    queue = collections.deque(roots)
    seen = set()
    while queue:
        pid = queue.popleft()
        if pid in seen:
            continue
        seen.add(pid)
        p = by_id[pid]
        p_start = p["startTime"] + offset.get(service[pid], 0)
        p_end = p_start + p.get("duration", 0)
        for cid in kids.get(pid, ()):
            c = by_id[cid]
            sc = service[cid]
            if sc not in offset:
                if p_start <= c["startTime"] <= p_end:
                    offset[sc] = 0
                else:
                    offset[sc] = p_start - c["startTime"]
            queue.append(cid)
    for s in spans:                      # unreachable services: leave as-is
        offset.setdefault(service[s["spanID"]], 0)
    return offset


def jaeger_to_spans(trace: dict):
    """One Jaeger trace object -> (spans, extra).

    `extra` carries the task-level facts the model's scalar features use
    (span count, hosts, edges, latency) so predictions use real values instead
    of values guessed from the span list."""
    raw = trace.get("spans", [])
    if not raw:
        raise ValueError("trace has no spans")
    processes = trace.get("processes", {})
    offset = align_service_clocks(raw, processes)

    by_id = {s["spanID"]: s for s in raw}
    depth = {}

    def depth_of(sid):
        if sid in depth:
            return depth[sid]
        p = _parent_of(by_id[sid])
        depth[sid] = 0 if (not p or p not in by_id) else depth_of(p) + 1
        return depth[sid]

    rows = []
    for s in raw:
        svc = processes.get(s.get("processID"), {}).get("serviceName", "?")
        start = s["startTime"] + offset.get(svc, 0)
        rows.append((start, depth_of(s["spanID"]), {
            "op": s.get("operationName", "?"),
            "status": span_status(s),
            "duration": int(s.get("duration", 0)),
            "host": svc,
            "agent": _tags(s).get("triage.agent", svc),
            "depth": depth_of(s["spanID"]),
        }))
    rows.sort(key=lambda r: (r[0], r[1]))
    spans = [r[2] for r in rows]

    starts = [r[0] for r in rows]
    ends = [r[0] + r[2]["duration"] for r in rows]
    extra = {
        "n_tid": len(raw),
        "n_host": len({sp["host"] for sp in spans}),
        "n_edge": sum(1 for s in raw if _parent_of(s) in by_id),
        "n_rep": len(raw),
        "lat": max(ends) - min(starts),
    }
    # An exporter that knows the training-time meaning of these task-level
    # counts (distinct threads, distinct hosts, reports) can supply them; a
    # stock Jaeger trace cannot, and gets the span/service-based approximation.
    hints = trace.get("triage_extra")
    extra.update(hints or {})
    extra["schema_attested"] = bool(hints)
    extra["from_adapter"] = True
    return spans, extra


def load_jaeger(path_or_obj):
    """Yield (trace_id, spans, extra) for every trace in a Jaeger JSON export."""
    if isinstance(path_or_obj, (str, Path)):
        doc = json.loads(Path(path_or_obj).read_text(encoding="utf-8"))
    else:
        doc = path_or_obj
    traces = doc.get("data", doc if isinstance(doc, list) else [doc])
    for t in traces:
        spans, extra = jaeger_to_spans(t)
        yield t.get("traceID", "unknown"), spans, extra


def drift_check(pred: dict, threshold: float = OOV_ABSTAIN_THRESHOLD,
                schema_attested: bool = True) -> dict:
    """Decide whether the model can be trusted on this trace at all.

    Two independent conditions, both required:

    1. VOCABULARY - the operations must be ones the model was trained on
       (OOV rate). Catches traces from a different system.
    2. SCHEMA - the task-level features (distinct threads/hosts, report count,
       latency) must follow the training definitions. A stock tracer cannot
       supply those, and the vocabulary check does not catch the problem:
       measured with scripts/adapter_fidelity.py, stock Jaeger exports of the
       model's OWN training traces (operation names known; typical OOV ~0.4,
       under the 0.5 limit) are classified correctly only ~20% of the time -
       90% come back REAL_BUG, because tiny latency and host counts resemble the
       short bug workloads. Only traces whose exporter attests the training
       schema (`triage_extra`) are trusted."""
    oov = pred.get("oov_rate", 0.0)
    vocab_ok = oov <= threshold
    if not vocab_ok:
        note = (f"{oov:.0%} of operations are unseen in training (HDFS "
                "vocabulary) - classification is not trustworthy; route to a "
                "human or retrain on this system's traces")
    elif not schema_attested:
        note = ("task-level features cannot be computed with the training "
                "schema from this trace (stock tracer export); measured "
                "accuracy on such exports is ~20% - route to a human or supply "
                "training-schema counts via `triage_extra`")
    else:
        note = "trace vocabulary and feature schema match the training system"
    return {"oov_rate": oov, "in_distribution": vocab_ok and schema_attested,
            "vocabulary_ok": vocab_ok, "schema_attested": schema_attested,
            "note": note}
