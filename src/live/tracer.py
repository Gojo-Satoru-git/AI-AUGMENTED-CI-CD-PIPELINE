"""
A tiny, dependency-free tracer for the minimal live system (src/live/minishop.py).

It records real wall-clock spans and can emit them in the two shapes the rest
of the project cares about:

    to_jaeger()  Jaeger's JSON trace shape - exactly what
                 src/observability/jaeger_adapter.py ingests, so the live system
                 feeds the agent through the same door a real Jaeger would.
    to_otlp()    OTLP/HTTP JSON, which Jaeger's collector (port 4318) accepts, so
                 the same traces show up in the Jaeger UI.

Why not the OpenTelemetry SDK? A dependency-free tracer keeps the live demo
runnable anywhere and makes every field testable here. NOTE: the OTLP output is
checked structurally by the tests but has NOT been sent to a real Jaeger from
the machine this was written on - if the push fails, the file/in-process path
still works (see push_otlp, which reports rather than raises).
"""
from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.error
import urllib.request

ERROR_KINDS = ("timeout", "refused", "checksum", "exception")


class SpanHandle:
    def __init__(self, rec):
        self._rec = rec

    @property
    def span_id(self):
        return self._rec["spanID"]

    def tag(self, key, value):
        self._rec["tags"].append({"key": key, "value": value})

    def fail(self, message, **tags):
        """Mark the span errored; the message's wording is what the Jaeger
        adapter reads to tell a timeout/connection failure from a plain error."""
        self.tag("error", True)
        self.tag("error.message", message)
        for k, v in tags.items():
            self.tag(k, v)


class Trace:
    def __init__(self, trace_id, rng):
        self.trace_id = trace_id
        self._rng = rng
        self.spans = []
        self.services = {}          # service name -> process id
        self.fault_fired = None     # set by the app: class of the fault that bit

    def _id(self, nbytes):
        return "".join(f"{self._rng.randrange(256):02x}" for _ in range(nbytes))

    @contextlib.contextmanager
    def span(self, service, op, parent=None, **tags):
        pid = self.services.setdefault(service, f"p{len(self.services) + 1}")
        rec = {"traceID": self.trace_id, "spanID": self._id(8),
               "operationName": op, "processID": pid, "logs": [],
               "references": ([{"refType": "CHILD_OF", "traceID": self.trace_id,
                                "spanID": parent.span_id}] if parent else []),
               "tags": [{"key": k, "value": v} for k, v in tags.items()]}
        handle = SpanHandle(rec)
        start = time.time_ns()
        rec["startTime"] = start // 1000
        try:
            yield handle
        finally:
            rec["duration"] = max((time.time_ns() - start) // 1000, 1)
            self.spans.append(rec)

    # ---------------------------------------------------------------- outputs
    def to_jaeger(self) -> dict:
        return {"traceID": self.trace_id,
                "spans": sorted(self.spans, key=lambda s: s["startTime"]),
                "processes": {pid: {"serviceName": svc, "tags": []}
                              for svc, pid in self.services.items()}}

    def to_otlp(self) -> dict:
        by_service = {}
        pid_to_svc = {pid: svc for svc, pid in self.services.items()}
        for s in self.spans:
            by_service.setdefault(pid_to_svc[s["processID"]], []).append(s)

        def attr(k, v):
            if isinstance(v, bool):
                return {"key": k, "value": {"boolValue": v}}
            if isinstance(v, int):
                return {"key": k, "value": {"intValue": str(v)}}
            return {"key": k, "value": {"stringValue": str(v)}}

        resource_spans = []
        for svc, spans in by_service.items():
            out = []
            for s in spans:
                errored = any(t["key"] == "error" and t["value"] is True
                              for t in s["tags"])
                msg = next((t["value"] for t in s["tags"]
                            if t["key"] == "error.message"), "")
                start_ns = s["startTime"] * 1000
                out.append({
                    "traceId": self.trace_id, "spanId": s["spanID"],
                    "parentSpanId": (s["references"][0]["spanID"]
                                     if s["references"] else ""),
                    "name": s["operationName"], "kind": 1,
                    "startTimeUnixNano": str(start_ns),
                    "endTimeUnixNano": str(start_ns + s["duration"] * 1000),
                    "attributes": [attr(t["key"], t["value"]) for t in s["tags"]],
                    "status": ({"code": 2, "message": str(msg)} if errored
                               else {"code": 1}),
                })
            resource_spans.append({
                "resource": {"attributes": [attr("service.name", svc)]},
                "scopeSpans": [{"scope": {"name": "minishop"}, "spans": out}]})
        return {"resourceSpans": resource_spans}


def push_otlp(url: str, trace: Trace, timeout: float = 3.0):
    """POST one trace to an OTLP/HTTP endpoint (Jaeger: http://127.0.0.1:4318/v1/traces).
    Returns None on success, or an error string - it never raises, so a missing
    Jaeger cannot take the live loop down."""
    req = urllib.request.Request(
        url, data=json.dumps(trace.to_otlp()).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return None if 200 <= r.status < 300 else f"HTTP {r.status}"
    except (urllib.error.URLError, OSError) as e:
        return str(getattr(e, "reason", e))
