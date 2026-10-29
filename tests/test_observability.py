"""
Tests for S5: the Jaeger adapter, drift guard, canary health score, metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.observability import jaeger_adapter as J      # noqa: E402
from src.observability.canary import canary_health      # noqa: E402


def make_span(sid, op, svc_pid, start, dur, parent=None, tags=None, logs=None):
    refs = ([{"refType": "CHILD_OF", "traceID": "t1", "spanID": parent}]
            if parent else [])
    return {"traceID": "t1", "spanID": sid, "operationName": op,
            "references": refs, "startTime": start, "duration": dur,
            "tags": tags or [], "logs": logs or [], "processID": svc_pid}


PROCS = {"p1": {"serviceName": "frontend"}, "p2": {"serviceName": "orders"},
         "p3": {"serviceName": "db"}}


def simple_trace(**overrides):
    spans = [
        make_span("a", "GET /checkout", "p1", 1_000_000, 90_000),
        make_span("b", "createOrder", "p2", 1_010_000, 60_000, parent="a"),
        make_span("c", "SELECT orders", "p3", 1_020_000, 20_000, parent="b"),
    ]
    return {"traceID": "t1", "spans": spans, "processes": PROCS, **overrides}


# --------------------------------------------------------------------------
def test_adapter_produces_triage_span_schema():
    spans, extra = J.jaeger_to_spans(simple_trace())
    assert [s["op"] for s in spans] == ["GET /checkout", "createOrder",
                                        "SELECT orders"]
    for s in spans:
        assert set(s) == {"op", "status", "duration", "host", "agent", "depth"}
    assert [s["depth"] for s in spans] == [0, 1, 2]
    assert extra["n_host"] == 3 and extra["n_edge"] == 2


def test_error_tag_maps_to_err_and_timeout_to_ioexc():
    t = simple_trace()
    t["spans"][2]["tags"] = [{"key": "error", "value": True}]
    assert J.span_status(t["spans"][2]) == "ERR"
    t["spans"][2]["tags"] = [{"key": "error", "value": True},
                             {"key": "error.message", "value": "read timeout"}]
    assert J.span_status(t["spans"][2]) == "IOEXC"


def test_http_status_codes():
    s = make_span("x", "op", "p1", 0, 1, tags=[
        {"key": "http.status_code", "value": 503}])
    assert J.span_status(s) == "ERR"
    s = make_span("x", "op", "p1", 0, 1, tags=[
        {"key": "http.status_code", "value": 404}])
    assert J.span_status(s) == "EXC"
    s = make_span("x", "op", "p1", 0, 1, tags=[
        {"key": "http.status_code", "value": 200}])
    assert J.span_status(s) == "OK"


def test_clock_skew_is_corrected_across_services():
    """The orders service's clock runs 5 seconds ahead of the frontend's.
    Raw start times would put its span AFTER the whole request ended."""
    t = simple_trace()
    t["spans"][1]["startTime"] += 5_000_000
    t["spans"][2]["startTime"] += 5_000_000   # db shares orders' skewed clock
    off = J.align_service_clocks(t["spans"], t["processes"])
    assert off["frontend"] == 0
    # first-reach rule: the child is anchored at its parent's start, so the
    # orders service is shifted by (parent_start - child_raw_start)
    assert off["orders"] == 1_000_000 - 6_010_000
    assert off["db"] == 1_000_000 - 6_020_000
    spans, _ = J.jaeger_to_spans(t)
    # after alignment the call order is the true order: caller before callee
    assert [s["op"] for s in spans] == ["GET /checkout", "createOrder",
                                        "SELECT orders"]


def test_consistent_clocks_get_zero_offset():
    off = J.align_service_clocks(simple_trace()["spans"], PROCS)
    assert set(off.values()) == {0}


def test_load_jaeger_accepts_api_envelope():
    doc = {"data": [simple_trace(), simple_trace(traceID="t2")]}
    out = list(J.load_jaeger(doc))
    assert len(out) == 2 and out[0][0] == "t1"


def test_empty_trace_is_rejected():
    with pytest.raises(ValueError):
        J.jaeger_to_spans({"spans": [], "processes": {}})


def test_drift_check_abstains_on_foreign_vocabulary():
    assert J.drift_check({"oov_rate": 0.9})["in_distribution"] is False
    assert J.drift_check({"oov_rate": 0.1})["in_distribution"] is True


def test_drift_check_requires_the_training_schema_even_when_vocabulary_matches():
    """The vocabulary check cannot see feature-schema mismatch; this is the
    failure that took stock exports of HDFS traces to ~20% accuracy."""
    d = J.drift_check({"oov_rate": 0.0}, schema_attested=False)
    assert d["vocabulary_ok"] is True and d["in_distribution"] is False
    assert "training schema" in d["note"]
    assert J.drift_check({"oov_rate": 0.0}, schema_attested=True)["in_distribution"]


def test_adapter_attests_the_schema_only_when_the_exporter_supplies_it():
    stock = simple_trace()
    assert J.jaeger_to_spans(stock)[1]["schema_attested"] is False
    rich = simple_trace(triage_extra={"n_tid": 3, "n_host": 3, "n_edge": 2,
                                      "n_rep": 3, "lat": 90000})
    spans, extra = J.jaeger_to_spans(rich)
    assert extra["schema_attested"] is True and extra["n_host"] == 3


def test_explicit_agent_tag_overrides_the_service_name():
    t = simple_trace()
    t["spans"][1]["tags"] = [{"key": "triage.agent", "value": "Datanode"}]
    spans, _ = J.jaeger_to_spans(t)
    assert [s["agent"] for s in spans][1] == "Datanode"


# --------------------------------------------------------------------------
def _trace(errored=False, dur=100_000):
    status = "ERR" if errored else "OK"
    return [{"op": "root", "status": "OK", "duration": dur, "depth": 0,
             "host": "a", "agent": "a"},
            {"op": "leaf", "status": status, "duration": 10, "depth": 1,
             "host": "b", "agent": "b"}]


def test_canary_holds_when_too_little_traffic():
    r = canary_health([_trace()] * 200, [_trace()] * 5)
    assert r.verdict == "HOLD" and "insufficient" in r.reason


def test_canary_promotes_when_equivalent():
    r = canary_health([_trace()] * 300, [_trace()] * 100)
    assert r.verdict == "PROMOTE"


def test_canary_rolls_back_on_significant_error_increase():
    base = [_trace()] * 300 + [_trace(errored=True)] * 3          # ~1%
    canary = [_trace()] * 70 + [_trace(errored=True)] * 30        # 30%
    r = canary_health(base, canary)
    assert r.verdict == "ROLLBACK"


def test_canary_ignores_noise_level_difference():
    """3% vs 1% over 40 traces is inside the sampling noise - no rollback."""
    base = [_trace()] * 297 + [_trace(errored=True)] * 3
    canary = [_trace()] * 39 + [_trace(errored=True)] * 1
    assert canary_health(base, canary).verdict != "ROLLBACK"


def test_canary_rolls_back_on_latency_regression():
    base = [_trace(dur=100_000)] * 300
    canary = [_trace(dur=350_000)] * 100
    r = canary_health(base, canary)
    assert r.verdict == "ROLLBACK" and r.p95_ratio > 2


# --------------------------------------------------------------------------
def test_metrics_render_is_prometheus_text(tmp_path, monkeypatch):
    from src.policy_engine import database as db
    from src.observability.metrics import render_metrics
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "m.db"))
    db.init_db()
    db.log_decision("triage_agent", "t", "PASS", 0.9, "ALLOW_MERGE", "r", 2, True)
    text = render_metrics()
    assert "# TYPE triage_decisions_total counter" in text
    assert 'triage_decisions_total{classification="PASS",action="ALLOW_MERGE"} 1' in text
    assert 'agent_trust_tier{agent_id="triage_agent"}' in text


# --------------------------------------------------------------------------
# Poller: Jaeger -> agent over real HTTP, against stub servers
# --------------------------------------------------------------------------
def _serve(handler_cls):
    import threading
    from http.server import HTTPServer
    srv = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_poller_triages_only_new_traces_and_can_filter_errors():
    import json
    from http.server import BaseHTTPRequestHandler
    sys.path.insert(0, str(ROOT / "scripts"))
    import jaeger_poller as JP

    ok = simple_trace()
    bad = simple_trace(traceID="t-bad")
    bad["spans"][2]["tags"] = [{"key": "error", "value": True}]
    posted = []

    class FakeJaeger(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path.startswith("/api/traces?service=frontend")
            body = json.dumps({"data": [ok, bad]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
        def log_message(self, *a): pass

    class FakeAPI(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["Content-Length"])
            posted.append((self.path, json.loads(self.rfile.read(n))))
            body = json.dumps({"action": "PAUSE_FOR_MANUAL_REVIEW", "rule": "ood-abstain",
                               "test_id": posted[-1][1]["test_id"]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
        def log_message(self, *a): pass

    jsrv, jurl = _serve(FakeJaeger)
    asrv, aurl = _serve(FakeAPI)
    try:
        seen = set()
        out = JP.poll_once(jurl, aurl, "frontend", seen, only_errors=True)
        assert len(out) == 1 and posted[0][1]["test_id"] == "t-bad"
        assert posted[0][0] == "/api/v1/triage/jaeger"
        # second pass: nothing new
        assert JP.poll_once(jurl, aurl, "frontend", seen, only_errors=True) == []
        # without the filter the healthy trace is picked up too
        out = JP.poll_once(jurl, aurl, "frontend", seen, only_errors=False)
        assert len(out) == 1 and posted[-1][1]["test_id"] == "t1"
    finally:
        jsrv.shutdown(); asrv.shutdown()
