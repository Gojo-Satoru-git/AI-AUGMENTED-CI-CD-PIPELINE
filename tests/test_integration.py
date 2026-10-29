"""
End-to-end integration: HTTP API -> triage model -> drift guard -> gate ->
actuator (dry run). Uses the real trained checkpoint, the replacement for the
old simulate_triage.py script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CKPT = ROOT / "src" / "models" / "saved" / "triage_lstm.pth"
VOCAB = ROOT / "src" / "models" / "saved" / "vocab.json"
pytestmark = pytest.mark.skipif(not (CKPT.exists() and VOCAB.exists()),
                                reason="model not trained yet")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.policy_engine.main import app
    yield TestClient(app)
    # these tests use the real trust DB; leave no test agents behind
    from src.policy_engine import database as db
    with db._conn() as c:
        for t in ("decisions", "tier_history", "agent_trust"):
            c.execute(f"DELETE FROM {t} WHERE agent_id LIKE 'it\\_%' ESCAPE '\\'")


HDFS_SPANS = [
    {"op": "fs -copyToLocal", "status": "USER", "duration": 100, "depth": 0},
    {"op": "RPC:getFileInfo", "status": "OK", "duration": 400, "depth": 1},
    {"op": "getFileInfo", "status": "OK", "duration": 120, "depth": 2},
    {"op": "RPC:addBlock", "status": "IOEXC", "duration": 9000, "depth": 1},
    {"op": "addBlock", "status": "OK", "duration": 500, "depth": 2},
]


def jaeger_trace(op_prefix="svc"):
    procs = {"p1": {"serviceName": "frontend"}, "p2": {"serviceName": "orders"}}
    spans = [
        {"traceID": "t", "spanID": "a", "operationName": f"{op_prefix}.checkout",
         "references": [], "startTime": 1_000_000, "duration": 90_000,
         "tags": [], "logs": [], "processID": "p1"},
        {"traceID": "t", "spanID": "b", "operationName": f"{op_prefix}.createOrder",
         "references": [{"refType": "CHILD_OF", "traceID": "t", "spanID": "a"}],
         "startTime": 1_010_000, "duration": 60_000,
         "tags": [{"key": "error", "value": True}], "logs": [], "processID": "p2"},
    ]
    return {"traceID": "t", "spans": spans, "processes": procs}


def test_metrics_endpoint_is_prometheus_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "# TYPE triage_decisions_total counter" in r.text
    assert "agent_trust_tier" in r.text


def test_foreign_system_trace_abstains_instead_of_guessing(client):
    """A TrainTicket/HotROD-style trace shares no vocabulary with HDFS. The
    model must not be allowed to act on it."""
    r = client.post("/api/v1/triage/jaeger", json={
        "test_id": "ood-1", "agent_id": "it_ood_agent",
        "trace": {"data": [jaeger_trace()]}})
    assert r.status_code == 200
    body = r.json()
    assert body["rule"] == "ood-abstain"
    assert body["action"] == "PAUSE_FOR_MANUAL_REVIEW" and not body["autonomous"]
    assert body["signals"]["drift"]["in_distribution"] is False
    assert body["oov_rate"] > 0.5


ATTESTED = {"schema_attested": True, "n_tid": 3, "n_host": 3, "n_edge": 4,
            "n_rep": 5, "lat": 9000}


def test_stock_trace_without_schema_attestation_abstains(client):
    """Same HDFS operation names, but no training-schema task counts: measured
    ~20% accuracy on such traces (reports/adapter_fidelity.json), so the gate
    must not act on the classification."""
    r = client.post("/api/v1/gate", json={
        "test_id": "it-unattested", "agent_id": "it_unatt_agent",
        "spans": HDFS_SPANS, "use_security_report": False})
    body = r.json()
    assert body["rule"] == "ood-abstain" and not body["autonomous"]
    d = body["signals"]["drift"]
    assert d["vocabulary_ok"] is True and d["schema_attested"] is False
    assert d["in_distribution"] is False


def test_gate_with_an_in_distribution_trace_reports_every_signal(client):
    r = client.post("/api/v1/gate", json={
        "test_id": "it-1", "agent_id": "it_gate_agent", "spans": HDFS_SPANS,
        "extra": ATTESTED, "use_security_report": False})
    assert r.status_code == 200
    body = r.json()
    assert body["classification"] in ("PASS", "INFRA_FLAKE", "DATA_FAULT",
                                      "ENV_FAULT", "REAL_BUG")
    assert body["signals"]["drift"]["in_distribution"] is True
    assert body["decision_id"] is not None
    assert body["rule"] != "ood-abstain"


def test_gate_canary_rollback_for_a_sick_canary(client):
    ok = [{"op": "root", "status": "OK", "duration": 100_000, "depth": 0}]
    bad = [{"op": "root", "status": "OK", "duration": 100_000, "depth": 0},
           {"op": "leaf", "status": "ERR", "duration": 10, "depth": 1}]
    r = client.post("/api/v1/gate", json={
        "test_id": "it-canary", "agent_id": "it_canary_agent",
        "baseline_traces": [ok] * 200, "canary_traces": [ok] * 20 + [bad] * 30,
        "use_security_report": False})
    assert r.status_code == 200
    body = r.json()
    assert body["recommended"] == "ROLLBACK_CANARY"
    assert body["rule"] == "canary-rollback"
    assert body["signals"]["canary"]["verdict"] == "ROLLBACK"


def test_gate_without_any_signal_is_rejected_cleanly(client):
    r = client.post("/api/v1/gate", json={"test_id": "empty",
                                          "use_security_report": False})
    assert r.status_code == 200          # no negative signal -> T0 may only recommend
    assert r.json()["action"] == "PAUSE_FOR_MANUAL_REVIEW"


def test_security_verdict_endpoint_reflects_the_report(client):
    report = ROOT / "reports" / "security_gate.json"
    r = client.get("/api/v1/security/verdict")
    if report.exists():
        assert r.status_code == 200 and "verdict" in r.json()
    else:
        assert r.status_code == 404


def test_decision_flows_into_the_actuator_in_dry_run(client):
    """API decision -> actuator: the full path a CI run takes."""
    from src.actuator.actuator import Actuator, DryRunBackend, exit_code_for
    body = client.post("/api/v1/triage/jaeger", json={
        "test_id": "e2e-1", "agent_id": "it_e2e_agent",
        "trace": jaeger_trace()}).json()
    backend = DryRunBackend()
    recs = Actuator(backend).execute(body)
    assert [c[0] for c in backend.calls] == ["set_status", "comment"]
    assert backend.calls[0][1]["state"] == "pending"      # waiting on a human
    assert all(r.dry_run for r in recs)
    assert exit_code_for(body["action"]) == 1             # required check stays red


def test_ground_truth_feedback_still_closes_the_loop(client):
    body = client.post("/api/v1/gate", json={
        "test_id": "fb-1", "agent_id": "it_fb_agent", "spans": HDFS_SPANS,
        "extra": ATTESTED, "use_security_report": False}).json()
    r = client.post("/api/v1/triage/ground-truth", json={
        "decision_id": body["decision_id"], "ground_truth": "PASS"})
    assert r.status_code == 200 and r.json()["ok"] is True


# --------------------------------------------------------------------------
# CI contract: exit codes of scripts/ci_gate.py on the shipped samples
# --------------------------------------------------------------------------
SAMPLES = ROOT / "samples"


def run_ci_gate(tmp_path, *args):
    import subprocess
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ci_gate.py"), "--tier", "3",
         "--out-dir", str(tmp_path), *args],
        capture_output=True, text=True, encoding="utf-8", timeout=120)


@pytest.mark.skipif(not (SAMPLES / "canary_good.json").exists(),
                    reason="run scripts/make_samples.py")
def test_ci_gate_exit_codes(tmp_path):
    ok = run_ci_gate(tmp_path, "--baseline-jaeger", str(SAMPLES / "canary_baseline.json"),
                     "--canary-jaeger", str(SAMPLES / "canary_good.json"))
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = run_ci_gate(tmp_path, "--baseline-jaeger", str(SAMPLES / "canary_baseline.json"),
                      "--canary-jaeger", str(SAMPLES / "canary_bad.json"))
    assert bad.returncode == 1 and "ROLLBACK_CANARY" in bad.stdout
    foreign = run_ci_gate(tmp_path, "--jaeger", str(SAMPLES / "jaeger_foreign_checkout.json"))
    assert foreign.returncode == 1 and "ood-abstain" in foreign.stdout
    comment = (tmp_path / "gate_comment.md").read_text(encoding="utf-8")
    assert "<!-- ai-gate -->" in comment
    assert (tmp_path / "gate_decision.json").exists()
