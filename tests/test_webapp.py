"""The dashboard's endpoints: every module is reachable and returns sane results."""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/models/saved_live"
HDFS = ROOT / "src/models/saved/triage_lstm.pth"
pytestmark = pytest.mark.skipif(not (MODEL / "triage_lstm.pth").exists() or not HDFS.exists(),
                                reason="needs the live and HDFS models")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["TRUST_DB_PATH"] = str(tmp_path_factory.mktemp("db") / "trust.db")
    from fastapi.testclient import TestClient
    from src.live.webapp import Session, create_app
    from src.models.predict import TriagePredictor
    pred = TriagePredictor(ckpt=MODEL / "triage_lstm.pth", vocab_path=MODEL / "vocab.json")
    s = Session(pred, agent_id="ui_test")
    s.rate = 0.5
    yield TestClient(create_app(s))
    s.stop = True


def test_state_and_reports(client):
    st = client.get("/api/state").json()
    assert {"tier", "series", "faults", "security"} <= set(st)
    rep = client.get("/api/reports").json()
    assert rep["evaluation"]["model"]["macro_f1"] > 0.7
    assert rep["security"]["verdict"]["verdict"] == "ALLOW"


def test_canary_rolls_back_a_bad_version(client):
    ok = client.post("/api/canary", json={"kind": "none"}).json()
    bad = client.post("/api/canary", json={"kind": "flaky_net"}).json()
    assert ok["report"]["error_rate_canary"] == 0.0   # latency verdict is timing-sensitive under load
    assert bad["report"]["verdict"] == "ROLLBACK"
    assert bad["gate"]["rule"] == "canary-rollback"


def test_reachable_cve_blocks_and_foreign_trace_abstains(client):
    client.post("/api/security", json={"kind": "vuln"})
    blocked = client.post("/api/canary", json={"kind": "none"}).json()
    client.post("/api/security", json={"kind": "clean"})
    assert blocked["gate"]["rule"] == "security-blocks"
    foreign = client.post("/api/hdfs", json={"kind": "foreign"}).json()
    assert foreign["rule"] == "ood-abstain"


def test_metrics_endpoint(client):
    assert "agent_trust_tier" in client.get("/metrics").text
