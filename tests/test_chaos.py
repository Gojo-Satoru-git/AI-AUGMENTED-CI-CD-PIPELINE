"""
Tests for S7: fault injectors keep the corpus record well-formed, behave as
documented, and the runner wires them into the trust engine correctly.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chaos import experiments as X     # noqa: E402
from src.policy_engine import database as db   # noqa: E402


def healthy(task="T1"):
    toks = ["fs_-copyToLocal|USER", "RPC:getFileInfo|OK", "getFileInfo|OK",
            "chooseDataNode|OK", "sendBlock|OK", "receiveBlock|OK",
            "checksumOk|OK", "getFileInfo|OK"]
    n = len(toks)
    return {
        "task": task, "scen": "NM_test", "cls": "PASS", "title": "fs -cp",
        "toks": " ".join(toks),
        "dur": ",".join(["100", "200", "50", "250000", "9000", "8000", "400", "60"]),
        "hr": "22011100", "ag": "42011100", "dep": ",".join(["0", "1", "2", "2", "2", "2", "2", "2"]),
        "f": {"n_ev": n, "n_tid": n, "n_host": 3, "n_edge": n - 1, "n_rep": n,
              "lat": 300000, "n_err": 0, "n_op": 7},
    }


def well_formed(rec):
    n = rec["toks"].count(" ") + 1
    return (len(rec["dur"].split(",")) == n and len(rec["dep"].split(",")) == n
            and len(rec["hr"]) == n and len(rec["ag"]) == n
            and rec["f"]["n_ev"] == n)


@pytest.mark.parametrize("name", list(X.FAULTS))
def test_every_injector_keeps_parallel_arrays_aligned(name):
    """A misaligned injector would corrupt the model's inputs silently - the
    same class of bug as the OpName-with-spaces defect fixed in S1."""
    rng = random.Random(0)
    for sev in (0.1, 0.5, 1.0):
        out = X.FAULTS[name].inject(healthy(), sev, rng)
        assert well_formed(out), f"{name} @ {sev} broke array alignment"


def test_injectors_do_not_mutate_their_input():
    rec = healthy()
    snapshot = dict(rec["f"]), rec["toks"]
    for f in X.FAULTS.values():
        f.inject(rec, 1.0, random.Random(1))
    assert (rec["f"], rec["toks"]) == snapshot


def test_slow_datanode_scales_only_datanode_spans():
    out = X.slow_datanode(healthy(), 1.0, random.Random(0))
    before = [int(x) for x in healthy()["dur"].split(",")]
    after = [int(x) for x in out["dur"].split(",")]
    for i, h in enumerate(healthy()["hr"]):
        if h == "1":
            assert after[i] > before[i] * 50
        else:
            assert after[i] == before[i]
    assert out["f"]["lat"] > healthy()["f"]["lat"]


def test_datanode_kill_adds_the_real_50_minute_hangs():
    out = X.datanode_kill(healthy(), 1.0, random.Random(0))
    durs = [int(x) for x in out["dur"].split(",")]
    assert sum(1 for d in durs if d == X.HUGE_US) == 3
    assert out["f"]["n_err"] >= 2          # the IOEXC retries


def test_corruption_and_readonly_add_their_failure_signatures():
    c = X.block_corruption(healthy(), 1.0, random.Random(0))
    assert "sendBlock|ERR" in c["toks"] and "reportBadBlocks|OK" in c["toks"]
    r = X.readonly_datanode(healthy(), 1.0, random.Random(0))
    assert "OP:_new_BlockReceiver|OTH" in r["toks"]
    assert c["f"]["n_err"] > 0 and r["f"]["n_err"] > 0


def test_severity_scales_the_amount_of_damage():
    lo = X.block_corruption(healthy(), 0.1, random.Random(0))
    hi = X.block_corruption(healthy(), 1.0, random.Random(0))
    assert hi["f"]["n_ev"] > lo["f"]["n_ev"]


def test_silent_regression_changes_nothing_visible():
    rec = healthy()
    assert X.silent_regression(rec, 1.0, random.Random(0)) == rec
    assert X.FAULTS["silent_regression"].detectable_by_trace is False


# --------------------------------------------------------------------------
class FakePredictor:
    """Always answers PASS @0.9 - a model blind to every injected fault."""
    def __init__(self, cls="PASS", conf=0.9):
        self.cls, self.conf = cls, conf

    def predict_record(self, rec, top_spans=5):
        return {"classification": self.cls, "confidence": self.conf,
                "evidence_spans": []}


@pytest.fixture()
def tmpdb(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "chaos.db"))
    db.init_db()


def pool(n=40):
    return [healthy(f"T{i}") for i in range(n)]


def test_blind_agent_is_demoted_by_a_silent_regression(tmpdb):
    rep = X.run_chaos(FakePredictor(), pool(), faults=["silent_regression"],
                      n_per_fault=5, start_tier=3, retention=False)
    assert rep["demotions"], "an autonomous miss of a REAL_BUG must demote"
    assert rep["demotions"][0]["from"] == 3
    assert rep["by_fault"]["silent_regression"]["missed_real_bugs"] == 5
    assert rep["unsafe_actions"].get("unsafe-merge", 0) >= 1


def test_retention_demotes_a_blind_agent_facing_visible_faults(tmpdb):
    rep = X.run_chaos(FakePredictor(), pool(60),
                      faults=["block_corruption", "readonly_datanode"],
                      n_per_fault=30, start_tier=3, retention=True)
    assert rep["final_tier"] < 3
    assert any("retention" in d["reason"] for d in rep["demotions"])


def test_correct_agent_keeps_its_tier(tmpdb):
    class Oracle(FakePredictor):
        def predict_record(self, rec, top_spans=5):
            n_err = rec["f"]["n_err"]
            cls = "DATA_FAULT" if "sendBlock|ERR" in rec["toks"] else "PASS"
            return {"classification": cls, "confidence": 0.95, "evidence_spans": []}
    rep = X.run_chaos(Oracle(), pool(60), faults=["block_corruption"],
                      n_per_fault=30, start_tier=3, retention=True)
    assert rep["final_tier"] == 3 and not rep["demotions"]
    assert rep["by_fault"]["block_corruption"]["class_match_rate"] == 1.0
    assert rep["by_fault"]["control"]["class_match_rate"] == 1.0


def test_runner_restores_the_global_retention_flag(tmpdb):
    before = db.RETENTION_ENABLED
    X.run_chaos(FakePredictor(), pool(), faults=["slow_datanode"],
                n_per_fault=3, retention=True)
    assert db.RETENTION_ENABLED == before


def test_log_false_leaves_the_database_untouched(tmpdb):
    X.run_chaos(FakePredictor(), pool(), faults=["slow_datanode"],
                n_per_fault=3, log=False)
    with db._conn() as c:
        assert c.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"] == 0


def test_controls_are_identical_across_severities():
    """Sweeps compare against the same healthy sample, whatever the injectors
    did to the shared RNG."""
    a = X.run_chaos(FakePredictor(), pool(60), n_per_fault=10, severity=0.1,
                    log=False)["by_fault"]["control"]
    b = X.run_chaos(FakePredictor(), pool(60), n_per_fault=10, severity=1.0,
                    log=False)["by_fault"]["control"]
    assert a == b
