"""
Tests for the Test-Triage Agent.

The Phase 1 suite imported `src.models.main`, which does not exist, so it had
never run. These cover the things that were actually wrong: the label/action
inversion, token-to-feature alignment, and split leakage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_sequences as bs          # noqa: E402
from src.policy_engine import policy as P           # noqa: E402

CORPUS = ROOT / "data" / "processed" / "corpus.jsonl.gz"
CKPT = ROOT / "src" / "models" / "saved" / "triage_lstm.pth"
VOCAB = ROOT / "src" / "models" / "saved" / "vocab.json"


# --------------------------------------------------------------------------
# Label taxonomy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("folder,expected", [
    ("NM_CL_rw_30C_1to19B_0to120INTR_0to600INTW_50RT_10WT", "PASS"),
    ("NM_DN_w_25DN_30C_1to19B_0to600INT_50RT_10WT", "PASS"),
    ("AN_Net_disconnectDN_w_30FDN_30C_1to19B_0to600INT_15RT_5WT",
     "INFRA_FLAKE"),
    ("AN_Proc_killDN_w_10FDN_30C_1to19B_0to600INT_15RT_5WT", "INFRA_FLAKE"),
    ("AN_Data_corruptBlk_r_01FDN_30C_1to19B_0to120INT_15RT_5WT", "DATA_FAULT"),
    ("AN_Sys_panicDN_w_03FDN_30C_1to19B_0to600INT_15RT_5WT", "ENV_FAULT"),
    ("COM_Sin_Bug_rwrpc_1_30C_1to5B_0to120INTR_0to600INTW_0to10INTRPC_30RT_5WT",
     "REAL_BUG"),
    ("COM_Sin_Net_rwrpc_2_30C_1to19B_0to120INTR_0to600INTW_0to0INTRPC_60RT_10WT",
     "COMPOUND"),
    ("convert_sql_to_csv", None),
])
def test_class_of(folder, expected):
    assert bs.class_of(folder) == expected


def test_real_bug_is_not_swallowed_by_compound():
    """COM_Sin_Bug is the only genuine code-bug class; it must not fall into
    the generic COMPOUND bucket, or REAL_BUG has no training data."""
    assert bs.class_of("COM_Sin_Bug_rwrpc_1_x") == "REAL_BUG"
    assert bs.class_of("COM_Sin_Data_rwrpc_1_x") == "COMPOUND"


@pytest.mark.parametrize("desc,expected", [
    ("Success: return(OW[class=...])", "OK"),
    ("IOException: Bad connect ack", "IOEXC"),
    ("Exception: something", "EXC"),
    ("ERROR: fatal", "ERR"),
    ("A user task", "USER"),
    ("totally unexpected text", "OTH"),
])
def test_status_of(desc, expected):
    assert bs.status_of(desc) == expected


def test_host_role():
    assert bs.host_role("namenode") == 0
    assert bs.host_role("datanode038") == 1
    assert bs.host_role("client012") == 2
    assert bs.host_role("weird-box") == 3


# --------------------------------------------------------------------------
# Clock alignment - the piece the sequence order depends on
# --------------------------------------------------------------------------
def test_align_clocks_puts_caller_before_handler():
    """Two hosts with wildly different clock bases. The client calls at its
    t=1000; the namenode handles it at its own t=500000. Raw StartTime would
    order the namenode first; alignment must not."""
    rows = [
        # (order, TID, OpName, start, end, hostname, agent, status)
        (0, "NN1", "getFileInfo", 500000, 500100, "namenode", "Namenode", "OK"),
        (1, "CL1", "RPC:getFileInfo", 1000, 1300, "client001", "RPC Client",
         "OK"),
    ]
    edges = [("ROOT", 900, "CL1"), ("CL1", 1000, "NN1")]
    keyed, aligned = bs.build_task(rows, edges)
    assert aligned, "both hosts should be reachable from the root"
    assert [k[2] for k in keyed] == ["RPC:getFileInfo", "getFileInfo"]


def test_align_clocks_reports_unaligned_hosts():
    rows = [
        (0, "A", "op1", 10, 20, "hostA", "Namenode", "OK"),
        (1, "B", "op2", 999999, 1000000, "hostB", "Datanode", "OK"),
    ]
    keyed, aligned = bs.build_task(rows, edges=[])   # no edges -> no offsets
    assert not aligned
    assert len(keyed) == 2                            # still usable, flagged


def test_opname_with_spaces_is_normalised():
    """OpNames like "OP: connect next Datanode" contain spaces, and the token
    stream is space-joined. Without normalisation the per-token feature arrays
    desynchronise from the token list - this bit us once already."""
    rows = [(0, "T1", "OP: connect next Datanode", 10, 30, "datanode001",
             "Datanode", "IOEXC")]
    keyed, _ = bs.build_task(rows, edges=[])
    tok = f"{keyed[0][2].strip().replace(' ', '_').replace('|', '/')}|OK"
    assert " " not in tok.rsplit("|", 1)[0]


# --------------------------------------------------------------------------
# Policy - the inversion that Phase 1 shipped
# --------------------------------------------------------------------------
def test_healthy_trace_is_not_blocked():
    """Phase 1 mapped every non-anomalous trace to "Code Bug" and blocked the
    merge. A PASS must never block."""
    d = P.decide("PASS", confidence=0.95, trust_tier=3)
    assert d.action == P.ALLOW
    assert d.action != P.BLOCK


def test_real_bug_blocks_merge():
    d = P.decide("REAL_BUG", confidence=0.9, trust_tier=3)
    assert d.action == P.BLOCK


def test_only_infra_flake_is_retryable():
    assert P.decide("INFRA_FLAKE", 0.9, 3).action == P.RETRY
    for cls in ("DATA_FAULT", "ENV_FAULT", "REAL_BUG"):
        assert P.decide(cls, 0.9, 3).action != P.RETRY, (
            f"{cls} must never be retried")


def test_retry_budget_escalates():
    assert P.decide("INFRA_FLAKE", 0.9, 3, attempt=1).action == P.RETRY
    d = P.decide("INFRA_FLAKE", 0.9, 3, attempt=3, retry_budget=2)
    assert d.action == P.ESCALATE
    assert d.rule == "retry-budget-exhausted"


def test_low_confidence_always_goes_to_human():
    d = P.decide("REAL_BUG", confidence=0.31, trust_tier=3)
    assert d.action == P.REVIEW
    assert not d.autonomous
    assert d.rule == "low-confidence"


def test_low_tier_agent_cannot_act_autonomously():
    t0 = P.decide("INFRA_FLAKE", 0.99, trust_tier=0)
    t3 = P.decide("INFRA_FLAKE", 0.99, trust_tier=3)
    assert t0.action == P.REVIEW and not t0.autonomous
    assert t3.action == P.RETRY and t3.autonomous


def test_quarantine_requires_top_tier():
    assert P.decide("ENV_FAULT", 0.9, trust_tier=2).action == P.REVIEW
    assert P.decide("ENV_FAULT", 0.9, trust_tier=3).action == P.QUARANTINE


def test_unknown_class_is_safe():
    assert P.decide("SOMETHING_NEW", 0.99, 3).action == P.REVIEW


# --------------------------------------------------------------------------
# Corpus / split integrity (needs the built corpus)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_corpus_token_feature_alignment():
    import gzip
    import json
    checked = 0
    with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n = r["toks"].count(" ") + 1
            assert len(r["dur"].split(",")) == n
            assert len(r["dep"].split(",")) == n
            assert len(r["hr"]) == n and len(r["ag"]) == n
            checked += 1
            if checked >= 5000:
                break
    assert checked > 0


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_split_has_no_scenario_or_task_leakage():
    from src.data.dataset import prepare
    ds, vocab, itos, meta = prepare(CORPUS, max_len=256)   # asserts internally
    assert vocab["<PAD>"] == 0 and vocab["<UNK>"] == 1
    assert meta["scenarios_per_split"]["test"] > 0
    tr = {r["scen"] for r in ds["train"].recs}
    te = {r["scen"] for r in ds["test"].recs}
    assert not (tr & te)


# --------------------------------------------------------------------------
# Serving (needs a trained checkpoint)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not (CKPT.exists() and VOCAB.exists()),
                    reason="model not trained yet")
def test_predictor_returns_a_usable_decision():
    from src.models.predict import TriagePredictor
    p = TriagePredictor()
    out = p.predict(tokens="fs_-copyFromLocal|USER RPC:getFileInfo|OK "
                           "getFileInfo|OK RPC:addBlock|IOEXC addBlock|OK")
    assert out["classification"] in p.labels
    assert 0.0 <= out["confidence"] <= 1.0
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-3
    assert out["evidence_spans"], "a decision must carry its evidence"
    d = P.decide(out["classification"], out["confidence"], trust_tier=2)
    assert d.action in {P.ALLOW, P.RETRY, P.REVIEW, P.ESCALATE,
                        P.QUARANTINE, P.BLOCK}


@pytest.mark.skipif(not (CKPT.exists() and VOCAB.exists()),
                    reason="model not trained yet")
def test_api_triage_endpoint():
    from fastapi.testclient import TestClient
    from src.policy_engine.main import app
    c = TestClient(app)
    r = c.post("/api/v1/triage", json={
        "test_id": "T1",
        "trace_sequence": "fs_-mv|USER getFileInfo|OK RPC:getFileInfo|OK",
        "agent_id": "triage_agent",
    })
    assert r.status_code == 200
    body = r.json()
    for k in ("classification", "confidence", "action", "trust_tier",
              "evidence_spans", "rule"):
        assert k in body, f"missing {k}"


RCA_BASELINE = (ROOT / "src" / "models" / "saved" /
                "op_latency_baseline.json")


@pytest.mark.skipif(not (CKPT.exists() and VOCAB.exists()
                         and RCA_BASELINE.exists()),
                    reason="model or RCA baseline not built yet")
def test_api_postmortem_endpoint():
    from fastapi.testclient import TestClient
    from src.policy_engine.main import app
    c = TestClient(app)
    r = c.post("/api/v1/postmortem", json={
        "test_id": "T1",
        "spans": [
            {"op": "getFileInfo", "status": "OK", "duration": 100},
            {"op": "addBlock", "status": "ERR", "duration": 5000},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["suspects"], "must return a ranked suspect list"
    assert "addBlock" in body["postmortem_markdown"]
    assert body["suspects"][0]["error"] is True
