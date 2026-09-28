"""
Tests for the S3 trust-tier state machine (src/policy_engine/database.py).

Runs against a throwaway SQLite file, never the real trust_tiers.db, so these
are safe to run repeatedly and in any order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policy_engine import database as db  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test_trust.db"))
    db.init_db()
    yield


# ----------------------------------------------------------------------------
# Wilson lower bound - pure math, no DB
# ----------------------------------------------------------------------------
def test_wilson_lower_bound_zero_evidence_is_zero():
    assert db.wilson_lower_bound(0, 0) == 0.0


def test_wilson_lower_bound_punishes_small_samples():
    """17/20 = 85% point estimate, but the honest floor is well below that -
    this is the whole reason promotion is gated on the bound, not the raw
    accuracy."""
    lb = db.wilson_lower_bound(17, 20)
    assert lb < 0.85
    assert 0.55 < lb < 0.75


def test_wilson_lower_bound_converges_with_more_evidence():
    """Same 85% accuracy, far more evidence -> the bound should sit much
    closer to the point estimate."""
    lb_small = db.wilson_lower_bound(17, 20)
    lb_large = db.wilson_lower_bound(850, 1000)
    assert lb_large > lb_small
    assert lb_large > 0.82


def test_wilson_lower_bound_perfect_record_still_below_one():
    assert db.wilson_lower_bound(50, 50) < 1.0


# ----------------------------------------------------------------------------
# Promotion
# ----------------------------------------------------------------------------
def _feed_decisions(agent_id, n_correct, n_wrong, classification="PASS",
                    wrong_classification="INFRA_FLAKE"):
    for _ in range(n_correct):
        did = db.log_decision(agent_id, "t", classification, 0.9, "ALLOW",
                              "r", db.get_agent_tier(agent_id), True)
        db.record_ground_truth(did, classification)
    for _ in range(n_wrong):
        did = db.log_decision(agent_id, "t", wrong_classification, 0.9,
                              "RETRY", "r", db.get_agent_tier(agent_id), True)
        db.record_ground_truth(did, classification)


def test_promotion_holds_below_min_sample_size():
    agent = "agent_a"
    _feed_decisions(agent, n_correct=10, n_wrong=0)  # < min_n=15 for T0->T1
    assert db.get_agent_tier(agent) == 0


def test_promotion_fires_at_high_accuracy_with_enough_evidence():
    agent = "agent_b"
    _feed_decisions(agent, n_correct=20, n_wrong=0)  # 20/20, n>=15
    assert db.get_agent_tier(agent) == 1
    hist = db.wilson_lower_bound(20, 20)
    assert hist >= 0.70


def test_promotion_holds_on_mediocre_accuracy_despite_enough_evidence():
    agent = "agent_c"
    _feed_decisions(agent, n_correct=11, n_wrong=9)  # 55%, n=20 >= min_n
    assert db.get_agent_tier(agent) == 0


def test_promotion_climbs_multiple_tiers_over_a_long_stream():
    agent = "agent_d"
    _feed_decisions(agent, n_correct=60, n_wrong=0)
    assert db.get_agent_tier(agent) >= 2


def test_promotion_caps_at_t3():
    agent = "agent_e"
    _feed_decisions(agent, n_correct=200, n_wrong=0)
    assert db.get_agent_tier(agent) == 3
    result = db.evaluate_promotion(agent)
    assert result["action"] == "hold"
    assert result["reason"] == "already at T3"


# ----------------------------------------------------------------------------
# Demotion
# ----------------------------------------------------------------------------
def test_missed_real_bug_acted_on_autonomously_demotes_immediately():
    agent = "agent_f"
    db.set_agent_tier(agent, 3, "seeded for test")
    did = db.log_decision(agent, "t1", "PASS", 0.9, "ALLOW_MERGE", "r", 3,
                          True)  # autonomous!
    result = db.record_ground_truth(did, "REAL_BUG")
    assert db.get_agent_tier(agent) == 2
    assert result["tier_action"] == "demoted"


def test_missed_real_bug_under_human_review_does_not_demote():
    """If the decision was NOT autonomous (a human approved/rejected it),
    the miss is not the agent's failure alone - no demotion."""
    agent = "agent_g"
    db.set_agent_tier(agent, 3, "seeded for test")
    did = db.log_decision(agent, "t1", "PASS", 0.9, "PAUSE_FOR_MANUAL_REVIEW",
                          "r", 3, False)  # NOT autonomous
    db.record_ground_truth(did, "REAL_BUG")
    assert db.get_agent_tier(agent) == 3


def test_correctly_caught_real_bug_does_not_demote():
    agent = "agent_h"
    db.set_agent_tier(agent, 3, "seeded for test")
    did = db.log_decision(agent, "t1", "REAL_BUG", 0.9,
                          "BLOCK_MERGE_ALERT_DEV", "r", 3, True)
    db.record_ground_truth(did, "REAL_BUG")
    assert db.get_agent_tier(agent) == 3


def test_demotion_never_goes_below_t0():
    agent = "agent_i"
    assert db.get_agent_tier(agent) == 0
    did = db.log_decision(agent, "t1", "PASS", 0.9, "ALLOW_MERGE", "r", 0,
                          True)
    db.record_ground_truth(did, "REAL_BUG")
    assert db.get_agent_tier(agent) == 0


def test_tier_history_records_every_transition():
    agent = "agent_j"
    db.set_agent_tier(agent, 1, "manual bump for test")
    with db._conn() as c:
        rows = c.execute(
            "SELECT * FROM tier_history WHERE agent_id=? ORDER BY id",
            (agent,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["from_tier"] == 0 and rows[0]["to_tier"] == 1
    assert rows[0]["evidence"] == "manual bump for test"
