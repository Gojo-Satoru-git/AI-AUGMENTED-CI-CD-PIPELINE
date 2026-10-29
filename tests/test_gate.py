"""
Tests for the release gate: precedence of the four signals, and that the trust
tier still governs execution.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.observability.canary import CanaryReport      # noqa: E402
from src.policy_engine import policy as P              # noqa: E402
from src.policy_engine.gate import release_gate        # noqa: E402


def canary(verdict, score=0.9):
    return CanaryReport(verdict, score, f"canary says {verdict}", 100, 100,
                        0.0, 0.0, 1.0, 1.0, 1.0)


SEC_BLOCK = {"verdict": "BLOCK", "rule": "reachable-vuln-blocks",
             "reason": "1 reachable vulnerability: CVE-X"}
SEC_REVIEW = {"verdict": "REVIEW", "rule": "transitive-or-unknown-vuln",
              "reason": "needs a look"}
SEC_ALLOW = {"verdict": "ALLOW", "rule": "no-reachable-vuln", "reason": "ok"}
IN_DIST = {"in_distribution": True, "oov_rate": 0.0}
OOD = {"in_distribution": False, "oov_rate": 0.9, "note": "foreign vocabulary"}


def test_clean_signals_defer_to_the_triage_decision():
    t = P.decide("PASS", 0.95, trust_tier=3)
    g = release_gate(t, SEC_ALLOW, canary("PROMOTE"), IN_DIST)
    assert g.action == P.ALLOW and g.autonomous and g.rule == "pass-allows"


def test_reachable_vuln_blocks_even_a_healthy_pass():
    t = P.decide("PASS", 0.99, trust_tier=3)
    g = release_gate(t, SEC_BLOCK, canary("PROMOTE"), IN_DIST)
    assert g.action == P.BLOCK and g.rule == "security-blocks"
    assert "CVE-X" in g.reasons[0]


def test_canary_rollback_beats_a_passing_triage():
    t = P.decide("PASS", 0.99, trust_tier=3)
    g = release_gate(t, SEC_ALLOW, canary("ROLLBACK"), IN_DIST)
    assert g.action == P.ROLLBACK and g.rule == "canary-rollback"


def test_security_block_outranks_canary_rollback():
    g = release_gate(P.decide("PASS", 0.99, 3), SEC_BLOCK, canary("ROLLBACK"))
    assert g.rule == "security-blocks"


def test_out_of_distribution_trace_abstains_to_a_human():
    t = P.decide("PASS", 0.99, trust_tier=3)      # model is confident but blind
    g = release_gate(t, SEC_ALLOW, canary("PROMOTE"), OOD)
    assert g.action == P.REVIEW and g.rule == "ood-abstain"
    assert not g.autonomous


def test_real_bug_still_blocks_when_everything_else_is_clean():
    t = P.decide("REAL_BUG", 0.9, trust_tier=3)
    g = release_gate(t, SEC_ALLOW, canary("PROMOTE"), IN_DIST)
    assert g.action == P.BLOCK and g.rule == "real-bug-blocks"


def test_allow_is_downgraded_by_security_review_or_canary_hold():
    t = P.decide("PASS", 0.95, trust_tier=3)
    assert release_gate(t, SEC_REVIEW).rule == "security-review"
    assert release_gate(t, SEC_ALLOW, canary("HOLD")).rule == "canary-hold"
    assert release_gate(t, SEC_REVIEW).action == P.REVIEW


def test_low_tier_agent_can_only_recommend_a_security_block():
    t = P.decide("PASS", 0.95, trust_tier=0)
    g = release_gate(t, SEC_BLOCK, None, IN_DIST)
    assert g.recommended == P.BLOCK        # the signals still say block ...
    assert g.action == P.REVIEW            # ... but T0 may only recommend it
    assert not g.autonomous


def test_rollback_requires_tier_two():
    g1 = release_gate(None, None, canary("ROLLBACK"), trust_tier=1)
    g2 = release_gate(None, None, canary("ROLLBACK"), trust_tier=2)
    assert g1.action == P.REVIEW and g1.recommended == P.ROLLBACK
    assert g2.action == P.ROLLBACK and g2.autonomous


def test_signals_are_all_reported_for_the_pr_comment():
    t = P.decide("INFRA_FLAKE", 0.9, trust_tier=3)
    g = release_gate(t, SEC_ALLOW, canary("PROMOTE"), IN_DIST)
    assert set(g.signals) == {"triage", "security", "canary", "drift"}
    assert g.action == P.RETRY


def test_no_signals_at_all_defaults_to_allow_only_when_autonomous():
    g = release_gate(None, None, None, None, trust_tier=3)
    assert g.action == P.ALLOW
    g0 = release_gate(None, None, None, None, trust_tier=0)
    assert g0.action == P.REVIEW and g0.recommended == P.ALLOW
