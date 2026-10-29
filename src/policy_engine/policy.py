"""
Triage policy: map a classification and an agent trust tier to an action.

This replaces the Phase 1 rule, which was inverted. That rule read:

    is_flaky        = probability > 0.80
    classification  = "Flaky/Infrastructure" if is_flaky else "Code Bug"
    if classification == "Code Bug": BLOCK_MERGE_ALERT_DEV

The model it sat on was trained with label 1 = HDFS storage anomaly and
label 0 = a HEALTHY trace, so every healthy trace was classified "Code Bug"
and merge-blocked, and no class in the training data corresponded to a code
bug at all. The five classes below come from TraceBench's own fault taxonomy,
and COM_Sin_Bug gives REAL_BUG an honest definition: real HDFS defects.

Two distinctions carry the operational value:
  * INFRA_FLAKE is retry-eligible; DATA_FAULT and ENV_FAULT are not. Retrying
    a corruption fault burns CI minutes and still fails.
  * REAL_BUG blocks the merge. Missing one ships a defect, so it is the error
    we weight hardest.

Tiers here are AGENT autonomy levels, not developer seniority:
    T0 observe-only  T1 recommend  T2 act with rollback  T3 act autonomously
"""
from __future__ import annotations

from dataclasses import dataclass

# Actions the deployment actuator understands.
ALLOW = "ALLOW_MERGE"
RETRY = "AUTO_RETRY_PIPELINE"
REVIEW = "PAUSE_FOR_MANUAL_REVIEW"
ESCALATE = "ESCALATE_TO_INFRA"
QUARANTINE = "QUARANTINE_NODE"
BLOCK = "BLOCK_MERGE_ALERT_DEV"
ROLLBACK = "ROLLBACK_CANARY"        # S5/S8: canary health says revert

RETRYABLE = {"INFRA_FLAKE"}
NEVER_RETRY = {"DATA_FAULT", "ENV_FAULT", "REAL_BUG"}

# Minimum tier at which the agent may take each action without a human.
MIN_TIER = {ALLOW: 2, RETRY: 2, ESCALATE: 1, QUARANTINE: 3, BLOCK: 1,
            ROLLBACK: 2,
            REVIEW: 0}

DEFAULT_CONFIDENCE_FLOOR = 0.60
DEFAULT_RETRY_BUDGET = 2


@dataclass
class Decision:
    action: str
    reason: str
    rule: str
    autonomous: bool
    classification: str
    confidence: float
    trust_tier: int


def decide(classification: str, confidence: float, trust_tier: int,
           attempt: int = 1, retry_budget: int = DEFAULT_RETRY_BUDGET,
           confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR) -> Decision:
    """Pure function - no I/O - so it is trivially unit-testable and is the
    same logic the Rego policy will encode in S3."""
    def out(action, reason, rule):
        # An agent below the action's tier may only RECOMMEND it. The
        # recommendation still travels; a human approves the execution.
        autonomous = trust_tier >= MIN_TIER[action]
        return Decision(action if autonomous else REVIEW, reason, rule,
                        autonomous, classification, confidence, trust_tier)

    if classification not in RETRYABLE | NEVER_RETRY | {"PASS"}:
        return out(REVIEW, f"unknown class {classification!r}", "unknown-class")

    # Low confidence never acts autonomously, whatever the tier.
    if confidence < confidence_floor:
        return Decision(REVIEW,
                        f"confidence {confidence:.2f} below floor "
                        f"{confidence_floor:.2f}", "low-confidence",
                        False, classification, confidence, trust_tier)

    if classification == "REAL_BUG":
        return out(BLOCK, "genuine defect signature - merging ships the bug",
                   "real-bug-blocks")

    if classification == "PASS":
        return out(ALLOW, "no failure signature in the trace", "pass-allows")

    if classification == "INFRA_FLAKE":
        if attempt > retry_budget:
            return out(ESCALATE,
                       f"transient signature but retry budget "
                       f"({retry_budget}) exhausted after {attempt} attempts",
                       "retry-budget-exhausted")
        return out(RETRY, "transient network/process fault - retry is likely "
                          "to succeed", "flake-retries")

    if classification == "DATA_FAULT":
        return out(ESCALATE, "data corruption or loss is deterministic - a "
                             "retry will fail identically", "data-escalates")

    if classification == "ENV_FAULT":
        return out(QUARANTINE, "host-level failure - drain the node before "
                               "scheduling more work", "env-quarantines")

    return out(REVIEW, "no rule matched", "fallthrough")
