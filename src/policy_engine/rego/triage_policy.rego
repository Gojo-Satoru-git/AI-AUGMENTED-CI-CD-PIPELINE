# Triage policy, expressed for Open Policy Agent — a literal Rego mirror of
# src/policy_engine/policy.py's `decide()`. Python is the policy engine's
# actual runtime (see that file for why it stays a pure function); this
# module exists so the decision logic is also expressed as data-plane policy,
# the form S3 was scoped to produce, and so it can be evaluated independently
# of the Python process (e.g. from a CI step with only the OPA CLI installed).
#
# Equivalence with policy.py is enforced by tests/test_rego_policy.py, which
# feeds the same case table through both `decide()` and this file (via the
# `opa eval` CLI, skipped automatically if `opa` is not on PATH) and asserts
# identical actions.
package triage.policy

import future.keywords.if
import future.keywords.in

default confidence_floor := 0.60
default retry_budget := 2

retryable := {"INFRA_FLAKE"}
never_retry := {"DATA_FAULT", "ENV_FAULT", "REAL_BUG"}
known_classes := retryable | never_retry | {"PASS"}

# Minimum agent tier at which each action may be taken WITHOUT a human.
min_tier := {
	"ALLOW_MERGE": 2,
	"AUTO_RETRY_PIPELINE": 2,
	"ESCALATE_TO_INFRA": 1,
	"QUARANTINE_NODE": 3,
	"BLOCK_MERGE_ALERT_DEV": 1,
	"PAUSE_FOR_MANUAL_REVIEW": 0,
}

# ----------------------------------------------------------------------------
# `decision` is the single entry point: given input.classification,
# input.confidence, input.trust_tier, and optionally input.attempt /
# input.retry_budget, it resolves to one action string. Mirrors decide()'s
# rule order: unknown class -> low confidence -> per-class rule -> tier gate.
# ----------------------------------------------------------------------------
raw_action := "PAUSE_FOR_MANUAL_REVIEW" if {
	not input.classification in known_classes
} else := "PAUSE_FOR_MANUAL_REVIEW" if {
	input.confidence < confidence_floor
} else := "BLOCK_MERGE_ALERT_DEV" if {
	input.classification == "REAL_BUG"
} else := "ALLOW_MERGE" if {
	input.classification == "PASS"
} else := "ESCALATE_TO_INFRA" if {
	input.classification == "INFRA_FLAKE"
	attempt > retry_budget_in
} else := "AUTO_RETRY_PIPELINE" if {
	input.classification == "INFRA_FLAKE"
	attempt <= retry_budget_in
} else := "ESCALATE_TO_INFRA" if {
	input.classification == "DATA_FAULT"
} else := "QUARANTINE_NODE" if {
	input.classification == "ENV_FAULT"
} else := "PAUSE_FOR_MANUAL_REVIEW"

attempt := input.attempt if {
	input.attempt
} else := 1

retry_budget_in := input.retry_budget if {
	input.retry_budget
} else := retry_budget

# Low confidence never acts autonomously, whatever the tier - this mirrors
# the early-return in decide() that skips the tier gate entirely.
autonomous if {
	input.confidence >= confidence_floor
	input.classification in known_classes
	input.trust_tier >= min_tier[raw_action]
}

# Final action: the raw rule's action if the agent's tier clears it and
# confidence is above the floor, otherwise forced to human review - same
# downgrade behaviour as decide()'s `out()` helper.
decision := raw_action if {
	autonomous
} else := "PAUSE_FOR_MANUAL_REVIEW"
