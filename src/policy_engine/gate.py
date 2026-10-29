"""
The release gate: one decision from several independent signals.

    triage decision (S2+S3)   what kind of failure is this, and what do we do?
    security verdict (S6)     can a known CVE actually be reached?
    canary report (S5)        is the new version healthier or sicker than stable?
    drift check (S5)          does the model even understand this system's traces?

Precedence is "most conservative signal wins", encoded as a short, ordered rule
list so the reason a PR was blocked is always one named rule:

    1. security BLOCK            -> BLOCK_MERGE_ALERT_DEV    (security-blocks)
    2. canary ROLLBACK           -> ROLLBACK_CANARY          (canary-rollback)
    3. out-of-distribution trace -> PAUSE_FOR_MANUAL_REVIEW  (ood-abstain)
    4. otherwise the triage decision stands                  (its own rule)
    5. an ALLOW is downgraded to review if security says REVIEW or the canary
       says HOLD                                             (security-review /
                                                              canary-hold)

The trust tier still governs EXECUTION, exactly as in policy.decide(): if the
agent's tier is below the chosen action's minimum, it can only RECOMMEND that
action. `recommended` always carries the intended action; `action` is what the
agent may do without a human.

Like policy.py this is a pure function - no I/O, no model calls. The actuator
(src/actuator) executes; this only decides.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.policy_engine import policy as P


@dataclass
class GateDecision:
    action: str                      # what the agent may do without a human
    recommended: str                 # what the signals say should happen
    autonomous: bool
    rule: str                        # the rule that chose `recommended`
    reasons: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    trust_tier: int = 0


def release_gate(triage: "P.Decision | None" = None, security: dict | None = None,
                 canary=None, drift: dict | None = None,
                 trust_tier: int | None = None) -> GateDecision:
    tier = triage.trust_tier if triage is not None else (trust_tier or 0)
    reasons, signals = [], {}

    sec_v = (security or {}).get("verdict")
    can_v = getattr(canary, "verdict", None)
    in_dist = True if drift is None else drift.get("in_distribution", True)

    if triage is not None:
        signals["triage"] = {"classification": triage.classification,
                             "confidence": triage.confidence,
                             "action": triage.action, "rule": triage.rule}
    if security is not None:
        signals["security"] = {"verdict": sec_v, "rule": security.get("rule"),
                               "reason": security.get("reason")}
    if canary is not None:
        signals["canary"] = {"verdict": can_v, "score": canary.score,
                             "reason": canary.reason}
    if drift is not None:
        signals["drift"] = {"in_distribution": in_dist,
                            "oov_rate": drift.get("oov_rate"),
                            "vocabulary_ok": drift.get("vocabulary_ok", in_dist),
                            "schema_attested": drift.get("schema_attested", True),
                            "note": drift.get("note")}

    # ---- choose the recommended action -----------------------------------
    if sec_v == "BLOCK":
        recommended, rule = P.BLOCK, "security-blocks"
        reasons.append(security["reason"])
    elif can_v == "ROLLBACK":
        recommended, rule = P.ROLLBACK, "canary-rollback"
        reasons.append(canary.reason)
    elif not in_dist:
        recommended, rule = P.REVIEW, "ood-abstain"
        reasons.append(drift.get("note", "trace is out of distribution"))
    elif triage is not None:
        recommended, rule = triage.action, triage.rule
        reasons.append(triage.reason)
    else:
        recommended, rule = P.ALLOW, "no-negative-signal"
        reasons.append("no triage signal supplied and nothing objected")

    # ---- downgrade a clean ALLOW if a softer signal asks for a human -------
    if recommended == P.ALLOW:
        if sec_v == "REVIEW":
            recommended, rule = P.REVIEW, "security-review"
            reasons.append(security["reason"])
        elif can_v == "HOLD":
            recommended, rule = P.REVIEW, "canary-hold"
            reasons.append(canary.reason)

    # ---- trust tier decides whether the agent may do it alone ----------------
    if triage is not None and recommended == triage.action and rule == triage.rule:
        autonomous = triage.autonomous
    else:
        autonomous = recommended != P.REVIEW and tier >= P.MIN_TIER[recommended]
    action = recommended if autonomous or recommended == P.REVIEW else P.REVIEW

    return GateDecision(action=action, recommended=recommended,
                        autonomous=autonomous, rule=rule, reasons=reasons,
                        signals=signals, trust_tier=tier)
