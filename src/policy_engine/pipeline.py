"""
One entry point that runs the whole decision pipeline:

    spans/tokens -> triage model -> drift guard -> policy.decide (trust tier)
                 -> + security verdict + canary health -> release_gate
                 -> logged decision

Used by the FastAPI service (/api/v1/gate), the CI script (scripts/ci_gate.py)
and the demo, so all three behave identically. It DECIDES and records the
decision; it never executes anything - that is src/actuator's job.
"""
from __future__ import annotations

from src.observability.canary import canary_health
from src.observability.jaeger_adapter import drift_check
from src.policy_engine import database as db
from src.policy_engine import policy as P
from src.policy_engine.gate import release_gate


def _attested(predictor, extra) -> bool:
    """A trace's task-level features are trustworthy if either (a) the exporter
    attests the TraceBench training schema, or (b) the model was trained on the
    Jaeger adapter's own feature definitions and this trace came from that
    adapter - there is then no train/serve skew."""
    extra = extra or {}
    if extra.get("schema_attested"):
        return True
    return bool(extra.get("from_adapter")
                and getattr(predictor, "feature_schema", "") == "jaeger_adapter_v1")


def evaluate(predictor=None, *, spans=None, tokens=None, extra=None,
             agent_id: str = "triage_agent", tier: int | None = None,
             attempt: int = 1, retry_budget: int = P.DEFAULT_RETRY_BUDGET,
             security: dict | None = None, baseline_traces=None,
             canary_traces=None, test_id: str = "adhoc", log: bool = True,
             top_spans: int = 5) -> dict:
    pred = triage = drift = None
    if spans or tokens:
        if predictor is None:
            raise ValueError("a predictor is required to triage a trace")
        pred = predictor.predict(tokens=tokens, spans=spans, extra=extra,
                                 top_spans=top_spans)
        drift = drift_check(pred, schema_attested=_attested(predictor, extra))

    if tier is None:
        tier = db.get_agent_tier(agent_id)

    if pred is not None:
        triage = P.decide(pred["classification"], pred["confidence"], tier,
                          attempt=attempt, retry_budget=retry_budget)

    canary = None
    if canary_traces:
        canary = canary_health(baseline_traces or [], canary_traces)

    gate = release_gate(triage, security, canary, drift, trust_tier=tier)

    decision_id = None
    if log:
        db.init_db()
        decision_id = db.log_decision(
            agent_id, test_id,
            pred["classification"] if pred else "N/A",
            pred["confidence"] if pred else 1.0,
            gate.action, gate.rule, tier, gate.autonomous)

    return {
        "decision_id": decision_id,
        "test_id": test_id,
        "agent_id": agent_id,
        "trust_tier": tier,
        "action": gate.action,
        "recommended": gate.recommended,
        "autonomous": gate.autonomous,
        "rule": gate.rule,
        "reasons": gate.reasons,
        "signals": gate.signals,
        "classification": pred["classification"] if pred else None,
        "confidence": pred["confidence"] if pred else None,
        "probabilities": pred["probabilities"] if pred else None,
        "evidence_spans": pred["evidence_spans"] if pred else None,
        "oov_rate": pred["oov_rate"] if pred else None,
        "attempt": attempt,
        "retry_budget": retry_budget,
    }
