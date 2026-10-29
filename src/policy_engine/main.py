"""
Test-Triage API.

Ingests a trace, classifies it into the five operational classes, applies the
trust-tiered policy, and returns a decision the deployment actuator can act on
together with the evidence spans behind it.

The decision engine never executes anything - it only decides. Execution is the
actuator's job (workstream S8). Keeping those apart is what makes the trust
tiers meaningful.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policy_engine import policy as P                      # noqa: E402
from src.policy_engine.database import (                       # noqa: E402
    agent_scorecard, get_agent_tier, init_db, log_decision,
    record_ground_truth, set_agent_tier)
from src.observability.jaeger_adapter import jaeger_to_spans   # noqa: E402
from src.observability.metrics import render_metrics           # noqa: E402
from src.policy_engine.pipeline import evaluate as run_pipeline  # noqa: E402

app = FastAPI(
    title="AI Test-Triage Policy Engine",
    version="2.0",
    description="Five-class trace triage with trust-tiered policy decisions.",
)

_predictor = None
_rca_baseline = None
RCA_BASELINE_PATH = ROOT / "src" / "models" / "saved" / "op_latency_baseline.json"


def predictor():
    """Loaded lazily so the API imports even before a model is trained."""
    global _predictor
    if _predictor is None:
        from src.models.predict import TriagePredictor
        mdir = os.environ.get("TRIAGE_MODEL_DIR")
        if mdir:       # e.g. a model trained on this system's own Jaeger traces
            _predictor = TriagePredictor(
                ckpt=Path(mdir) / "triage_lstm.pth",
                vocab_path=Path(mdir) / "vocab.json")
        else:
            _predictor = TriagePredictor()
    return _predictor


def rca_baseline():
    """Loaded lazily; built by scripts/run_rca_eval.py from PASS-class
    traces in the corpus (see src/rca/localize.py)."""
    global _rca_baseline
    if _rca_baseline is None:
        from src.rca.localize import load_baseline
        _rca_baseline = load_baseline(RCA_BASELINE_PATH)
    return _rca_baseline


# Initialise at import, not on the startup event: TestClient(app) used without
# a context manager never fires startup hooks, and a half-initialised database
# is a worse failure than a redundant CREATE TABLE IF NOT EXISTS.
init_db()


# ---------------------------------------------------------------------------
class Span(BaseModel):
    op: str
    status: str = "OK"
    duration: int = 0
    host: str = ""
    agent: str = ""
    depth: int = 0


class TriageRequest(BaseModel):
    test_id: str
    trace_sequence: Optional[str] = Field(
        None, description='Pre-tokenised, e.g. "getFileInfo|OK RPC:addBlock|IOEXC"')
    spans: Optional[list[Span]] = Field(
        None, description="Raw spans, as the Jaeger adapter emits")
    agent_id: str = "triage_agent"
    attempt: int = 1
    retry_budget: int = P.DEFAULT_RETRY_BUDGET


class GroundTruth(BaseModel):
    decision_id: int
    ground_truth: str


class TierUpdate(BaseModel):
    agent_id: str = "triage_agent"
    tier: int
    evidence: str = ""


# ---------------------------------------------------------------------------
def _triage(req: TriageRequest) -> dict:
    if not req.trace_sequence and not req.spans:
        raise HTTPException(422, "provide either trace_sequence or spans")
    try:
        pred = predictor().predict(
            tokens=req.trace_sequence,
            spans=[s.model_dump() for s in req.spans] if req.spans else None)
    except FileNotFoundError as e:
        raise HTTPException(
            503, f"no trained model available: {e}. Run src/models/train.py"
        ) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    tier = get_agent_tier(req.agent_id)
    d = P.decide(pred["classification"], pred["confidence"], tier,
                 attempt=req.attempt, retry_budget=req.retry_budget)
    decision_id = log_decision(
        req.agent_id, req.test_id, d.classification, d.confidence, d.action,
        d.rule, d.trust_tier, d.autonomous)

    return {
        "decision_id": decision_id,
        "test_id": req.test_id,
        "agent_id": req.agent_id,
        "classification": d.classification,
        "confidence": d.confidence,
        "probabilities": pred["probabilities"],
        "action": d.action,
        "rule": d.rule,
        "reason": d.reason,
        "autonomous": d.autonomous,
        "trust_tier": d.trust_tier,
        "retryable": d.classification in P.RETRYABLE,
        "n_spans": pred["n_spans"],
        "oov_rate": pred["oov_rate"],
        "evidence_spans": pred["evidence_spans"],
    }


@app.post("/api/v1/triage")
def triage(req: TriageRequest):
    return _triage(req)


@app.post("/api/v1/triage/ground-truth")
def submit_ground_truth(gt: GroundTruth):
    """Close the loop: once CI knows what the failure really was, record it.
    This runs the T0-T3 promotion/demotion check and returns whatever tier
    action it triggered, if any."""
    result = record_ground_truth(gt.decision_id, gt.ground_truth)
    return {"ok": True, "decision_id": gt.decision_id, **result}


@app.get("/api/v1/agent/scorecard")
def scorecard(agent_id: str = "triage_agent",
              last_n: int = Query(200, ge=1, le=5000)):
    return agent_scorecard(agent_id, last_n)


@app.post("/api/v1/agent/tier")
def update_tier(u: TierUpdate):
    if not 0 <= u.tier <= 3:
        raise HTTPException(422, "tier must be 0-3")
    set_agent_tier(u.agent_id, u.tier, u.evidence)
    return agent_scorecard(u.agent_id)


class PostmortemRequest(BaseModel):
    test_id: str
    trace_sequence: Optional[str] = None
    spans: Optional[list[Span]] = Field(
        None, description="Raw spans - required for a meaningful ranking, "
        "since localization scores duration against a per-op baseline; "
        "trace_sequence alone carries no timing.")
    top_n: int = 5


@app.post("/api/v1/postmortem")
def postmortem(req: PostmortemRequest):
    """S4: rank the spans most likely responsible for this trace's failure,
    against a PASS-class latency baseline (src/rca/localize.py), and return
    a ready-to-paste Markdown postmortem alongside the ranked list."""
    if not req.trace_sequence and not req.spans:
        raise HTTPException(422, "provide either trace_sequence or spans")
    if not RCA_BASELINE_PATH.exists():
        raise HTTPException(
            503, "no RCA baseline built yet. Run scripts/run_rca_eval.py")

    from src.models.predict import spans_to_tokens
    from src.rca.localize import generate_postmortem, top_k

    if req.spans:
        toks, dur, hr, ag, dep = spans_to_tokens(
            [s.model_dump() for s in req.spans])
    else:
        toks = req.trace_sequence.split(" ")
        dur = [0] * len(toks)
        dep = [0] * len(toks)
    record = {"task": req.test_id, "scen": "live", "cls": "?",
              "toks": " ".join(toks), "dur": ",".join(map(str, dur)),
              "dep": ",".join(map(str, dep))}

    pred = predictor().predict(
        tokens=req.trace_sequence,
        spans=[s.model_dump() for s in req.spans] if req.spans else None)
    # trust_tier=3 here is display-only, for "what action does this
    # classification imply" in the postmortem text - the actual decision (at
    # the agent's real tier) was already made by /api/v1/triage.
    d = P.decide(pred["classification"], pred["confidence"], trust_tier=3)
    suspects = top_k(record, rca_baseline(), req.top_n)
    md = generate_postmortem(record, rca_baseline(), pred["classification"],
                             d.action, req.top_n)
    return {"test_id": req.test_id, "classification": pred["classification"],
            "suspects": suspects, "postmortem_markdown": md}



# ---------------------------------------------------------------------------
# S5/S6/S8: Jaeger ingestion, the full release gate, metrics, security verdict
# ---------------------------------------------------------------------------
SECURITY_REPORT = ROOT / "reports" / "security_gate.json"


def _security_verdict():
    if not SECURITY_REPORT.exists():
        return None
    import json
    return json.loads(SECURITY_REPORT.read_text(encoding="utf-8"))["verdict"]


def _first_jaeger_trace(doc: dict) -> dict:
    if "data" in doc:
        if not doc["data"]:
            raise HTTPException(422, "jaeger document contains no traces")
        return doc["data"][0]
    return doc


class JaegerTriageRequest(BaseModel):
    test_id: str
    trace: dict = Field(..., description="a Jaeger trace object, or the "
                        "{'data': [...]} envelope from GET /api/traces/<id>")
    agent_id: str = "triage_agent"
    attempt: int = 1
    retry_budget: int = P.DEFAULT_RETRY_BUDGET


@app.post("/api/v1/triage/jaeger")
def triage_jaeger(req: JaegerTriageRequest):
    """Triage a live Jaeger trace. The result carries an out-of-distribution
    guard: a trace from a system the model was not trained on is routed to a
    human instead of being classified with false confidence."""
    try:
        spans, extra = jaeger_to_spans(_first_jaeger_trace(req.trace))
        return run_pipeline(predictor(), spans=spans, extra=extra,
                            agent_id=req.agent_id, attempt=req.attempt,
                            retry_budget=req.retry_budget, test_id=req.test_id)
    except FileNotFoundError as e:
        raise HTTPException(503, f"no trained model available: {e}") from e
    except (ValueError, KeyError) as e:
        raise HTTPException(422, f"bad trace: {e}") from e


class GateRequest(BaseModel):
    test_id: str
    trace_sequence: Optional[str] = None
    spans: Optional[list[Span]] = None
    jaeger: Optional[dict] = None
    extra: Optional[dict] = Field(
        None, description="training-schema task counts (n_tid, n_host, n_edge, "
        "n_rep, lat) plus \"schema_attested\": true. Without it a raw-span "
        "trace is treated as out-of-distribution - see "
        "reports/adapter_fidelity.json for why")
    baseline_traces: Optional[list[list[Span]]] = Field(
        None, description="stable-version traces, for the canary comparison")
    canary_traces: Optional[list[list[Span]]] = None
    agent_id: str = "triage_agent"
    attempt: int = 1
    retry_budget: int = P.DEFAULT_RETRY_BUDGET
    use_security_report: bool = True
    security: Optional[dict] = Field(
        None, description="a security verdict computed by the caller (e.g. the "
        "CI runner scanning the PR's own dependencies); overrides the server's "
        "stored report")


@app.post("/api/v1/gate")
def gate(req: GateRequest):
    """The full release gate: triage + security + canary + drift -> one action."""
    spans, extra = None, None
    if req.jaeger:
        try:
            spans, extra = jaeger_to_spans(_first_jaeger_trace(req.jaeger))
        except (ValueError, KeyError) as e:
            raise HTTPException(422, f"bad jaeger trace: {e}") from e
    elif req.spans:
        spans = [s.model_dump() for s in req.spans]
    try:
        return run_pipeline(
            predictor() if (spans or req.trace_sequence) else None,
            spans=spans, tokens=req.trace_sequence, extra=extra or req.extra,
            agent_id=req.agent_id, attempt=req.attempt,
            retry_budget=req.retry_budget,
            security=req.security or (_security_verdict()
                                      if req.use_security_report else None),
            baseline_traces=[[s.model_dump() for s in t]
                             for t in (req.baseline_traces or [])],
            canary_traces=[[s.model_dump() for s in t]
                           for t in (req.canary_traces or [])],
            test_id=req.test_id)
    except FileNotFoundError as e:
        raise HTTPException(503, f"no trained model available: {e}") from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/v1/security/verdict")
def security_verdict():
    v = _security_verdict()
    if v is None:
        raise HTTPException(
            404, "no security report yet. Run scripts/run_security_gate.py")
    return v


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus scrape endpoint (see infra/prometheus.yml)."""
    return render_metrics()


@app.get("/api/v1/health")
def health():
    ckpt = ROOT / "src" / "models" / "saved" / "triage_lstm.pth"
    return {
        "status": "ok",
        "model_present": ckpt.exists(),
        "classes": list(P.RETRYABLE | P.NEVER_RETRY | {"PASS"}),
        "actions": sorted(P.MIN_TIER),
    }


if __name__ == "__main__":
    import uvicorn
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
