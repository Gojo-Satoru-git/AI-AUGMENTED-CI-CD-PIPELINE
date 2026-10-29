"""
The closed loop, as a generator of events.

    for ev in run_loop(predictor=..., agent_id="live_agent"):
        ...                      # ev["type"] is "phase", "step" or "done"

One implementation behind scripts/live_run.py (prints every decision),
scripts/showcase.py (prints a curated narration) and the trajectory chart.

Per request: minishop serves it -> trace -> Jaeger adapter -> triage model ->
drift guard -> policy -> release gate -> decision; then the TRUE outcome (exact,
because the injector knows whether the fault bit) is fed back and the trust tier
may move. In-process by default; pass `api=` to drive a running API instead (the
API then owns the trust state and /metrics shows the same tier).
"""
from __future__ import annotations

import collections
import json
import time
import urllib.request

from src.actuator.actuator import Actuator, DryRunBackend
from src.live.minishop import Fault, Shop
from src.live.tracer import push_otlp
from src.observability.jaeger_adapter import jaeger_to_spans
from src.policy_engine import database as db
from src.policy_engine import policy as P

DEFAULT_MS = 0.5     # time scale; collection and live runs MUST agree (durations are
                     # a model input, so a different scale is train/serve skew)

SCHEDULE = [("none", 40), ("slow_db", 30), ("none", 20), ("flaky_net", 30),
            ("none", 20), ("db_corruption", 30), ("none", 20),
            ("host_down", 30), ("none", 20), ("code_bug", 30), ("none", 20),
            ("novel_bug", 30), ("none", 20),    # an unseen failure that aborts early
            ("silent_bug", 30), ("none", 20)]   # a defect with no trace signature
WORKLOAD = {"none": "checkout", "slow_db": "checkout", "flaky_net": "checkout",
            "db_corruption": "checkout", "host_down": "browse",
            "code_bug": "checkout", "novel_bug": "checkout",
            "silent_bug": "checkout"}


def post(url, payload, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def unsafe(action, autonomous, truth):
    """An autonomous action that the true outcome shows was wrong."""
    if not autonomous:
        return None
    if action == P.ALLOW and truth != "PASS":
        return "unsafe-merge"
    if action == P.RETRY and truth in P.NEVER_RETRY:
        return "wasted-retry"
    if action == P.BLOCK and truth == "PASS":
        return "false-block"
    if action == P.QUARANTINE and truth != "ENV_FAULT":
        return "wrong-quarantine"
    return None


def run_loop(*, predictor=None, api=None, agent_id="live_agent", start_tier=0,
             ms=DEFAULT_MS, seed=2024, schedule=None, scale=1.0, otlp=None,
             pause=0.0, hit_p=0.9, severity=0.8):
    if (predictor is None) == (api is None):
        raise ValueError("pass exactly one of predictor= or api=")
    api = api.rstrip("/") if api else None
    if predictor is not None:
        from src.policy_engine.pipeline import evaluate
        db.init_db()
        db.RETENTION_ENABLED = True
        with db._conn() as c:
            for t in ("decisions", "tier_history", "agent_trust"):
                c.execute(f"DELETE FROM {t} WHERE agent_id=?", (agent_id,))
        db.set_agent_tier(agent_id, start_tier, f"live run start at T{start_tier}")
        with db._conn() as c:
            c.execute("DELETE FROM tier_history WHERE agent_id=?", (agent_id,))
    else:
        # the API owns the trust state: set this run's starting tier through it
        post(api + "/api/v1/agent/tier", {"agent_id": agent_id, "tier": start_tier,
                                          "evidence": f"live run start at T{start_tier}"})

    shop = Shop(seed=seed, ms=ms)
    backend = DryRunBackend()
    actuator = Actuator(backend)
    phases, tier_events, push_err = [], [], None
    tier, step = start_tier, 0

    for pi, (kind, n) in enumerate(schedule or SCHEDULE):
        n = max(1, int(n * scale))
        shop.set_fault(Fault(kind, hit_p=hit_p, severity=severity))
        ph = {"fault": kind, "n": 0, "correct": 0, "truth": collections.Counter(),
              "actions": collections.Counter(), "unsafe": 0, "abstained": 0}
        yield {"type": "phase", "index": pi, "fault": kind, "n": n, "first_step": step + 1}
        for _ in range(n):
            tr = shop.handle(WORKLOAD[kind])
            truth = Shop.label(tr)
            if otlp and push_err is None:
                push_err = push_otlp(otlp, tr)
            spans, extra = jaeger_to_spans(tr.to_jaeger())
            step += 1
            if api:
                res = post(api + "/api/v1/gate", {
                    "test_id": tr.trace_id, "agent_id": agent_id, "spans": spans,
                    "extra": extra, "use_security_report": False})
                fb = post(api + "/api/v1/triage/ground-truth",
                          {"decision_id": res["decision_id"], "ground_truth": truth})
            else:
                res = evaluate(predictor, spans=spans, extra=extra,
                               agent_id=agent_id, test_id=tr.trace_id)
                fb = db.record_ground_truth(res["decision_id"], truth)
            tier_before = tier
            ta = fb.get("tier_action")
            if ta in ("promoted", "demoted"):
                tier = fb["to_tier"]
                tier_events.append({"step": step, "action": ta,
                                    "from": fb.get("from_tier"), "to": fb["to_tier"],
                                    "reason": fb.get("reason", "")})
            else:
                tier = res["trust_tier"]
            pred = res["classification"]
            ok = pred == truth
            u = unsafe(res["action"], res["autonomous"], truth)
            ph["n"] += 1
            ph["correct"] += ok
            ph["truth"][truth] += 1
            ph["actions"][res["action"]] += 1
            ph["unsafe"] += bool(u)
            ph["abstained"] += res["rule"] == "ood-abstain"
            if res["autonomous"]:
                actuator.execute(res)
            yield {"type": "step", "step": step, "phase": pi, "fault": kind,
                   "truth": truth, "pred": pred, "conf": res["confidence"],
                   "action": res["action"], "rule": res["rule"],
                   "autonomous": res["autonomous"], "ok": ok, "unsafe": u,
                   "tier_acted_at": res["trust_tier"], "tier": tier,
                   "tier_before": tier_before, "tier_action": ta,
                   "from_tier": fb.get("from_tier"), "to_tier": fb.get("to_tier"),
                   "reason": fb.get("reason", ""), "trace_id": tr.trace_id,
                   "n_spans": len(tr.spans)}
            if pause:
                time.sleep(pause)
        phases.append(ph)

    total = sum(p["n"] for p in phases)
    right = sum(p["correct"] for p in phases)
    yield {"type": "done", "requests": total,
           "agent_correct": round(right / total, 4),
           "unsafe_autonomous_actions": sum(p["unsafe"] for p in phases),
           "autonomous_actuations": sum(1 for c, _ in backend.calls if c == "set_status"),
           "start_tier": start_tier, "end_tier": tier, "tier_events": tier_events,
           "push_err": push_err, "otlp": bool(otlp),
           "phases": [{**p, "truth": dict(p["truth"]), "actions": dict(p["actions"])}
                      for p in phases]}
