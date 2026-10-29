"""
A browser dashboard for the live system - a control room that shows every module.

    python scripts/serve_ui.py          # then open http://127.0.0.1:8765

A background thread keeps serving traffic to minishop. Every request goes through
the real pipeline (trace -> Jaeger adapter -> drift guard -> triage model -> RCA ->
trust-tier policy -> security / canary signals -> release gate -> dry-run actuator),
and the true outcome is fed back so the agent's trust tier moves. Tabs expose the
rest: canary releases, the security gate, the real HDFS model, chaos, metrics.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.actuator.actuator import Actuator, DryRunBackend, render_comment
from src.live.loop import DEFAULT_MS, SCHEDULE, WORKLOAD, unsafe
from src.live.minishop import Fault, Shop
from src.live.tracer import push_otlp
from src.observability.canary import canary_health
from src.observability.jaeger_adapter import jaeger_to_spans, span_status
from src.observability.metrics import render_metrics
from src.policy_engine import database as db
from src.policy_engine.pipeline import evaluate
from src.rca import localize

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
FAULTS = ["none", "slow_db", "flaky_net", "db_corruption", "host_down",
          "code_bug", "novel_bug", "silent_bug"]
MAX_SERIES = 1500
HDFS_SAMPLES = {"pass": "jaeger_hdfs_pass.json", "infra_flake": "jaeger_hdfs_infra_flake.json",
                "data_fault": "jaeger_hdfs_data_fault.json", "env_fault": "jaeger_hdfs_env_fault.json",
                "foreign": "jaeger_foreign_checkout.json"}
VULN = {"verdict": "BLOCK", "rule": "reachable-critical-cve",
        "reason": "CVE in `requests` is reachable from the running service (simulated finding)"}


def _read(path):
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def rca_record(spans, trace_id):
    """adapter spans -> the corpus-record shape src/rca/localize.py scores."""
    return {"task": trace_id, "scen": "live",
            "toks": " ".join(f"{sp['op'].replace(' ', '_')}|{sp['status']}" for sp in spans),
            "dur": ",".join(str(sp["duration"]) for sp in spans),
            "dep": ",".join(str(sp["depth"]) for sp in spans)}


def rca(spans, trace_id, baseline, cls, action):
    rec = rca_record(spans, trace_id)
    return {"suspects": localize.top_k(rec, baseline, 5),
            "markdown": localize.generate_postmortem(rec, baseline, cls, action)}


def trace_tree(tr):
    kids, roots = {}, []
    for s in tr.spans:
        par = s["references"][0]["spanID"] if s["references"] else None
        (kids.setdefault(par, []).append(s)) if par else roots.append(s)
    svc = {pid: name for name, pid in tr.services.items()}
    out = []

    def walk(s, depth):
        out.append({"depth": depth, "svc": svc[s["processID"]], "op": s["operationName"],
                    "ms": round(s["duration"] / 1000, 1), "status": span_status(s)})
        for c in sorted(kids.get(s["spanID"], []), key=lambda x: x["startTime"]):
            walk(c, depth + 1)

    for r in roots:
        walk(r, 0)
    return out


class Session:
    def __init__(self, predictor, agent_id="ui_agent", ms=DEFAULT_MS, otlp=None, seed=7):
        self.predictor, self.agent_id, self.ms, self.otlp = predictor, agent_id, ms, otlp
        self.seed = seed
        self.lock = threading.Lock()
        self.rate = 4.0
        self.fault = "none"
        self.schedule = []            # remaining [kind, n] of a scripted scenario
        self.scenario_active = False
        self.paused = False
        self.stop = False
        self.push_err = None
        self.security_mode = "clean"  # clean | vuln
        self.chaos = False
        self.last_canary = None
        corpus = ROOT / "data/processed/corpus_live.jsonl.gz"
        self.baseline = localize.build_baseline(corpus) if corpus.exists() else None
        hb = ROOT / "src/models/saved/op_latency_baseline.json"
        self.hdfs_baseline = localize.load_baseline(hb) if hb.exists() else None
        self._hdfs = None
        self.security = (_read("reports/security_gate.json") or {}).get("verdict") or \
            {"verdict": "ALLOW", "rule": "no-security-report", "reason": "no report"}
        self.reset(0)
        threading.Thread(target=self._run, daemon=True).start()

    # ------------------------------------------------------------------ control
    def reset(self, start_tier):
        with self.lock:
            db.init_db()
            db.RETENTION_ENABLED = True
            with db._conn() as c:
                for t in ("decisions", "tier_history", "agent_trust"):
                    c.execute(f"DELETE FROM {t} WHERE agent_id=?", (self.agent_id,))
            db.set_agent_tier(self.agent_id, start_tier, f"ui start at T{start_tier}")
            with db._conn() as c:
                c.execute("DELETE FROM tier_history WHERE agent_id=?", (self.agent_id,))
            self.shop = Shop(seed=self.seed, ms=self.ms)
            self.backend = DryRunBackend()
            self.actuator = Actuator(self.backend)
            self.start_tier, self.tier = start_tier, start_tier
            self.n = self.correct = self.unsafe_n = 0
            self.steps = collections.deque(maxlen=60)
            self.series = []          # [tier, fault index, ok, unsafe]
            self.events = []
            self.act_log = []
            self.incident = None
            self.last_comment = None
            self.latest = None
            self.fault, self.schedule, self.scenario_active = "none", [], False

    def set_fault(self, kind):
        with self.lock:
            self.fault, self.schedule, self.scenario_active = kind, [], False
            self.chaos = False

    def start_scenario(self):
        with self.lock:
            self.schedule = [list(x) for x in SCHEDULE]
            self.scenario_active = True
            self.chaos = False

    def _sec(self):
        return VULN if self.security_mode == "vuln" else self.security

    # ------------------------------------------------------------------ worker
    def _next_kind(self):
        if self.scenario_active:
            if not self.schedule:
                self.scenario_active, self.fault = False, "none"
            else:
                self.fault = self.schedule[0][0]
                self.schedule[0][1] -= 1
                if self.schedule[0][1] <= 0:
                    self.schedule.pop(0)
        elif self.chaos and self.n % 25 == 0:          # chaos monkey: a fault, then calm
            self.fault = random.choice(FAULTS[1:]) if (self.n // 25) % 2 == 0 else "none"
        return self.fault

    def _run(self):
        while not self.stop:
            if self.paused:
                time.sleep(0.2)
                continue
            t0 = time.time()
            try:
                self._one()
            except Exception as e:                       # keep the dashboard alive
                self.latest = {"error": repr(e)}
                time.sleep(1)
            time.sleep(max(0.0, 1.0 / self.rate - (time.time() - t0)))

    def _one(self):
        with self.lock:
            kind = self._next_kind()
            self.shop.set_fault(Fault(kind, hit_p=0.9, severity=0.8))
            tr = self.shop.handle(WORKLOAD[kind])
            truth = Shop.label(tr)
            if self.otlp and self.push_err is None:
                self.push_err = push_otlp(self.otlp, tr)
            spans, extra = jaeger_to_spans(tr.to_jaeger())
            res = evaluate(self.predictor, spans=spans, extra=extra, security=self._sec(),
                           agent_id=self.agent_id, test_id=tr.trace_id)
            fb = db.record_ground_truth(res["decision_id"], truth)
            ta = fb.get("tier_action")
            self.tier = fb["to_tier"] if ta in ("promoted", "demoted") else res["trust_tier"]
            ok = res["classification"] == truth
            u = None if res["rule"] == "security-blocks" else \
                unsafe(res["action"], res["autonomous"], truth)
            if res["autonomous"]:
                self.actuator.execute(res)
                self.act_log.append(f"#{self.n + 1} {res['action']} ({res['classification']})")
                self.act_log = self.act_log[-8:]
            self.n += 1
            self.correct += ok
            self.unsafe_n += bool(u)
            step = {"step": self.n, "fault": kind, "truth": truth,
                    "pred": res["classification"], "conf": round(res["confidence"], 2),
                    "action": res["action"], "rule": res["rule"],
                    "autonomous": res["autonomous"], "ok": ok, "unsafe": u,
                    "tier": self.tier, "tier_action": ta}
            self.steps.appendleft(step)
            self.series.append([self.tier, FAULTS.index(kind), int(ok), int(bool(u))])
            self.series = self.series[-MAX_SERIES:]
            if ta in ("promoted", "demoted"):
                self.events.append({"step": self.n, "action": ta, "from": fb.get("from_tier"),
                                    "to": fb["to_tier"], "reason": fb.get("reason", "")})
            self.latest = {"step": self.n, "fault": kind, "truth": truth,
                           "pred": res["classification"], "conf": round(res["confidence"], 2),
                           "action": res["action"], "rule": res["rule"], "ok": ok,
                           "autonomous": res["autonomous"], "tree": trace_tree(tr),
                           "trace_id": tr.trace_id, "signals": res["signals"],
                           "reasons": res["reasons"], "recommended": res["recommended"],
                           "probs": res["probabilities"]}
            self.last_comment = render_comment(res)
            if self.baseline and (truth != "PASS" or res["classification"] != "PASS"):
                self.incident = {"step": self.n, "truth": truth, "fault": kind,
                                 "pred": res["classification"], "action": res["action"],
                                 **rca(spans, tr.trace_id, self.baseline,
                                       res["classification"], res["action"])}

    # ------------------------------------------------------------------ modules
    def run_canary(self, kind, n=40):
        base, can = Shop(seed=101, ms=self.ms), Shop(seed=202, ms=self.ms)
        base.set_fault(Fault("none"))
        can.set_fault(Fault(kind, hit_p=0.9, severity=0.8))

        def spans_of(shop):
            return [jaeger_to_spans(shop.handle("checkout").to_jaeger())[0] for _ in range(n)]
        b, c = spans_of(base), spans_of(can)
        res = evaluate(None, security=self._sec(), baseline_traces=b, canary_traces=c,
                       agent_id=self.agent_id, tier=self.tier, test_id=f"canary:{kind}", log=False)
        rep = dataclasses.asdict(canary_health(b, c))
        self.last_canary = {"kind": kind, "verdict": rep["verdict"]}
        return {"kind": kind, "report": rep,
                "gate": {k: res[k] for k in ("action", "recommended", "autonomous", "rule",
                                              "reasons", "signals", "trust_tier")},
                "comment": render_comment(res)}

    def hdfs_predictor(self):
        if self._hdfs is None:
            from src.models.predict import TriagePredictor
            d = ROOT / "src/models/saved"
            self._hdfs = TriagePredictor(ckpt=d / "triage_lstm.pth", vocab_path=d / "vocab.json")
        return self._hdfs

    def run_hdfs(self, name):
        doc = json.loads((ROOT / "samples" / HDFS_SAMPLES[name]).read_text(encoding="utf-8"))
        trace = doc["data"][0] if "data" in doc else doc
        spans, extra = jaeger_to_spans(trace)
        res = evaluate(self.hdfs_predictor(), spans=spans, extra=extra, security=self._sec(),
                       agent_id="hdfs_demo", tier=self.tier, test_id=f"hdfs:{name}", log=False)
        out = {"sample": name, "n_spans": len(spans), "classification": res["classification"],
               "confidence": res["confidence"], "action": res["action"], "rule": res["rule"],
               "autonomous": res["autonomous"], "oov_rate": res["oov_rate"],
               "signals": res["signals"], "reasons": res["reasons"], "probs": res["probabilities"],
               "evidence": res["evidence_spans"], "comment": render_comment(res)}
        if self.hdfs_baseline and res["classification"]:
            out["rca"] = rca(spans, name, self.hdfs_baseline, res["classification"], res["action"])
        return out

    # ------------------------------------------------------------------ view
    def state(self):
        with self.lock:
            recent = list(self.series[-50:])
            return {
                "agent": self.agent_id, "tier": self.tier, "start_tier": self.start_tier,
                "n": self.n, "accuracy": round(self.correct / self.n, 4) if self.n else None,
                "recent_accuracy": (round(sum(r[2] for r in recent) / len(recent), 3)
                                    if recent else None),
                "unsafe": self.unsafe_n, "actuations": self.act_log[-6:],
                "n_actuations": len(self.backend.calls),
                "fault": self.fault, "scenario_active": self.scenario_active,
                "scenario_left": sum(x[1] for x in self.schedule),
                "paused": self.paused, "rate": self.rate, "faults": FAULTS,
                "steps": list(self.steps)[:25], "series": self.series, "events": self.events,
                "latest": self.latest, "incident": self.incident, "comment": self.last_comment,
                "security_mode": self.security_mode, "security": self._sec(),
                "chaos": self.chaos, "canary": self.last_canary,
                "otlp": bool(self.otlp), "push_err": self.push_err,
            }


class FaultReq(BaseModel):
    kind: str


class ResetReq(BaseModel):
    start_tier: int = 0


class RateReq(BaseModel):
    rate: float


def create_app(session: Session) -> FastAPI:
    app = FastAPI(title="Live triage dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    def state():
        return session.state()

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return render_metrics()

    @app.get("/api/reports")
    def reports():
        base = _read("reports/baselines.json") or {}
        b5 = _read("reports/baseline_B5.json")
        return {
            "evaluation": _read("reports/evaluation.json"),
            "live_evaluation": _read("reports/live/evaluation.json"),
            "baselines": {**{k: v["macro_f1"] for k, v in base.items()},
                          **({"B5_scalars_plus_ops_gbm": b5["macro_f1"]} if b5 else {})},
            "rca": _read("reports/rca_evaluation.json"),
            "fidelity": _read("reports/adapter_fidelity.json"),
            "chaos": _read("reports/chaos_report.json"),
            "security": _read("reports/security_gate.json") or {},
        }

    @app.post("/api/fault")
    def fault(r: FaultReq):
        if r.kind not in FAULTS:
            return {"ok": False}
        session.set_fault(r.kind)
        return {"ok": True}

    @app.post("/api/scenario")
    def scenario():
        session.start_scenario()
        return {"ok": True}

    @app.post("/api/reset")
    def reset(r: ResetReq):
        session.reset(max(0, min(3, r.start_tier)))
        return {"ok": True}

    @app.post("/api/rate")
    def rate(r: RateReq):
        session.rate = max(0.5, min(30.0, r.rate))
        return {"ok": True}

    @app.post("/api/pause")
    def pause():
        session.paused = not session.paused
        return {"paused": session.paused}

    @app.post("/api/chaos")
    def chaos():
        session.chaos = not session.chaos
        if session.chaos:
            session.scenario_active, session.schedule = False, []
        return {"chaos": session.chaos}

    @app.post("/api/security")
    def security(r: FaultReq):
        session.security_mode = "vuln" if r.kind == "vuln" else "clean"
        return {"ok": True}

    @app.post("/api/security/run")
    def security_run():
        p = subprocess.run([sys.executable, "scripts/run_security_gate.py"], cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        session.security = (_read("reports/security_gate.json") or {}).get("verdict", session.security)
        return {"ok": True, "tail": p.stdout[-1500:]}

    @app.post("/api/canary")
    def canary(r: FaultReq):
        return session.run_canary(r.kind if r.kind in FAULTS else "none")

    @app.post("/api/hdfs")
    def hdfs(r: FaultReq):
        if r.kind not in HDFS_SAMPLES:
            return {"error": "unknown sample"}
        return session.run_hdfs(r.kind)

    app.mount("/reports", StaticFiles(directory=ROOT / "reports"), name="reports")
    return app
