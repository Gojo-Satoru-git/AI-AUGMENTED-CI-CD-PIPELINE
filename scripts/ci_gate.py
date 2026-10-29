"""
S8: the CI gate. Called by .github/workflows/ai-gate.yml on every pull request.

    # in-process (loads the trained model locally)
    python scripts/ci_gate.py --jaeger samples/jaeger_failed_run.json \\
        --security-json reports/security_gate.json

    # against the deployed service (which owns the persistent trust state)
    python scripts/ci_gate.py --api-url https://triage.example.com --spans trace.json

    # actually post the status / comment / retry (needs GITHUB_TOKEN)
    python scripts/ci_gate.py --jaeger trace.json --live

By default nothing is sent anywhere (dry run): the decision and the PR comment
are printed and written to reports/gate_decision.json / gate_comment.md.

Inputs (all optional - the gate decides from whatever signals it is given):
    --jaeger / --spans / --tokens      the failed run's trace
    --security-json                    output of scripts/run_security_gate.py
    --baseline-jaeger/--canary-jaeger  Jaeger exports for the canary comparison

Exit code: 0 only when the agent says ALLOW (or an in-budget RETRY is under
way); anything else leaves the required check red until a human acts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.actuator.actuator import (Actuator, DryRunBackend, GitHubBackend,   # noqa: E402
                                   exit_code_for, render_comment)
from src.observability.jaeger_adapter import jaeger_to_spans, load_jaeger    # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jaeger_span_lists(path):
    return [spans for _tid, spans, _extra in load_jaeger(path)]


def github_pr_number():
    """GATE_PR wins (workflow_run events carry the PR elsewhere); otherwise read
    the event payload for a pull_request or workflow_run event."""
    if os.environ.get("GATE_PR"):
        return int(os.environ["GATE_PR"])
    ev = os.environ.get("GITHUB_EVENT_PATH")
    if ev and Path(ev).exists():
        doc = read_json(ev)
        if (doc.get("pull_request") or {}).get("number"):
            return doc["pull_request"]["number"]
        prs = (doc.get("workflow_run") or {}).get("pull_requests") or []
        if prs:
            return prs[0].get("number")
    return None


def call_api(url, payload):
    req = urllib.request.Request(
        url.rstrip("/") + "/api/v1/gate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        sys.exit(f"gate API unreachable: {e}")


def main():
    # PR comments contain emoji; Windows consoles default to a narrow codepage
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--jaeger"), ap.add_argument("--spans"), ap.add_argument("--tokens")
    ap.add_argument("--security-json")
    ap.add_argument("--baseline-jaeger"), ap.add_argument("--canary-jaeger")
    ap.add_argument("--agent-id", default="triage_agent")
    ap.add_argument("--tier", type=int, default=None,
                    help="override the agent's tier (an ephemeral CI runner has "
                         "no persistent trust DB; prefer --api-url in production)")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--retry-budget", type=int, default=2)
    ap.add_argument("--test-id", default=os.environ.get("GITHUB_SHA", "local")[:12])
    ap.add_argument("--target", help="node/deployment for quarantine/rollback")
    ap.add_argument("--api-url")
    ap.add_argument("--live", action="store_true",
                    help="really post status/comment/retry (needs GITHUB_TOKEN)")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    spans = extra = None
    if args.jaeger:
        traces = list(load_jaeger(args.jaeger))
        _tid, spans, extra = traces[0]
    elif args.spans:
        spans = read_json(args.spans)
    security = None
    if args.security_json and Path(args.security_json).exists():
        security = read_json(args.security_json)["verdict"]
    baseline = jaeger_span_lists(args.baseline_jaeger) if args.baseline_jaeger else None
    canary = jaeger_span_lists(args.canary_jaeger) if args.canary_jaeger else None

    if args.api_url:
        payload = {"test_id": args.test_id, "agent_id": args.agent_id,
                   "attempt": args.attempt, "retry_budget": args.retry_budget,
                   "spans": spans, "trace_sequence": args.tokens,
                   "security": security, "use_security_report": False,
                   "baseline_traces": baseline, "canary_traces": canary}
        result = call_api(args.api_url, payload)
    else:
        from src.policy_engine.pipeline import evaluate
        predictor = None
        if spans or args.tokens:
            from src.models.predict import TriagePredictor
            predictor = TriagePredictor()
        result = evaluate(predictor, spans=spans, tokens=args.tokens, extra=extra,
                          agent_id=args.agent_id, tier=args.tier,
                          attempt=args.attempt, retry_budget=args.retry_budget,
                          security=security, baseline_traces=baseline,
                          canary_traces=canary, test_id=args.test_id,
                          log=args.tier is None)

    if args.live:
        # GATE_* are set by the workflow_run workflow: for that event GITHUB_SHA
        # and GITHUB_RUN_ID describe the default branch, not the PR under test.
        backend = GitHubBackend(
            os.environ.get("GATE_REPO") or os.environ["GITHUB_REPOSITORY"],
            os.environ.get("GATE_SHA") or os.environ["GITHUB_SHA"],
            pr_number=github_pr_number(),
            run_id=os.environ.get("GATE_RUN_ID") or os.environ.get("GITHUB_RUN_ID"))
    else:
        backend = DryRunBackend()
    records = Actuator(backend).execute(result, target=args.target)

    comment = render_comment(result)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "gate_comment.md").write_text(comment, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(comment + "\n")

    print(comment)
    print("\n--- actuation ({}) ---".format("LIVE" if args.live else "dry run"))
    for r in records:
        print(f"  {r.step:<20} executed={r.executed} {r.detail}")
    code = exit_code_for(result["action"])
    print(f"\naction={result['action']}  exit={code}")
    sys.exit(code)


if __name__ == "__main__":
    main()
