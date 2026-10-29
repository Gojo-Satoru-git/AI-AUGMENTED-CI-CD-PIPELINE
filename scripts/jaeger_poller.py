"""
S5: feed live Jaeger traces to the triage agent.

    python scripts/jaeger_poller.py --jaeger http://127.0.0.1:16686 \\
        --service frontend --once
    python scripts/jaeger_poller.py --jaeger http://127.0.0.1:16686 \\
        --service frontend --interval 30 --only-errors

For each trace Jaeger has not shown us before (state kept in a small JSON file),
POST it to the agent's /api/v1/triage/jaeger and print the decision. Traces from
a system the model was not trained on come back as `ood-abstain` - that is the
honest outcome, and exactly what to expect from HotROD or TrainTicket until a
model is trained on that system's traces (USER_TODO.md, "Live trace -> gate").

Only the standard library is used, so this runs anywhere Python does.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _get(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_traces(jaeger, service, limit=20, lookback="1h"):
    q = urllib.parse.urlencode({"service": service, "limit": limit,
                                "lookback": lookback})
    return _get(f"{jaeger.rstrip('/')}/api/traces?{q}").get("data", [])


def has_error(trace):
    for s in trace.get("spans", []):
        for t in s.get("tags", []) or []:
            if t.get("key") == "error" and str(t.get("value")).lower() == "true":
                return True
    return False


def triage_trace(api, trace, agent_id="triage_agent", attempt=1):
    body = json.dumps({"test_id": trace.get("traceID", "unknown"),
                       "trace": trace, "agent_id": agent_id,
                       "attempt": attempt}).encode()
    req = urllib.request.Request(
        f"{api.rstrip('/')}/api/v1/triage/jaeger", data=body, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "test_id": trace.get("traceID")}


def poll_once(jaeger, api, service, seen, only_errors=False, limit=20,
              lookback="1h", agent_id="triage_agent"):
    results = []
    for tr in fetch_traces(jaeger, service, limit, lookback):
        tid = tr.get("traceID")
        if tid in seen or (only_errors and not has_error(tr)):
            continue
        seen.add(tid)
        results.append(triage_trace(api, tr, agent_id))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jaeger", default="http://127.0.0.1:16686")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--service", required=True)
    ap.add_argument("--agent-id", default="triage_agent")
    ap.add_argument("--only-errors", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--lookback", default="1h")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--state", default=".jaeger_poller_seen.json")
    args = ap.parse_args()

    state = Path(args.state)
    seen = set(json.loads(state.read_text())) if state.exists() else set()
    while True:
        try:
            results = poll_once(args.jaeger, args.api, args.service, seen,
                                args.only_errors, args.limit, args.lookback,
                                args.agent_id)
        except urllib.error.URLError as e:
            sys.exit(f"cannot reach Jaeger/API: {e.reason}")
        for r in results:
            if "error" in r:
                print(f"{r['test_id']}  ERROR {r['error']}")
                continue
            print(f"{r['test_id'][:16]:<16} {r['action']:<24} rule={r['rule']:<20} "
                  f"class={r['classification']} conf={r['confidence']} "
                  f"oov={r['oov_rate']}")
        state.write_text(json.dumps(sorted(seen)))
        if args.once:
            print(f"{len(results)} new trace(s) triaged")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
