"""
S8: the actuator.

    The decision engine never executes; the actuator never decides.

This module turns an already-made decision (the dict returned by
src/policy_engine/pipeline.evaluate) into side effects on GitHub / the cluster.
It contains no policy: it does not pick actions, check confidence, or consult
the trust tier - the decision's `action` is already the thing the agent is
allowed to do. The two safety rails here are not decisions, they are limits on
what any decision may touch:

  * DRY-RUN BY DEFAULT. Nothing is sent anywhere unless a live backend is
    constructed explicitly.
  * COMMAND INJECTION GUARD. Node and deployment names ultimately come from
    trace data, which is untrusted input. They are validated against a strict
    pattern and commands run as argv lists (never through a shell).
  * RETRY CAP. A retry beyond the budget is refused even if told to proceed
    (defence in depth against a mis-wired caller; the policy already escalates).

Backends:
    DryRunBackend    records what it would have done (tests, demos, --dry-run)
    GitHubBackend    commit status, PR comment (updated in place), rerun of
                     failed jobs, issues - GitHub REST API, token from env only
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from src.policy_engine import policy as P

GITHUB_API = "https://api.github.com"
MARKER = "<!-- ai-gate -->"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,252}$")

STATUS_FOR = {
    P.ALLOW: "success", P.RETRY: "pending", P.REVIEW: "pending",
    P.ESCALATE: "failure", P.QUARANTINE: "failure", P.BLOCK: "failure",
    P.ROLLBACK: "failure",
}
# a required CI check should only go green when the agent says it is safe
EXIT_CODE_FOR = {
    P.ALLOW: 0, P.RETRY: 0, P.REVIEW: 1, P.ESCALATE: 1, P.QUARANTINE: 1,
    P.BLOCK: 1, P.ROLLBACK: 1,
}


def exit_code_for(action: str) -> int:
    return EXIT_CODE_FOR.get(action, 1)


class ActuationError(RuntimeError):
    pass


@dataclass
class ActionRecord:
    step: str
    executed: bool
    dry_run: bool
    detail: str = ""


# ----------------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------------
class DryRunBackend:
    dry_run = True

    def __init__(self):
        self.calls = []

    def _rec(self, name, **kw):
        self.calls.append((name, kw))
        return {"dry_run": True}

    def set_status(self, state, description, context="ai-gate"):
        return self._rec("set_status", state=state, description=description,
                         context=context)

    def comment(self, body):
        return self._rec("comment", body=body)

    def rerun_failed_jobs(self):
        return self._rec("rerun_failed_jobs")

    def open_issue(self, title, body, labels=()):
        return self._rec("open_issue", title=title, body=body, labels=list(labels))

    def run_command(self, argv):
        return self._rec("run_command", argv=list(argv))


class GitHubBackend:
    dry_run = False

    def __init__(self, repo, sha, token=None, pr_number=None, run_id=None,
                 timeout=15):
        self.repo, self.sha = repo, sha
        self.pr_number, self.run_id = pr_number, run_id
        self.timeout = timeout
        self._token = token or os.environ.get("GITHUB_TOKEN")
        if not self._token:
            raise ActuationError("GITHUB_TOKEN is not set; refusing to go live")

    def _call(self, method, path, payload=None):
        req = urllib.request.Request(
            f"{GITHUB_API}{path}", method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": f"Bearer {self._token}",
                     "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28",
                     "Content-Type": "application/json",
                     "User-Agent": "ai-gate-actuator"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:           # never echo the token
            raise ActuationError(f"GitHub {method} {path} -> HTTP {e.code}") from None
        except urllib.error.URLError as e:
            raise ActuationError(f"GitHub {method} {path} unreachable: {e.reason}") from None

    def set_status(self, state, description, context="ai-gate"):
        return self._call("POST", f"/repos/{self.repo}/statuses/{self.sha}", {
            "state": state, "description": description[:140], "context": context})

    def comment(self, body):
        if not self.pr_number:
            raise ActuationError("no pull request number; cannot comment")
        base = f"/repos/{self.repo}/issues/{self.pr_number}/comments"
        for c in self._call("GET", base + "?per_page=100") or []:
            if MARKER in c.get("body", ""):
                return self._call("PATCH",
                                  f"/repos/{self.repo}/issues/comments/{c['id']}",
                                  {"body": body})
        return self._call("POST", base, {"body": body})

    def rerun_failed_jobs(self):
        if not self.run_id:
            raise ActuationError("no workflow run id; cannot rerun")
        return self._call("POST", f"/repos/{self.repo}/actions/runs/"
                                  f"{self.run_id}/rerun-failed-jobs")

    def open_issue(self, title, body, labels=()):
        return self._call("POST", f"/repos/{self.repo}/issues",
                          {"title": title, "body": body, "labels": list(labels)})

    def run_command(self, argv):
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60,
                             shell=False, check=False)
        if out.returncode != 0:
            raise ActuationError(f"{argv[0]} exited {out.returncode}: "
                                 f"{out.stderr.strip()[:200]}")
        return {"stdout": out.stdout.strip()[:500]}


# ----------------------------------------------------------------------------
def render_comment(result: dict) -> str:
    """The PR comment: what was decided, by which rule, from which signals."""
    a, rec = result["action"], result["recommended"]
    icon = {"success": "✅", "pending": "⏸️", "failure": "⛔"}[STATUS_FOR.get(a, "failure")]
    L = [MARKER, f"## {icon} AI release gate: `{a}`", ""]
    if rec != a:
        L.append(f"> The signals recommend **`{rec}`**, but the agent is at "
                 f"**T{result['trust_tier']}** and may only recommend it - "
                 "a human must act.")
        L.append("")
    L += [f"**Rule:** `{result['rule']}`  |  **Trust tier:** T{result['trust_tier']}"
          f"  |  **Autonomous:** {'yes' if result['autonomous'] else 'no'}", ""]
    if result.get("classification"):
        L.append(f"**Triage:** {result['classification']} "
                 f"(confidence {result['confidence']:.2f})")
    for r in result.get("reasons", []):
        L.append(f"- {r}")
    sig = result.get("signals", {})
    if sig:
        L += ["", "| signal | verdict | detail |", "|---|---|---|"]
        if "triage" in sig:
            t = sig["triage"]
            L.append(f"| triage | {t['classification']} -> {t['action']} | "
                     f"rule `{t['rule']}`, conf {t['confidence']:.2f} |")
        if "security" in sig:
            s = sig["security"]
            L.append(f"| security | {s['verdict']} | {s.get('reason')} |")
        if "canary" in sig:
            c = sig["canary"]
            L.append(f"| canary | {c['verdict']} | score {c['score']}: {c['reason']} |")
        if "drift" in sig:
            d = sig["drift"]
            fit = ("in-distribution" if d["in_distribution"]
                   else "OUT OF DISTRIBUTION")
            why = (d.get("note") or "") if not d["in_distribution"] else ""
            L.append(f"| model fit | {fit} | OOV rate {d['oov_rate']}. {why} |")
    ev = result.get("evidence_spans")
    if ev:
        L += ["", "<details><summary>Evidence spans</summary>", ""]
        L += [f"- `{e['token']}` (attention {e['attention']})" for e in ev[:5]]
        L += ["", "</details>"]
    ref = (f"decision #{result['decision_id']}" if result.get("decision_id")
           else "decision not logged")
    L += ["", f"<sub>{ref} · {result['test_id']}"
              " · report the true outcome via /api/v1/triage/ground-truth so the "
              "agent's trust tier can learn from it</sub>"]
    return "\n".join(L)


class Actuator:
    def __init__(self, backend=None, quarantine_cmd=None, rollback_cmd=None):
        self.backend = backend or DryRunBackend()
        self.quarantine_cmd = quarantine_cmd or os.environ.get("QUARANTINE_CMD")
        self.rollback_cmd = rollback_cmd or os.environ.get("ROLLBACK_CMD")

    # -- helpers ---------------------------------------------------------------
    def _guarded_argv(self, template, **names):
        if not template:
            return None
        for k, v in names.items():
            if not v or not SAFE_NAME.match(str(v)):
                raise ActuationError(f"refusing to run command: {k}={v!r} is not "
                                     "a safe name")
        return shlex.split(template.format(**names))

    def _step(self, records, step, fn, *a, **kw):
        try:
            fn(*a, **kw)
            records.append(ActionRecord(step, True, self.backend.dry_run))
        except ActuationError as e:
            records.append(ActionRecord(step, False, self.backend.dry_run, str(e)))

    # -- main ------------------------------------------------------------------
    def execute(self, result: dict, target: str | None = None) -> list[ActionRecord]:
        a = result["action"]
        b, rec = self.backend, []
        status = STATUS_FOR.get(a, "failure")
        desc = f"{a} ({result['rule']})"

        if a == P.RETRY and result.get("attempt", 1) > result.get(
                "retry_budget", P.DEFAULT_RETRY_BUDGET):
            self._step(rec, "set_status", b.set_status, "failure",
                       "retry refused: budget exhausted")
            rec.append(ActionRecord("rerun_failed_jobs", False, b.dry_run,
                                    "refused: retry budget exhausted"))
            self._step(rec, "comment", b.comment, render_comment(result))
            return rec

        self._step(rec, "set_status", b.set_status, status, desc)

        if a == P.RETRY:
            self._step(rec, "rerun_failed_jobs", b.rerun_failed_jobs)
        elif a == P.ESCALATE:
            self._step(rec, "open_issue", b.open_issue,
                       f"[ai-gate] {result['test_id']}: escalated to infra",
                       render_comment(result), ("ai-gate", "infra"))
        elif a == P.QUARANTINE:
            try:
                argv = self._guarded_argv(self.quarantine_cmd, node=target)
            except ActuationError as e:
                rec.append(ActionRecord("run_command", False, b.dry_run, str(e)))
            else:
                if argv:
                    self._step(rec, "run_command", b.run_command, argv)
                else:
                    rec.append(ActionRecord("run_command", False, b.dry_run,
                                            "no QUARANTINE_CMD configured; "
                                            "recommendation posted only"))
        elif a == P.ROLLBACK:
            try:
                argv = self._guarded_argv(self.rollback_cmd, deployment=target)
            except ActuationError as e:
                rec.append(ActionRecord("run_command", False, b.dry_run, str(e)))
            else:
                if argv:
                    self._step(rec, "run_command", b.run_command, argv)
                else:
                    rec.append(ActionRecord("run_command", False, b.dry_run,
                                            "no ROLLBACK_CMD configured; "
                                            "recommendation posted only"))

        self._step(rec, "comment", b.comment, render_comment(result))
        return rec
