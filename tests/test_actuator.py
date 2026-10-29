"""
Tests for S8: the actuator executes decisions faithfully, refuses unsafe
inputs, and never makes policy decisions of its own.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.actuator import actuator as A      # noqa: E402
from src.policy_engine import policy as P   # noqa: E402


def result(action, **kw):
    base = {"decision_id": 7, "test_id": "pr-42", "agent_id": "triage_agent",
            "trust_tier": 3, "action": action, "recommended": action,
            "autonomous": True, "rule": "some-rule", "reasons": ["because"],
            "signals": {}, "classification": "INFRA_FLAKE", "confidence": 0.9,
            "evidence_spans": [], "attempt": 1, "retry_budget": 2}
    base.update(kw)
    return base


def calls(backend):
    return [c[0] for c in backend.calls]


def test_allow_sets_success_and_comments():
    b = A.DryRunBackend()
    A.Actuator(b).execute(result(P.ALLOW))
    assert b.calls[0] == ("set_status", {"state": "success",
                          "description": "ALLOW_MERGE (some-rule)", "context": "ai-gate"})
    assert calls(b) == ["set_status", "comment"]


def test_block_fails_the_check():
    b = A.DryRunBackend()
    A.Actuator(b).execute(result(P.BLOCK))
    assert b.calls[0][1]["state"] == "failure"


def test_retry_within_budget_reruns_failed_jobs():
    b = A.DryRunBackend()
    A.Actuator(b).execute(result(P.RETRY, attempt=2))
    assert "rerun_failed_jobs" in calls(b)
    assert b.calls[0][1]["state"] == "pending"


def test_retry_beyond_budget_is_refused_even_if_told_to_proceed():
    b = A.DryRunBackend()
    recs = A.Actuator(b).execute(result(P.RETRY, attempt=3, retry_budget=2))
    assert "rerun_failed_jobs" not in calls(b)
    assert any("refused" in r.detail for r in recs)


def test_review_leaves_the_check_pending_for_a_human():
    b = A.DryRunBackend()
    A.Actuator(b).execute(result(P.REVIEW, recommended=P.BLOCK, autonomous=False,
                                 trust_tier=0))
    assert b.calls[0][1]["state"] == "pending"


def test_escalate_opens_an_issue():
    b = A.DryRunBackend()
    A.Actuator(b).execute(result(P.ESCALATE))
    assert "open_issue" in calls(b)
    labels = dict(b.calls)["open_issue"]["labels"]
    assert "ai-gate" in labels and "infra" in labels


# --- command-injection guard ---------------------------------------------------
def test_quarantine_runs_a_validated_argv_never_a_shell():
    b = A.DryRunBackend()
    A.Actuator(b, quarantine_cmd="kubectl cordon {node}").execute(
        result(P.QUARANTINE), target="datanode-03")
    assert dict(b.calls)["run_command"]["argv"] == ["kubectl", "cordon", "datanode-03"]


@pytest.mark.parametrize("evil", ["dn1; rm -rf /", "$(reboot)", "a b", "../x",
                                  "node`id`", "", "-flag"])
def test_hostile_node_names_from_trace_data_are_rejected(evil):
    b = A.DryRunBackend()
    recs = A.Actuator(b, quarantine_cmd="kubectl cordon {node}").execute(
        result(P.QUARANTINE), target=evil)
    assert "run_command" not in calls(b)
    assert any(r.step == "run_command" and not r.executed for r in recs)


def test_quarantine_without_a_configured_command_only_recommends():
    b = A.DryRunBackend()
    recs = A.Actuator(b, quarantine_cmd=None).execute(result(P.QUARANTINE),
                                                      target="datanode-03")
    assert "run_command" not in calls(b)
    assert any("no QUARANTINE_CMD" in r.detail for r in recs)


def test_rollback_uses_its_own_template():
    b = A.DryRunBackend()
    A.Actuator(b, rollback_cmd="kubectl rollout undo deployment/{deployment}").execute(
        result(P.ROLLBACK), target="orders")
    assert dict(b.calls)["run_command"]["argv"][-1] == "deployment/orders"


# --- exit codes / comment ------------------------------------------------------
def test_only_allow_and_retry_pass_the_required_check():
    assert A.exit_code_for(P.ALLOW) == 0 and A.exit_code_for(P.RETRY) == 0
    for act in (P.REVIEW, P.BLOCK, P.ESCALATE, P.QUARANTINE, P.ROLLBACK):
        assert A.exit_code_for(act) == 1
    assert A.exit_code_for("SOMETHING_NEW") == 1


def test_comment_names_the_rule_and_the_tier_gap():
    md = A.render_comment(result(P.REVIEW, recommended=P.BLOCK, autonomous=False,
                                 trust_tier=0, rule="security-blocks"))
    assert A.MARKER in md and "security-blocks" in md
    assert "may only recommend" in md and "BLOCK_MERGE_ALERT_DEV" in md


# --- GitHub backend (HTTP faked) ---------------------------------------------
class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(log, responses=None):
    responses = responses or {}

    def _open(req, timeout=None):
        log.append((req.get_method(), req.full_url,
                    json.loads(req.data) if req.data else None,
                    req.get_header("Authorization")))
        body = responses.get((req.get_method(), req.full_url.split("?")[0]), {})
        return FakeResp(json.dumps(body).encode())
    return _open


def test_github_backend_refuses_to_go_live_without_a_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(A.ActuationError):
        A.GitHubBackend("o/r", "sha1")


def test_github_status_call_shape(monkeypatch):
    log = []
    monkeypatch.setattr(A.urllib.request, "urlopen", fake_urlopen(log))
    gh = A.GitHubBackend("o/r", "abc123", token="tok", pr_number=5)
    gh.set_status("success", "x" * 300)
    method, url, body, auth = log[0]
    assert (method, url) == ("POST", "https://api.github.com/repos/o/r/statuses/abc123")
    assert body["state"] == "success" and len(body["description"]) == 140
    assert auth == "Bearer tok"


def test_comment_updates_in_place_instead_of_spamming(monkeypatch):
    log = []
    listing = {("GET", "https://api.github.com/repos/o/r/issues/5/comments"):
               [{"id": 99, "body": f"old {A.MARKER}"}]}
    monkeypatch.setattr(A.urllib.request, "urlopen", fake_urlopen(log, listing))
    A.GitHubBackend("o/r", "s", token="t", pr_number=5).comment("new")
    assert log[-1][0] == "PATCH" and log[-1][1].endswith("/issues/comments/99")


def test_comment_creates_when_no_previous_one(monkeypatch):
    log = []
    monkeypatch.setattr(A.urllib.request, "urlopen", fake_urlopen(log, {}))
    A.GitHubBackend("o/r", "s", token="t", pr_number=5).comment("new")
    assert log[-1][0] == "POST"


def test_http_errors_become_actuation_errors_without_leaking_the_token(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "forbidden", {}, None)
    monkeypatch.setattr(A.urllib.request, "urlopen", boom)
    gh = A.GitHubBackend("o/r", "s", token="SECRET-TOKEN", pr_number=1)
    with pytest.raises(A.ActuationError) as e:
        gh.set_status("success", "d")
    assert "SECRET-TOKEN" not in str(e.value) and "403" in str(e.value)


def test_a_failed_step_is_recorded_not_raised(monkeypatch):
    class Flaky(A.DryRunBackend):
        def comment(self, body):
            raise A.ActuationError("no pull request number")
    recs = A.Actuator(Flaky()).execute(result(P.ALLOW))
    assert [r.step for r in recs] == ["set_status", "comment"]
    assert recs[1].executed is False and "pull request" in recs[1].detail
