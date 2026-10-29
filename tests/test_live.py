"""
Tests for the minimal live system: tracer, minishop faults, collection, and the
closed loop. These pin the properties the live demo's claims rest on - above all
that labels are exact and that each fault leaves the signature it is supposed to.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.minishop import FAULT_CLASS, Fault, Shop          # noqa: E402
from src.live.tracer import push_otlp                           # noqa: E402
from src.observability.jaeger_adapter import jaeger_to_spans    # noqa: E402

FAST = 0.02          # time scale: fast tests, same structure


def traces(kind, n=40, workload="checkout", hit_p=1.0, severity=1.0, seed=1):
    shop = Shop(Fault(kind, hit_p=hit_p, severity=severity), seed=seed, ms=FAST)
    return [shop.handle(workload) for _ in range(n)]


def tokens(tr):
    spans, _ = jaeger_to_spans(tr.to_jaeger())
    return [f"{s['op']}|{s['status']}" for s in spans]


# ------------------------------------------------------------------ tracer
def test_trace_converts_to_the_jaeger_shape_the_adapter_reads():
    tr = traces("none", 1)[0]
    doc = tr.to_jaeger()
    assert doc["traceID"] and doc["spans"] and doc["processes"]
    spans, extra = jaeger_to_spans(doc)
    assert spans[0]["depth"] == 0
    assert {s["host"] for s in spans} == {"frontend", "orders", "inventory", "db"}
    assert extra["n_edge"] == len(spans) - 1


def test_otlp_json_is_structurally_valid():
    """Structure only - this was never sent to a real Jaeger from the dev machine."""
    tr = traces("db_corruption", 1)[0]
    doc = tr.to_otlp()
    spans = [s for rs in doc["resourceSpans"]
             for ss in rs["scopeSpans"] for s in ss["spans"]]
    assert len(spans) == len(tr.spans)
    for s in spans:
        assert len(s["traceId"]) == 32 and len(s["spanId"]) == 16
        assert int(s["endTimeUnixNano"]) > int(s["startTimeUnixNano"])
        assert s["status"]["code"] in (1, 2)
    services = {a["value"]["stringValue"] for rs in doc["resourceSpans"]
                for a in rs["resource"]["attributes"] if a["key"] == "service.name"}
    assert services == {"frontend", "orders", "inventory", "db"}
    assert len([s for s in spans if s["parentSpanId"] == ""]) == 1


def test_push_otlp_never_raises_when_jaeger_is_absent():
    err = push_otlp("http://127.0.0.1:1/v1/traces", traces("none", 1)[0], timeout=0.5)
    assert isinstance(err, str) and err


# ------------------------------------------------------------ fault signatures
def test_healthy_system_is_all_pass_and_all_ok():
    ts = traces("none")
    assert all(Shop.label(t) == "PASS" and t.ok for t in ts)
    assert not any("ERR" in x or "IOEXC" in x for t in ts for x in tokens(t))


def test_flaky_network_shows_failed_attempts_followed_by_retries():
    ts = [t for t in traces("flaky_net", 80, severity=1.0)
          if Shop.label(t) == "INFRA_FLAKE"]
    assert ts
    recovered = 0
    for t in ts:
        seq = [x for x in tokens(t) if x.startswith("db.insert")]
        assert any(x.endswith("IOEXC") for x in seq)
        recovered += seq[-1].endswith("OK")
    assert recovered > 0, "retries never recovered - not a transient fault"


def test_dead_host_fails_every_attempt_and_the_request_never_recovers():
    for t in traces("host_down", 10):
        assert Shop.label(t) == "ENV_FAULT" and not t.ok
        connects = [x for x in tokens(t) if x.startswith("inventory.connect")]
        assert connects == ["inventory.connect|IOEXC"] * 3


def test_data_corruption_is_deterministic_and_never_retried():
    ts = [t for t in traces("db_corruption", 80) if Shop.label(t) == "DATA_FAULT"]
    assert ts
    for t in ts:
        inserts = [x for x in tokens(t) if x.startswith("db.insert")]
        assert len(inserts) == 1 and inserts[0].endswith("ERR")
        assert not t.ok


def test_code_bug_is_an_exception_in_business_logic_before_any_db_call():
    ts = [t for t in traces("code_bug", 60) if Shop.label(t) == "REAL_BUG"]
    assert ts
    for t in ts:
        toks = tokens(t)
        assert any(x.startswith("orders.validate") and x.endswith("ERR") for x in toks)
        assert not any(x.startswith("db.") for x in toks)


def test_slow_db_changes_durations_not_statuses():
    fast = traces("none", 15, seed=2)
    slow = [t for t in traces("slow_db", 30, severity=1.0, seed=2)
            if Shop.label(t) == "INFRA_FLAKE"]

    def insert_us(t):
        return next(s["duration"] for s in t.spans
                    if s["operationName"].startswith("db.insert"))

    assert (sum(map(insert_us, slow)) / len(slow)
            > 5 * sum(map(insert_us, fast)) / len(fast))
    assert all(t.ok for t in slow)


def test_silent_bug_leaves_a_trace_indistinguishable_from_healthy():
    """REAL_BUG by label, healthy by every observable: the case only the trust
    engine can catch."""
    bug = [t for t in traces("silent_bug", 30, hit_p=1.0)
           if Shop.label(t) == "REAL_BUG"]
    assert bug
    healthy_ops = {x for t in traces("none", 20) for x in tokens(t)}
    for t in bug:
        assert t.ok and set(tokens(t)) <= healthy_ops


def test_labels_are_exact_not_inherited_from_the_experiment():
    """A request that never touched the faulty path is PASS even in a faulty
    experiment; any other label names the injected fault."""
    for kind in ("flaky_net", "db_corruption", "code_bug", "slow_db"):
        labels = {Shop.label(t) for t in traces(kind, 60, hit_p=0.5, severity=0.5)}
        assert labels <= {"PASS", FAULT_CLASS[kind]}
        assert "PASS" in labels and FAULT_CLASS[kind] in labels


def test_same_seed_same_structure():
    a = [tokens(t) for t in traces("flaky_net", 15, seed=5)]
    b = [tokens(t) for t in traces("flaky_net", 15, seed=5)]
    assert a == b


# ------------------------------------------------------------------ collection
def test_collection_writes_a_manifest_the_training_route_accepts(tmp_path):
    from scripts import live_collect as LC
    from src.data.dataset import prepare
    from src.data.jaeger_corpus import build_corpus
    totals, err = LC.run(10, tmp_path, seed=3, ms=FAST)
    assert err is None and sum(totals.values()) == 10 * len(LC.EXPERIMENTS)
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["meta"]["ms"] == FAST
    info = build_corpus(tmp_path / "manifest.json", tmp_path / "c.jsonl.gz")
    assert all(n >= 3 for n in info["scenarios_per_class"].values())
    ds, vocab, itos, meta = prepare(str(tmp_path / "c.jsonl.gz"), max_len=32)
    # regression: mixed-class scenarios used to drop whole classes from the test set
    for split in ("train", "val", "test"):
        present = {c for c, n in meta["class_counts"][split].items() if n}
        assert {"INFRA_FLAKE", "DATA_FAULT", "ENV_FAULT", "REAL_BUG"} <= present, \
            (split, present)


# ------------------------------------------------------------------ closed loop
LIVE_MODEL = ROOT / "src" / "models" / "saved_live" / "triage_lstm.pth"


@pytest.mark.skipif(not LIVE_MODEL.exists(), reason="train the live model first")
def test_closed_loop_runs_end_to_end(tmp_path):
    out = tmp_path / "run.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "live_run.py"), "--quiet",
         "--scale", "0.25", "--agent-id", "it_live_agent", "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", timeout=300, cwd=ROOT)
    try:
        assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]
        res = json.loads(out.read_text())
        assert res["requests"] > 50 and res["agent_correct"] > 0.5
        # in-distribution by construction: the model was trained on adapter features
        assert "ood-abstain" not in json.dumps(res["phases"])
    finally:
        from src.policy_engine import database as db
        with db._conn() as c:
            for t in ("decisions", "tier_history", "agent_trust"):
                c.execute(f"DELETE FROM {t} WHERE agent_id='it_live_agent'")
