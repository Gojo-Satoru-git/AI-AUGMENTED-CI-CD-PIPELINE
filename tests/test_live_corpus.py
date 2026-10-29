"""
The route to a model for a NEW system: labelled Jaeger traces -> corpus ->
train -> serve, with train/serve feature parity. Uses small synthetic
microservice traces, so it proves the plumbing, not real-world accuracy.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import jaeger_corpus as JC     # noqa: E402
from src.observability.jaeger_adapter import load_jaeger   # noqa: E402

PROCS = {"p1": {"serviceName": "frontend"}, "p2": {"serviceName": "orders"},
         "p3": {"serviceName": "db"}}


def make_trace(tid, kind, rng):
    t0 = 1_700_000_000_000_000
    j = lambda x: int(x * rng.uniform(0.8, 1.2))          # noqa: E731
    db_tags, db_dur = [], j(20_000)
    ord_tags = []
    if kind == "INFRA_FLAKE":
        ord_tags = [{"key": "error", "value": True},
                    {"key": "error.message", "value": "connection timeout"}]
        db_dur = j(900_000)
    elif kind == "DATA_FAULT":
        db_tags = [{"key": "error", "value": True},
                   {"key": "error.message", "value": "checksum mismatch"}]
    spans = [
        {"traceID": tid, "spanID": "a", "operationName": "GET /checkout",
         "references": [], "startTime": t0, "duration": j(100_000), "tags": [],
         "logs": [], "processID": "p1"},
        {"traceID": tid, "spanID": "b", "operationName": "orders.create",
         "references": [{"refType": "CHILD_OF", "traceID": tid, "spanID": "a"}],
         "startTime": t0 + 1000, "duration": j(70_000), "tags": ord_tags,
         "logs": [], "processID": "p2"},
        {"traceID": tid, "spanID": "c", "operationName": "db.insert orders",
         "references": [{"refType": "CHILD_OF", "traceID": tid, "spanID": "b"}],
         "startTime": t0 + 2000, "duration": db_dur, "tags": db_tags,
         "logs": [], "processID": "p3"},
    ]
    return {"traceID": tid, "spans": spans, "processes": PROCS}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("live")
    rng = random.Random(0)
    exps = []
    for cls in ("PASS", "INFRA_FLAKE", "DATA_FAULT"):
        for run in range(4):                       # 4 independent scenarios/class
            name = f"{cls.lower()}-run{run}"
            traces = [make_trace(f"{name}-{i}", cls, rng) for i in range(25)]
            (d / f"{name}.json").write_text(json.dumps({"data": traces}))
            exps.append({"scenario": name, "class": cls, "files": [f"{name}.json"]})
    (d / "manifest.json").write_text(json.dumps({"experiments": exps}))
    out = d / "corpus_live.jsonl.gz"
    info = JC.build_corpus(d / "manifest.json", out)
    return d, out, info


def test_corpus_records_match_the_training_format(corpus):
    import gzip
    _d, out, info = corpus
    assert info["traces"] == 300 and info["feature_schema"] == "jaeger_adapter_v1"
    assert all(v >= 3 for v in info["scenarios_per_class"].values())
    with gzip.open(out, "rt", encoding="utf-8") as f:
        r = json.loads(f.readline())
    n = r["toks"].count(" ") + 1
    assert len(r["dur"].split(",")) == n and len(r["dep"].split(",")) == n
    assert len(r["hr"]) == n and len(r["ag"]) == n and r["f"]["n_ev"] == n
    assert set(r["f"]) == {"n_ev", "n_tid", "n_host", "n_edge", "n_rep", "lat",
                           "n_err", "n_op"}
    assert " " not in r["toks"].split(" ")[2].split("|")[0]    # op normalised


def test_unknown_class_is_rejected():
    with pytest.raises(ValueError):
        JC.trace_to_record("t", [], {}, "NOT_A_CLASS", "s")


def test_the_corpus_is_loadable_by_the_existing_pipeline(corpus):
    from src.data.dataset import prepare
    _d, out, _ = corpus
    ds, vocab, itos, meta = prepare(str(out), max_len=32)
    assert set(meta["labels"]) == {"PASS", "INFRA_FLAKE", "DATA_FAULT"}
    assert all(n > 0 for n in meta["split_sizes"].values())      # leakage asserts passed


def test_train_then_serve_with_train_serve_parity(corpus, tmp_path):
    from src.models.predict import TriagePredictor
    from src.policy_engine.pipeline import evaluate
    _d, out, _ = corpus
    saved = tmp_path / "saved"
    r = subprocess.run(
        [sys.executable, str(ROOT / "src/models/train.py"), "--corpus", str(out),
         "--out", str(saved), "--reports", str(tmp_path / "rep"),
         "--epochs", "8", "--batch-size", "32", "--hidden", "16", "--emb-dim", "8",
         "--max-len", "32", "--patience", "8", "--lr", "0.01",
         "--feature-schema", "jaeger_adapter_v1"],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]

    pred = TriagePredictor(ckpt=saved / "triage_lstm.pth",
                           vocab_path=saved / "vocab.json", device="cpu")
    assert pred.feature_schema == "jaeger_adapter_v1"

    rng = random.Random(99)
    right = 0
    for kind in ("PASS", "INFRA_FLAKE", "DATA_FAULT"):
        for i in range(10):
            doc = {"data": [make_trace(f"new-{kind}-{i}", kind, rng)]}
            _t, spans, extra = next(iter(load_jaeger(doc)))
            res = evaluate(pred, spans=spans, extra=extra, tier=3, log=False)
            # parity: an adapter-made trace is in-distribution for this model
            assert res["signals"]["drift"]["in_distribution"] is True
            right += res["classification"] == kind
    assert right >= 24, f"model failed to learn a trivially separable problem ({right}/30)"


def test_the_hdfs_model_still_refuses_the_same_traces():
    """Control: same synthetic microservice traces, the shipped HDFS model."""
    from src.models.predict import TriagePredictor
    from src.policy_engine.pipeline import evaluate
    ckpt = ROOT / "src/models/saved/triage_lstm.pth"
    if not ckpt.exists():
        pytest.skip("no shipped model")
    pred = TriagePredictor()
    assert pred.feature_schema == "tracebench_v1"
    doc = {"data": [make_trace("ctl", "DATA_FAULT", random.Random(1))]}
    _t, spans, extra = next(iter(load_jaeger(doc)))
    res = evaluate(pred, spans=spans, extra=extra, tier=3, log=False)
    assert res["rule"] == "ood-abstain"


def test_split_keeps_every_fault_class_in_test_when_scenarios_mix_classes():
    """Regression: a faulty experiment also contains healthy (PASS) requests.
    Assigning scenarios per class let the PASS pass overwrite the others, and a
    live corpus ended up with NO INFRA_FLAKE or DATA_FAULT in its test set - a
    'test accuracy 1.0' that meant nothing."""
    from src.data.dataset import split_by_scenario
    recs = []
    for cls in ("INFRA_FLAKE", "DATA_FAULT", "ENV_FAULT"):
        for run in range(4):
            scen = f"{cls}-run{run}"
            recs += [{"scen": scen, "cls": cls}] * 30
            recs += [{"scen": scen, "cls": "PASS"}] * 20      # same scenario, healthy
    for run in range(4):
        recs += [{"scen": f"pass-run{run}", "cls": "PASS"}] * 40
    assign, _ = split_by_scenario(recs, seed=42)
    for cls in ("INFRA_FLAKE", "DATA_FAULT", "ENV_FAULT"):
        splits = {assign[f"{cls}-run{r}"] for r in range(4)}
        assert {"train", "val", "test"} <= splits, f"{cls} missing from a split: {splits}"
    # a scenario is never divided between splits
    assert len(assign) == 16
