"""
Tests for src/rca/localize.py - the S4 root-cause span ranking that replaces
the hardcoded, never-actually-localizing graph_generator.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rca import localize as R  # noqa: E402


def _rec(toks, dur, dep=None):
    dep = dep or [0] * len(toks)
    return {"task": "T1", "scen": "unit-test", "cls": "DATA_FAULT",
            "toks": " ".join(toks), "dur": ",".join(map(str, dur)),
            "dep": ",".join(map(str, dep))}


BASELINE = {
    "getFileInfo": {"mean": 100.0, "std": 20.0, "n": 500},
    "addBlock": {"mean": 200.0, "std": 50.0, "n": 500},
    "__global__": {"mean": 150.0, "std": 40.0, "n": 1000},
}


def test_error_status_span_outranks_a_mildly_slow_ok_span():
    """A modestly-off-baseline OK span (z~1.5) should not outrank an ERR
    span at baseline latency - the flat error bonus dominates ordinary
    latency noise. (An extreme outlier, by contrast, legitimately can
    outrank a small error bonus - that is not tested here, it's correct
    behaviour: a 10-sigma stall is suspicious even with no error status.)"""
    rec = _rec(
        toks=["getFileInfo|OK", "addBlock|ERR", "getFileInfo|OK"],
        dur=[105, 200, 130],   # third span mildly slow (z=1.5), not extreme
    )
    ranked = R.score_spans(rec, BASELINE)
    assert ranked[0]["index"] == 1
    assert ranked[0]["error"] is True


def test_unseen_op_falls_back_to_global_baseline():
    rec = _rec(toks=["neverSeenBefore|OK"], dur=[1000])
    ranked = R.score_spans(rec, BASELINE)
    g = BASELINE["__global__"]
    expected_z = (1000 - g["mean"]) / g["std"]
    assert abs(ranked[0]["z_score"] - max(expected_z, 0.0)) < 1e-6


def test_top_k_respects_k():
    rec = _rec(
        toks=[f"getFileInfo|OK" for _ in range(10)],
        dur=[100 + i for i in range(10)],
    )
    assert len(R.top_k(rec, BASELINE, k=3)) == 3


def test_negative_z_is_clipped_not_negative_ranked():
    """A span FASTER than baseline should not out-rank one that is merely at
    baseline but errored - fast-and-ok is never suspicious."""
    rec = _rec(
        toks=["addBlock|OK", "getFileInfo|ERR"],
        dur=[10, 100],  # addBlock way faster than its 200us mean
    )
    ranked = R.score_spans(rec, BASELINE)
    assert ranked[0]["token"] == "getFileInfo|ERR"
    assert ranked[-1]["z_score"] >= 0.0  # clipped, never negative


def test_generate_postmortem_contains_top_suspect_and_class():
    rec = _rec(toks=["addBlock|ERR", "getFileInfo|OK"], dur=[400, 100])
    md = R.generate_postmortem(rec, BASELINE, "DATA_FAULT",
                               "ESCALATE_TO_INFRA")
    assert "DATA_FAULT" in md
    assert "addBlock" in md
    assert "ESCALATE_TO_INFRA" in md


def test_build_baseline_only_uses_pass_class(tmp_path):
    import gzip
    import json
    corpus = tmp_path / "mini_corpus.jsonl.gz"
    recs = [
        {"cls": "PASS", "toks": "op1|OK op1|OK", "dur": "100,120"},
        {"cls": "DATA_FAULT", "toks": "op1|OK", "dur": "999999"},
    ]
    with gzip.open(corpus, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    baseline = R.build_baseline(corpus)
    assert baseline["op1"]["n"] == 2
    assert 100 <= baseline["op1"]["mean"] <= 120
