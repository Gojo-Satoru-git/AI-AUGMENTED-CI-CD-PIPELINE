"""
Equivalence check between policy.py's decide() and the Rego mirror at
src/policy_engine/rego/triage_policy.rego.

Requires the `opa` CLI on PATH. It is NOT installed in this environment (no
binary was downloaded to build this test - see the module docstring in the
.rego file), so every test here is skipped rather than faked, and reports
that skip reason explicitly. Once `opa` is available (e.g. in CI, or a dev
machine with it installed), this file starts actually enforcing the
equivalence with no other changes needed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policy_engine import policy as P  # noqa: E402

REGO = ROOT / "src" / "policy_engine" / "rego" / "triage_policy.rego"
OPA = shutil.which("opa")

CASES = [
    dict(classification="PASS", confidence=0.95, trust_tier=3),
    dict(classification="PASS", confidence=0.95, trust_tier=0),
    dict(classification="REAL_BUG", confidence=0.9, trust_tier=3),
    dict(classification="REAL_BUG", confidence=0.9, trust_tier=0),
    dict(classification="INFRA_FLAKE", confidence=0.9, trust_tier=3,
         attempt=1, retry_budget=2),
    dict(classification="INFRA_FLAKE", confidence=0.9, trust_tier=3,
         attempt=3, retry_budget=2),
    dict(classification="DATA_FAULT", confidence=0.9, trust_tier=3),
    dict(classification="ENV_FAULT", confidence=0.9, trust_tier=3),
    dict(classification="ENV_FAULT", confidence=0.9, trust_tier=2),
    dict(classification="REAL_BUG", confidence=0.31, trust_tier=3),
    dict(classification="SOMETHING_NEW", confidence=0.99, trust_tier=3),
]


def _rego_decision(case: dict) -> str:
    proc = subprocess.run(
        [OPA, "eval", "-f", "raw", "-d", str(REGO),
         "-i", "-", "data.triage.policy.decision"],
        input=json.dumps(case), capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().strip('"')


@pytest.mark.skipif(OPA is None, reason="opa CLI not installed - see module "
                    "docstring; this test enforces nothing until it is")
@pytest.mark.parametrize("case", CASES)
def test_rego_matches_python_policy(case):
    py = P.decide(**case)
    assert _rego_decision(case) == py.action
