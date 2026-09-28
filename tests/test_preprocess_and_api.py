"""
Legacy Phase 1 suite - retained only as a marker.

This file imported `src.models.main`, which does not exist (the app lives at
`src.policy_engine.main`), so it had never passed. The pipeline it tested read
data/raw/failure_trace.csv, where each row is a 2155-wide EVENT-COUNT VECTOR
that Phase 1 fed to nn.Embedding as if the counts were token IDs. That
representation is superseded by src/data/build_sequences.py.

Current tests: tests/test_triage.py
"""
import pytest

pytest.skip("superseded by tests/test_triage.py - see module docstring",
            allow_module_level=True)
