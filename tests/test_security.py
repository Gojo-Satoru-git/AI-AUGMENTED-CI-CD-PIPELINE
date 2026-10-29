"""
Tests for S6: CVE reachability over a throwaway project, with injected
distribution maps so the result does not depend on what is installed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.security import reachability as R    # noqa: E402


@pytest.fixture()
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "run_api.py").write_text("from src.app import main\nmain()\n")
    (tmp_path / "src" / "app.py").write_text(
        "import yaml\nimport requests\n\n"
        "def main():\n    return yaml.load('a: 1'), requests.get\n")
    (tmp_path / "scripts" / "train.py").write_text(
        "import sklearn\nfrom src import app\n")
    return tmp_path


DIST_MAP = {"yaml": ["PyYAML"], "requests": ["requests"],
            "sklearn": ["scikit-learn"]}
REQUIRES = {"requests": ["urllib3", "idna"]}
INSTALLED = {"pyyaml", "requests", "urllib3", "idna", "scikit-learn", "pip"}


def run(project, findings):
    return {r.finding.package: r for r in R.analyse(
        findings, project, dist_map=DIST_MAP,
        requires_fn=lambda d: REQUIRES.get(d, []), installed=INSTALLED)}


def F(pkg, **kw):
    return R.Finding(pkg, "1.0", f"VULN-{pkg}", **kw)


def test_directly_imported_and_called_is_reachable(project):
    res = run(project, [F("PyYAML", symbols=["yaml.load"])])
    assert res["PyYAML"].reachability == "reachable"
    assert "yaml.load" in res["PyYAML"].evidence


def test_imported_but_vulnerable_function_not_called_is_unreachable(project):
    res = run(project, [F("PyYAML", symbols=["yaml.unsafe_load"])])
    assert res["PyYAML"].reachability == "unreachable"


def test_no_symbol_data_falls_back_to_package_level_reachable(project):
    assert run(project, [F("requests")])["requests"].reachability == "reachable"


def test_transitive_dependency_of_runtime_package(project):
    assert run(project, [F("urllib3")])["urllib3"].reachability == "transitive"


def test_only_imported_by_scripts_is_dev_only(project):
    assert run(project, [F("scikit-learn")])["scikit-learn"].reachability == "dev-only"


def test_installed_but_never_imported(project):
    assert run(project, [F("pip")])["pip"].reachability == "not-imported"


def test_uninstalled_unmapped_package_is_unknown(project):
    assert run(project, [F("ghostpkg")])["ghostpkg"].reachability == "unknown"


def test_name_normalisation():
    assert R.norm("Scikit_Learn") == R.norm("scikit-learn")


# ---------------------------------------------------------------- verdict
def test_verdict_blocks_on_reachable(project):
    results = list(run(project, [F("PyYAML", symbols=["yaml.load"]),
                                 F("pip")]).values())
    v = R.security_verdict(results)
    assert v["verdict"] == "BLOCK" and v["rule"] == "reachable-vuln-blocks"


def test_verdict_allows_when_everything_is_unreachable(project):
    results = list(run(project, [F("scikit-learn"), F("pip"),
                                 F("PyYAML", symbols=["yaml.unsafe_load"])]).values())
    v = R.security_verdict(results)
    assert v["verdict"] == "ALLOW"
    assert v["summary"]["false_positive_reduction_pct"] == 100.0


def test_verdict_reviews_transitive_and_low_severity_reachable(project):
    results = list(run(project, [F("urllib3")]).values())
    assert R.security_verdict(results)["verdict"] == "REVIEW"
    results = list(run(project, [F("requests", severity="low")]).values())
    assert R.security_verdict(results)["verdict"] == "REVIEW"


def test_false_positive_reduction_figure(project):
    results = list(run(project, [F("requests"), F("scikit-learn"), F("pip"),
                                 F("urllib3")]).values())
    s = R.summarise(results)
    assert s["flagged"] == 4 and s["actionable"] == 2
    assert s["false_positive_reduction_pct"] == 50.0


# ---------------------------------------------------------------- parsing
def test_parse_pip_audit_with_advisory_enrichment():
    doc = {"dependencies": [
        {"name": "PyYAML", "version": "5.3", "vulns": [
            {"id": "PYSEC-1", "aliases": ["CVE-2020-1747"],
             "fix_versions": ["5.4"]}]},
        {"name": "safe", "version": "1", "vulns": []}]}
    adv = {"CVE-2020-1747": {"severity": "critical", "symbols": ["yaml.load"]}}
    out = R.parse_pip_audit(doc, adv)
    assert len(out) == 1
    assert out[0].severity == "critical" and out[0].symbols == ["yaml.load"]
    assert out[0].fix_versions == ["5.4"]


def test_real_project_graph_finds_the_service_entrypoint():
    g = R.ProjectGraph(ROOT)
    runtime = g.reachable_from(R.RUNTIME_ENTRIES)
    assert "src.policy_engine.main" in runtime
    assert "src.policy_engine.policy" in runtime
    # training code is not part of the running service
    assert "src.models.train" not in runtime


def test_positive_control_on_the_real_project():
    """Guards against an analyzer that dismisses everything: packages the
    service genuinely imports must be flagged, dev tooling must not be."""
    fs = [R.Finding(p, "x", "CONTROL") for p in
          ("fastapi", "torch", "pytest", "matplotlib")]
    got = {r.finding.package: r.reachability for r in R.analyse(fs, ROOT)}
    assert got["fastapi"] == "reachable" and got["torch"] == "reachable"
    assert got["pytest"] == "dev-only" and got["matplotlib"] == "dev-only"
