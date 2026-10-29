"""
S6: CVE reachability.

A dependency scanner (pip-audit) lists every known vulnerability in every
installed package. Most are noise for a given project: the package may be a
dev tool, may never be imported, or the vulnerable function may never be
called. This module answers the question the scanner cannot: *can the service
actually reach this code?*

Method (static, AST-based - the scoped-down stand-in for a CodeQL/Semgrep call
graph; see PHASE2_PLAN.md section 6b):

  1. Parse every project .py file; record the internal import graph, the
     external packages each module imports, and every dotted call it makes
     (`yaml.load`, resolved through import aliases).
  2. Walk the internal import graph from the SERVICE entry points. Modules
     reached are "runtime"; the rest of the project (scripts, tests, training)
     is "dev".
  3. Map imported module names to distributions and take the dependency
     closure (importlib.metadata) so transitively loaded packages are seen.
  4. Classify each finding:

       reachable      imported by runtime code (and, if the advisory names
                      vulnerable symbols, one of them is actually called)
       transitive     not imported by us but pulled in by a runtime package
       unreachable    package imported by runtime, but the named vulnerable
                      function is never called
       dev-only       only imported by scripts/tests/training, never by the
                      running service
       not-imported   installed but nothing in the project imports it
       unknown        cannot be mapped to an installed distribution

The verdict feeds the same release gate as the triage decision.
"""
from __future__ import annotations

import ast
import collections
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

RUNTIME_ENTRIES = ("run_api.py", "src/policy_engine/main.py")
SKIP_DIRS = {"venv", ".venv", "__pycache__", ".git", "node_modules",
             "site-packages", ".pytest_cache", "build", "dist"}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass
class Finding:
    package: str
    version: str
    vuln_id: str
    aliases: list = field(default_factory=list)
    fix_versions: list = field(default_factory=list)
    severity: str | None = None
    symbols: list = field(default_factory=list)   # e.g. ["yaml.load"]


@dataclass
class Result:
    finding: Finding
    reachability: str
    evidence: str


# ----------------------------------------------------------------------------
def parse_pip_audit(doc: dict, advisories: dict | None = None) -> list[Finding]:
    """pip-audit `-f json` -> Findings. `advisories` optionally enriches by
    vuln id/alias with {"severity": ..., "symbols": [...]} - pip-audit itself
    reports neither severity nor affected functions."""
    advisories = advisories or {}
    out = []
    for dep in doc.get("dependencies", []):
        for v in dep.get("vulns", []):
            keys = [v.get("id", "")] + list(v.get("aliases", []))
            extra = next((advisories[k] for k in keys if k in advisories), {})
            out.append(Finding(
                package=dep["name"], version=dep.get("version", "?"),
                vuln_id=v.get("id", "?"), aliases=list(v.get("aliases", [])),
                fix_versions=list(v.get("fix_versions", [])),
                severity=extra.get("severity"),
                symbols=list(extra.get("symbols", []))))
    return out


# ----------------------------------------------------------------------------
class ProjectGraph:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.files: dict[str, Path] = {}
        for p in self.root.rglob("*.py"):
            if SKIP_DIRS & set(p.relative_to(self.root).parts):
                continue
            rel = list(p.relative_to(self.root).with_suffix("").parts)
            if rel[-1] == "__init__":
                rel = rel[:-1]
            if rel:
                self.files[".".join(rel)] = p
        self.top_internal = {m.split(".")[0] for m in self.files}
        self.edges: dict[str, set[str]] = {}
        self.external: dict[str, set[str]] = {}
        self.calls: dict[str, set[str]] = {}
        for mod, path in self.files.items():
            self._scan(mod, path)

    def _scan(self, mod, path):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            self.edges[mod], self.external[mod], self.calls[mod] = set(), set(), set()
            return
        pkg = mod.split(".")
        if path.name != "__init__.py":
            pkg = pkg[:-1]
        cands, alias = set(), {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    cands.add(a.name)
                    alias[a.asname or a.name.split(".")[0]] = a.name
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg[: len(pkg) - node.level + 1]
                    prefix = ".".join(base + ([node.module] if node.module else []))
                else:
                    prefix = node.module or ""
                if prefix:
                    cands.add(prefix)
                for a in node.names:
                    cands.add(f"{prefix}.{a.name}" if prefix else a.name)
                    alias[a.asname or a.name] = f"{prefix}.{a.name}"
        internal, external = set(), set()
        for c in cands:
            top = c.split(".")[0]
            if top in self.top_internal:
                internal.add(c)
            elif top not in sys.stdlib_module_names:
                external.add(top)
        self.edges[mod] = internal
        self.external[mod] = external
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                parts = []
                cur = node
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id in alias:
                    calls.add(".".join([alias[cur.id]] + parts[::-1]))
            elif isinstance(node, ast.Name) and node.id in alias:
                calls.add(alias[node.id])
        self.calls[mod] = calls

    def reachable_from(self, entries) -> set[str]:
        start = []
        for e in entries:
            m = ".".join(Path(e).with_suffix("").parts)
            if m in self.files:
                start.append(m)
        seen, queue = set(), collections.deque(start)
        while queue:
            m = queue.popleft()
            if m in seen:
                continue
            seen.add(m)
            for tgt in self.edges.get(m, ()):
                # longest prefix that is a real project module
                parts = tgt.split(".")
                for i in range(len(parts), 0, -1):
                    cand = ".".join(parts[:i])
                    if cand in self.files:
                        queue.append(cand)
                        break
        return seen


def _default_dist_map() -> dict[str, list[str]]:
    from importlib.metadata import packages_distributions
    return {k: list(v) for k, v in packages_distributions().items()}


def _default_requires(dist: str) -> list[str]:
    from importlib.metadata import PackageNotFoundError, requires
    try:
        reqs = requires(dist) or []
    except PackageNotFoundError:
        return []
    out = []
    for r in reqs:
        if "extra ==" in r.replace("'", '"'):
            continue
        m = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", r)
        if m:
            out.append(m.group(0))
    return out


def _closure(seed: set[str], requires_fn) -> set[str]:
    seen, queue = set(), collections.deque(seed)
    while queue:
        d = norm(queue.popleft())
        if d in seen:
            continue
        seen.add(d)
        queue.extend(requires_fn(d))
    return seen


def analyse(findings, project_root, entries=RUNTIME_ENTRIES,
            dist_map=None, requires_fn=None, installed=None) -> list[Result]:
    graph = ProjectGraph(project_root)
    dist_map = dist_map if dist_map is not None else _default_dist_map()
    requires_fn = requires_fn or _default_requires
    if installed is None:
        from importlib.metadata import distributions
        installed = {norm(d.metadata["Name"]) for d in distributions()
                     if d.metadata["Name"]}
    installed = {norm(i) for i in installed}

    runtime_mods = graph.reachable_from(entries)
    runtime_ext = set().union(*(graph.external.get(m, set()) for m in runtime_mods)) \
        if runtime_mods else set()
    all_ext = set().union(*graph.external.values()) if graph.external else set()
    runtime_calls = set().union(*(graph.calls.get(m, set()) for m in runtime_mods)) \
        if runtime_mods else set()

    def dists_of(tops):
        return {norm(d) for t in tops for d in dist_map.get(t, [t])}

    direct_runtime = dists_of(runtime_ext)
    direct_all = dists_of(all_ext)
    closure_runtime = _closure(direct_runtime, requires_fn)
    closure_all = _closure(direct_all, requires_fn)

    results = []
    for f in findings:
        d = norm(f.package)
        if d in direct_runtime:
            if f.symbols:
                hit = [s for s in f.symbols if s in runtime_calls]
                if hit:
                    results.append(Result(f, "reachable",
                                          f"runtime code calls {', '.join(hit)}"))
                else:
                    results.append(Result(f, "unreachable",
                                          "package imported at runtime but the "
                                          f"vulnerable function(s) {f.symbols} "
                                          "are never called"))
            else:
                results.append(Result(f, "reachable",
                                      "imported by runtime code (no affected-"
                                      "function data to narrow further)"))
        elif d in closure_runtime:
            results.append(Result(f, "transitive",
                                  "pulled in as a dependency of a runtime package"))
        elif d in closure_all:
            results.append(Result(f, "dev-only",
                                  "only used by scripts/tests/training, never "
                                  "by the running service"))
        elif d in installed:
            results.append(Result(f, "not-imported",
                                  "installed but nothing in the project imports it"))
        else:
            results.append(Result(f, "unknown",
                                  "package is not installed here; cannot map it "
                                  "to the project's imports"))
    return results


# ----------------------------------------------------------------------------
def summarise(results) -> dict:
    by = collections.Counter(r.reachability for r in results)
    n = len(results)
    dismissed = sum(by[k] for k in ("unreachable", "dev-only", "not-imported"))
    return {
        "flagged": n,
        "by_reachability": dict(by),
        "actionable": by["reachable"] + by["transitive"],
        "false_positive_reduction_pct": round(100 * dismissed / n, 1) if n else 0.0,
    }


def security_verdict(results) -> dict:
    """ALLOW / REVIEW / BLOCK for the release gate."""
    reach = [r for r in results if r.reachability == "reachable"]
    blocking = [r for r in reach if (r.finding.severity or "").lower() != "low"]
    soft = [r for r in results if r.reachability in ("transitive", "unknown")]
    if blocking:
        v = {"verdict": "BLOCK", "rule": "reachable-vuln-blocks",
             "reason": f"{len(blocking)} reachable vulnerabilit"
                       f"{'y' if len(blocking)==1 else 'ies'}: "
                       + ", ".join(sorted({r.finding.vuln_id for r in blocking}))}
    elif reach or soft:
        v = {"verdict": "REVIEW", "rule": "transitive-or-unknown-vuln",
             "reason": f"{len(reach)+len(soft)} vulnerabilities need a human "
                       "look (transitive / unknown / low-severity reachable)"}
    else:
        v = {"verdict": "ALLOW", "rule": "no-reachable-vuln",
             "reason": "no vulnerability is reachable from the running service"}
    v["summary"] = summarise(results)
    return v
