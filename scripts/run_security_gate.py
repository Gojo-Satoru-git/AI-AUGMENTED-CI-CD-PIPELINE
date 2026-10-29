"""
S6: security gate.

    python scripts/run_security_gate.py                 # uses reports/pip_audit_raw.json,
                                                        # or runs pip-audit if absent
    python scripts/run_security_gate.py --audit-json my_audit.json
    python scripts/run_security_gate.py --advisories samples/advisories.json
    python scripts/run_security_gate.py --fail-on block # CI: non-zero exit only on BLOCK

Reads pip-audit JSON (`pip-audit -l -f json`), decides for every finding whether
the RUNNING SERVICE can reach it (src/security/reachability.py), prints the
table, and writes reports/security_gate.json - the verdict the release gate
(src/policy_engine/gate.py) consumes.

Exit codes: 0 ALLOW, 1 BLOCK, 2 REVIEW (only when --fail-on review).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.security.reachability import (analyse, parse_pip_audit,   # noqa: E402
                                       security_verdict, summarise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-json", default="reports/pip_audit_raw.json")
    ap.add_argument("--advisories", default=None,
                    help="optional JSON {vuln_id_or_alias: {severity, symbols}}")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="reports/security_gate.json")
    ap.add_argument("--fail-on", choices=["block", "review"], default="block")
    args = ap.parse_args()

    audit = Path(args.audit_json)
    if not audit.exists():
        print("no audit JSON found - running pip-audit (needs network)...")
        audit.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip_audit", "-l", "-f", "json",
                        "-o", str(audit), "--progress-spinner", "off"],
                       check=False)
    doc = json.loads(audit.read_text(encoding="utf-8"))
    adv = (json.loads(Path(args.advisories).read_text(encoding="utf-8"))
           if args.advisories else None)

    findings = parse_pip_audit(doc, adv)
    results = analyse(findings, args.root)
    verdict = security_verdict(results)

    by_pkg = {}
    for r in results:
        by_pkg.setdefault((r.finding.package, r.finding.version, r.reachability,
                           r.evidence), []).append(r.finding.vuln_id)
    print(f"\n{'package':<16}{'version':<10}{'#vulns':>7}  {'reachability':<14}evidence")
    print("-" * 100)
    for (pkg, ver, reach, ev), ids in sorted(by_pkg.items()):
        print(f"{pkg:<16}{ver:<10}{len(ids):>7}  {reach:<14}{ev}")

    s = summarise(results)
    print(f"\nflagged={s['flagged']}  actionable={s['actionable']}  "
          f"dismissed={s['flagged'] - s['actionable']}  "
          f"false-positive reduction={s['false_positive_reduction_pct']}%")
    print(f"VERDICT: {verdict['verdict']}  ({verdict['rule']})  {verdict['reason']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "verdict": verdict,
        "findings": [{
            "package": r.finding.package, "version": r.finding.version,
            "vuln_id": r.finding.vuln_id, "aliases": r.finding.aliases,
            "fix_versions": r.finding.fix_versions,
            "reachability": r.reachability, "evidence": r.evidence,
        } for r in results]}, indent=2))
    print(f"wrote {out}")

    code = {"ALLOW": 0, "BLOCK": 1, "REVIEW": 2 if args.fail_on == "review" else 0}
    sys.exit(code[verdict["verdict"]])


if __name__ == "__main__":
    main()
