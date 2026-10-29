"""
FINAL DEMO - the pieces added after Review 2: Jaeger ingestion with a drift
guard (S5), reachability-aware security gate (S6), chaos experiments (S7) and
the actuator / CI gate (S8), all feeding one release decision.

    python scripts/demo_final.py          # press Enter between sections
    python scripts/demo_final.py --auto   # no pauses (~30 s)

Everything printed is produced live from the trained model and the shipped
sample inputs under samples/ (python scripts/make_samples.py regenerates them).
The one deliberately HYPOTHETICAL input is scenario C's CVE, and it is labelled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.actuator.actuator import Actuator, DryRunBackend, exit_code_for   # noqa: E402
from src.chaos.experiments import FAULTS, run_chaos                         # noqa: E402
from src.data.dataset import prepare                                        # noqa: E402
from src.models.predict import TriagePredictor                              # noqa: E402
from src.observability.jaeger_adapter import load_jaeger                    # noqa: E402
from src.policy_engine import database as db                                # noqa: E402
from src.policy_engine.pipeline import evaluate                             # noqa: E402
from src.security import reachability as R                                  # noqa: E402

SAMPLES = ROOT / "samples"
AUTO = False
W = 78


def header(t):
    print("\n" + "=" * W + f"\n {t}\n" + "=" * W)


def pause():
    if not AUTO:
        try:
            input("\n    [Press Enter to continue] ")
        except EOFError:
            pass


def spans_of(name):
    _tid, spans, extra = next(iter(load_jaeger(SAMPLES / name)))
    return spans, extra


def span_lists(name):
    return [s for _t, s, _e in load_jaeger(SAMPLES / name)]


def show(label, res, target=None):
    recs = Actuator(DryRunBackend()).execute(res, target=target)
    print(f"\n  {label}")
    print(f"    triage     : {res['classification']} "
          f"(conf {res['confidence']}, OOV {res['oov_rate']})"
          if res["classification"] else "    triage     : (no trace supplied)")
    print(f"    recommended: {res['recommended']}   rule: {res['rule']}")
    print(f"    ACTION     : {res['action']}  (T{res['trust_tier']}, "
          f"autonomous={res['autonomous']})")
    print(f"    reason     : {res['reasons'][0][:100]}")
    print(f"    actuator   : {', '.join(r.step for r in recs if r.executed)}"
          f"   -> CI check exit {exit_code_for(res['action'])}")


def main():
    global AUTO
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true")
    AUTO = ap.parse_args().auto
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    header("FINAL DEMO - S5 observability, S6 security, S7 chaos, S8 actuator")
    predictor = TriagePredictor()
    db.init_db()
    sec = json.loads((ROOT / "reports" / "security_gate.json").read_text())["verdict"]
    pause()

    # ---------------------------------------------------------------- S5 + gate
    header("1. LIVE-TRACE GATE - Jaeger trace in, one release decision out")
    print("  Four situations, all dry-run (nothing is sent anywhere).")

    spans, extra = spans_of("jaeger_hdfs_data_fault.json")
    r = evaluate(predictor, spans=spans, extra=extra, tier=3, security=sec,
                 test_id="A-datafault", log=False)
    show("A. a REAL data-corruption trace, exported in Jaeger's JSON shape", r)

    spans, extra = spans_of("jaeger_foreign_checkout.json")
    r = evaluate(predictor, spans=spans, extra=extra, tier=3, security=sec,
                 test_id="B-foreign", log=False)
    show("B. a trace from a DIFFERENT system (microservices, not HDFS)", r)
    print("    -> the model never saw these operation names, so it abstains")
    print("       instead of guessing, even though the agent is at T3.")

    # the SAME data-corruption trace, as a stock tracer would export it
    doc = json.loads((SAMPLES / "jaeger_hdfs_data_fault.json").read_text())
    doc["data"][0].pop("triage_extra", None)
    for sp in doc["data"][0]["spans"]:
        sp["tags"] = [t for t in sp["tags"] if not t["key"].startswith("triage.")]
    _tid, spans, extra = next(iter(load_jaeger(doc)))
    r = evaluate(predictor, spans=spans, extra=extra, tier=3, security=sec,
                 test_id="B2-stock-export", log=False)
    show("B2. the SAME HDFS trace from A, exported by a stock tracer "
         "(no training-schema task counts)", r)
    print(f"    -> vocabulary is mostly known (OOV {r['oov_rate']:.2f}, under the "
          "0.50 limit), yet the gate abstains on the feature schema:")
    print(f"       {r['signals']['drift']['note'][:150]}")
    fid_path = ROOT / "reports" / "adapter_fidelity.json"
    if fid_path.exists():
        f = json.loads(fid_path.read_text())
        print(f"       measured on {f['n_traces']} held-out traces: accuracy "
              f"corpus {f['corpus']['accuracy']:.0%} | schema-attested export "
              f"{f['rich_export']['accuracy']:.0%} | stock export "
              f"{f['bare_export']['accuracy']:.0%}")
        print(f"       stock export predictions: {f['bare_export']['predicted']}")

    spans, extra = spans_of("jaeger_hdfs_pass.json")
    hypothetical = [R.Finding("fastapi", "x", "HYPOTHETICAL-CVE",
                              severity="high", symbols=[])]
    verdict = R.security_verdict(R.analyse(hypothetical, ROOT))
    r = evaluate(predictor, spans=spans, extra=extra, tier=3, security=verdict,
                 test_id="C-pass+cve", log=False)
    show("C. a HEALTHY trace, plus a HYPOTHETICAL high-severity CVE in fastapi "
         "(which the service really imports)", r)

    bad = evaluate(None, tier=3, baseline_traces=span_lists("canary_baseline.json"),
                   canary_traces=span_lists("canary_bad.json"), log=False,
                   test_id="D-canary-bad")
    show("D1. canary with a 33% error rate vs 1% baseline", bad, target="orders")
    good = evaluate(None, tier=3, baseline_traces=span_lists("canary_baseline.json"),
                    canary_traces=span_lists("canary_good.json"), log=False,
                    test_id="D-canary-good")
    show("D2. canary within tolerance", good)
    pause()

    # ---------------------------------------------------------------- trust gating
    header("2. THE TRUST TIER STILL GOVERNS WHAT THE AGENT MAY DO")
    spans, extra = spans_of("jaeger_hdfs_data_fault.json")
    print("  Same DATA_FAULT trace, three agents:")
    for tier in (0, 1, 3):
        r = evaluate(predictor, spans=spans, extra=extra, tier=tier, security=sec,
                     log=False)
        print(f"    T{tier}: recommended {r['recommended']:<18} -> may execute "
              f"{r['action']}")
    pause()

    # ---------------------------------------------------------------- S6
    header("3. SECURITY GATE - 'flagged' vs 'actually reachable'")
    s = sec["summary"]
    print(f"  pip-audit on this environment flagged {s['flagged']} vulnerabilities "
          f"(jupyter-server, jupyterlab, notebook, pip, tornado, urllib3).")
    print(f"  reachable from the running service: {s['actionable']}")
    print(f"  by reachability: {s['by_reachability']}")
    print(f"  false-positive reduction: {s['false_positive_reduction_pct']}%   "
          f"verdict: {sec['verdict']}")
    print("  (positive control in tests/test_security.py proves the analyzer "
          "does flag packages the service really imports)")
    pause()

    # ---------------------------------------------------------------- S7
    header("4. CHAOS - inject known faults into healthy traces, watch the agent")
    ds, *_ = prepare(str(ROOT / "data/processed/corpus.jsonl.gz"),
                     max_len=predictor.max_len)
    healthy = [r for r in ds["test"].recs if r["cls"] == "PASS"]
    rep = run_chaos(predictor, healthy, n_per_fault=15, start_tier=3,
                    agent_id="demo_final_chaos")
    print(f"  {'fault':<20}{'should be':<13}{'flagged as fault':>17}{'class match':>13}")
    for n, v in rep["by_fault"].items():
        print(f"  {n:<20}{v['truth']:<13}{v['flagged_as_fault_rate']:>16.0%}"
              f"{v['class_match_rate']:>13.0%}")
    print(f"\n  trust engine: started T3, ended T{rep['final_tier']}, "
          f"{len(rep['demotions'])} demotion(s)")
    for d in rep["demotions"][:3]:
        print(f"    #{d['experiment']} ({d['fault']}): T{d['from']}->T{d['to']}  "
              f"{d['reason'][:70]}")
    print("  note: 'control' = untouched healthy traces; its flagged rate is the")
    print("  model's false-alarm rate, the baseline every other row must beat.")
    print("  silent_regression has no trace signature: only the trust engine can")
    print("  catch it, which is the point of that experiment.")
    with db._conn() as c:
        for t in ("decisions", "tier_history", "agent_trust"):
            c.execute(f"DELETE FROM {t} WHERE agent_id='demo_final_chaos'")

    header("DEMO COMPLETE")
    print("  S5  Jaeger adapter + drift guard + canary + /metrics")
    print("  S6  CVE reachability gate (27 flagged -> 0 reachable, with a control)")
    print("  S7  chaos experiments + accuracy-retention demotion")
    print("  S8  actuator (dry-run by default) + CI gate + GitHub workflows")


if __name__ == "__main__":
    main()
