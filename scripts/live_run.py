"""
The minimal live project, closed loop (every decision printed).

    python scripts/live_run.py                                  # ~25 s, in-process
    python scripts/live_run.py --pause 0.15                     # paced for a projector
    python scripts/live_run.py --otlp http://127.0.0.1:4318/v1/traces   # traces also in Jaeger
    python scripts/live_run.py --api http://127.0.0.1:8000      # drive a running API instead

For a narrated, presentation-ready version of this plus everything around it,
use scripts/showcase.py.

What happens, for every request minishop serves:

    request -> trace -> Jaeger adapter -> triage model -> drift guard -> policy
            -> release gate -> decision (+ dry-run actuation)
            -> the TRUE outcome is fed back -> the trust tier moves

Faults are switched on and off on a schedule (steady -> slow db -> ... -> silent
bug). The agent starts at --start-tier (default T0, observe-only) and has to EARN
autonomy from its own track record; if its accuracy collapses, demotion takes it
back down. Ground truth is exact: the injector knows whether the fault bit.

Needs the model trained by the steps in LIVE_PROJECT.md (default --model-dir is
src/models/saved_live). Use 127.0.0.1, not localhost: on Windows localhost costs
~2 s per call.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.loop import DEFAULT_MS, run_loop        # noqa: E402
from src.live.plot import plot_trajectory             # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="src/models/saved_live")
    ap.add_argument("--agent-id", default="live_agent")
    ap.add_argument("--start-tier", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between requests")
    ap.add_argument("--ms", type=float, default=DEFAULT_MS,
                    help="time scale; must match the one used to collect training data")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--otlp")
    ap.add_argument("--api", help="drive a running API instead of calling in-process")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every phase length")
    ap.add_argument("--quiet", action="store_true", help="only phase summaries")
    ap.add_argument("--out", default="reports/live/live_run.json")
    ap.add_argument("--chart", default=None, help="also write the trajectory PNG here")
    a = ap.parse_args()

    predictor = None
    if not a.api:
        from src.models.predict import TriagePredictor
        mdir = Path(a.model_dir)
        if not (mdir / "triage_lstm.pth").exists():
            sys.exit(f"no model in {mdir}. Train one first (see LIVE_PROJECT.md).")
        predictor = TriagePredictor(ckpt=mdir / "triage_lstm.pth",
                                    vocab_path=mdir / "vocab.json")
        print(f"model {mdir} | feature schema {predictor.feature_schema} | "
              f"classes {predictor.labels}")

    steps, phases_meta, done = [], [], None
    print(f"\n{'#':>4} {'fault':<14} {'truth':<12} {'agent says':<22} {'action':<26} tier")
    print("-" * 92)
    for ev in run_loop(predictor=predictor, api=a.api, agent_id=a.agent_id,
                       start_tier=a.start_tier, ms=a.ms, seed=a.seed, scale=a.scale,
                       otlp=a.otlp, pause=a.pause):
        if ev["type"] == "phase":
            phases_meta.append({"fault": ev["fault"], "n": ev["n"],
                                "first_step": ev["first_step"]})
        elif ev["type"] == "step":
            steps.append(ev)
            if not a.quiet:
                pred = f"{ev['pred'] or '-'} ({ev['conf']:.2f})"
                line = (f"{ev['step']:>4} {ev['fault']:<14} {ev['truth']:<12} {pred:<22} "
                        f"{ev['action']:<26} T{ev['tier']} {'ok ' if ev['ok'] else 'MISS'}")
                if ev["unsafe"]:
                    line += f"  <-- {ev['unsafe']}"
                if ev["tier_action"]:
                    line += (f"  *** {ev['tier_action'].upper()} "
                             f"T{ev['from_tier']}->T{ev['to_tier']}")
                print(line)
        else:
            done = ev

    print("\nphase summary")
    print(f"{'fault':<14}{'n':>4}  {'agent right':>11}  truth mix")
    for ph in done["phases"]:
        print(f"{ph['fault']:<14}{ph['n']:>4}  {ph['correct'] / ph['n']:>10.0%}   {ph['truth']}")
    print(f"\nrequests {done['requests']} | agent correct {done['agent_correct']:.1%} | "
          f"autonomous actuations {done['autonomous_actuations']} | "
          f"unsafe autonomous actions {done['unsafe_autonomous_actions']}")
    print(f"trust tier: started T{done['start_tier']}, ended T{done['end_tier']}; "
          f"{len(done['tier_events'])} change(s)")
    for e in done["tier_events"]:
        print(f"  step {e['step']:>4}: {e['action']:<8} T{e['from']} -> T{e['to']}  "
              f"{e['reason'][:80]}")
    if done["otlp"]:
        print("Jaeger push:", "ok" if done["push_err"] is None
              else f"FAILED ({done['push_err']})")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in done.items()
                               if k not in ("push_err", "otlp")}, indent=2))
    print(f"wrote {out}")
    if a.chart:
        plot_trajectory(steps, phases_meta, a.chart, start_tier=a.start_tier)
        print(f"wrote {a.chart}")


if __name__ == "__main__":
    main()
