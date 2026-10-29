"""
Step 1 of the minimal live project: run labelled fault experiments against
minishop and save the traces + a manifest.

    python scripts/live_collect.py                     # ~3 min, 37 experiments x 50 traces
    python scripts/live_collect.py --n 100 --out live_data
    python scripts/live_collect.py --otlp http://127.0.0.1:4318/v1/traces   # also show them in Jaeger

Each experiment is an independent run (own seed and fault strength). Each trace
is labelled by what actually happened to THAT request (src/live/minishop.py): a
request that never touched the faulty path is PASS even inside a faulty
experiment. Experiments are the unit of the train/val/test split (never single
traces), and there are >= 3 of every class, which the grouped split requires.

Next: python scripts/jaeger_to_corpus.py --manifest live_data/manifest.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.minishop import FAULT_CLASS, Fault, Shop      # noqa: E402
from src.live.tracer import push_otlp                       # noqa: E402

from src.live.loop import DEFAULT_MS   # single source of truth for the time scale
                     # are a model input, so a different scale is train/serve skew.

# (name, fault kind, workload)
EXPERIMENTS = (
    [(f"pass-co-{i}", "none", "checkout") for i in (1, 2, 3, 4)]
    + [(f"pass-br-{i}", "none", "browse") for i in (1, 2, 3)]
    + [(f"slow-co-{i}", "slow_db", "checkout") for i in (1, 2, 3)]
    + [(f"slow-br-{i}", "slow_db", "browse") for i in (1, 2, 3)]
    + [(f"flaky-co-{i}", "flaky_net", "checkout") for i in range(1, 7)]
    + [(f"corrupt-co-{i}", "db_corruption", "checkout") for i in range(1, 7)]
    + [(f"down-co-{i}", "host_down", "checkout") for i in (1, 2, 3)]
    + [(f"down-br-{i}", "host_down", "browse") for i in (1, 2, 3)]
    + [(f"bug-co-{i}", "code_bug", "checkout") for i in range(1, 7)]
)
# Why so many: the split is by experiment, and a fault class has sub-types (slow
# db vs flaky network). With only one or two experiments of a sub-type, a split
# can end up holding a kind the model never saw in training - measured: with 3
# INFRA_FLAKE runs the LSTM scored 0.0 F1 on that class. Repeats are cheap.


def run(n, out_dir, seed=7, ms=DEFAULT_MS, otlp=None):
    out = Path(out_dir)
    (out / "exports").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest, totals, push_err = [], collections.Counter(), None
    for name, kind, workload in EXPERIMENTS:
        fault = Fault(kind, hit_p=rng.uniform(0.6, 1.0),
                      severity=rng.uniform(0.05, 1.0))   # includes mild gray failures
        shop = Shop(fault, seed=rng.randrange(1 << 30), ms=ms)
        by_label = collections.defaultdict(list)
        for _ in range(n):
            tr = shop.handle(workload)
            by_label[Shop.label(tr)].append(tr.to_jaeger())
            if otlp and push_err is None:
                push_err = push_otlp(otlp, tr)
        for label, traces in by_label.items():
            fname = f"exports/{name}.{label}.json"
            (out / fname).write_text(json.dumps({"data": traces}))
            manifest.append({"scenario": name, "class": label, "files": [fname]})
            totals[label] += len(traces)
        print(f"  {name:<14} {kind:<14} hit_p={fault.hit_p:.2f} sev={fault.severity:.2f}  "
              + "  ".join(f"{k}={len(v)}" for k, v in sorted(by_label.items())))
    (out / "manifest.json").write_text(json.dumps(
        {"meta": {"ms": ms, "seed": seed, "traces_per_experiment": n},
         "experiments": manifest}, indent=1))
    return totals, push_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="traces per experiment")
    ap.add_argument("--out", default="live_data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ms", type=float, default=DEFAULT_MS)
    ap.add_argument("--otlp", help="also push every trace to this OTLP/HTTP endpoint")
    a = ap.parse_args()
    print(f"collecting {len(EXPERIMENTS)} experiments x {a.n} traces (ms={a.ms})")
    totals, err = run(a.n, a.out, a.seed, a.ms, a.otlp)
    print("\nlabelled traces:", dict(totals), " total", sum(totals.values()))
    if a.otlp:
        print("OTLP push:", "ok" if err is None else f"FAILED ({err}) - file output is unaffected")
    print(f"wrote {a.out}/manifest.json")


if __name__ == "__main__":
    main()
