"""
Show what a trace of each fault looks like - the first thing to put in front of an
audience, before any model is mentioned.

    python scripts/live_peek.py                 # one trace per fault kind
    python scripts/live_peek.py --kind flaky_net --seed 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.minishop import FAULT_CLASS, Fault, Shop          # noqa: E402
from src.observability.jaeger_adapter import span_status        # noqa: E402

ORDER = ["none", "slow_db", "flaky_net", "db_corruption", "host_down", "code_bug",
         "silent_bug"]


def show(kind, seed):
    shop = Shop(Fault(kind, hit_p=1.0, severity=0.9), seed=seed, ms=0.5)
    # find a request on which the fault actually bit (or any, for "none")
    for _ in range(200):
        tr = shop.handle("checkout")
        if kind == "none" or tr.fault_fired:
            break
    kids, roots = {}, []
    for s in tr.spans:
        par = s["references"][0]["spanID"] if s["references"] else None
        (kids.setdefault(par, []).append(s)) if par else roots.append(s)
    svc = {pid: name for name, pid in tr.services.items()}
    print(f"\n=== fault: {kind:<14} -> label {Shop.label(tr):<12} "
          f"(request {'ok' if tr.ok else 'FAILED'})")

    def walk(s, depth):
        st = span_status(s)
        flag = "" if st in ("OK", "USER") else f"   <-- {st}"
        print(f"  {'  ' * depth}{svc[s['processID']]:<10} {s['operationName']:<24}"
              f"{s['duration'] / 1000:>8.1f} ms{flag}")
        for c in sorted(kids.get(s["spanID"], []), key=lambda x: x["startTime"]):
            walk(c, depth + 1)

    for r in roots:
        walk(r, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=ORDER)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    for kind in ([a.kind] if a.kind else ORDER):
        show(kind, a.seed)
    if a.kind in (None, "silent_bug"):
        print("\nNote silent_bug: labelled REAL_BUG, yet the trace is identical to a healthy one.")


if __name__ == "__main__":
    main()
