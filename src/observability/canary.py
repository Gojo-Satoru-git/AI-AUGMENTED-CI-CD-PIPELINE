"""
S5: canary health score.

Compares a canary deployment's traces against the stable baseline's traces and
answers: PROMOTE, HOLD, or ROLLBACK.

Design choices, deliberately in the same spirit as the trust engine:
  * ROLLBACK on error rate is gated on Wilson intervals, not raw rates. The
    canary is only called "worse" when the lower bound of ITS error rate is
    above the upper bound of the BASELINE's - i.e. the difference survives the
    sampling noise. A 2% vs 1% blip over 20 requests does not roll anything back.
  * Too little canary traffic never promotes: it returns HOLD ("insufficient
    evidence") rather than guessing.
  * Latency is compared on p95 of per-trace duration, a ratio, so it is
    unit-free.

A trace is "errored" if any of its spans has status ERR/EXC/IOEXC. Input is the
span-dict schema produced by jaeger_adapter (or the triage API's `spans`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.policy_engine.database import wilson_lower_bound

ERROR_STATUSES = {"ERR", "EXC", "IOEXC"}
MIN_CANARY_TRACES = 30
MAX_P95_RATIO = 2.0          # canary p95 more than 2x baseline -> rollback
PROMOTE_SCORE = 0.80


def _trace_errored(spans) -> bool:
    return any(s.get("status") in ERROR_STATUSES for s in spans)


def _trace_latency(spans) -> int:
    # root spans carry the end-to-end time; fall back to the longest span
    roots = [s["duration"] for s in spans if s.get("depth", 0) == 0]
    return max(roots) if roots else max((s["duration"] for s in spans), default=0)


def _p95(values):
    if not values:
        return 0.0
    v = sorted(values)
    return float(v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))])


def _wilson_upper(k: int, n: int) -> float:
    return 1.0 - wilson_lower_bound(n - k, n) if n else 1.0


@dataclass
class CanaryReport:
    verdict: str
    score: float
    reason: str
    n_baseline: int
    n_canary: int
    error_rate_baseline: float
    error_rate_canary: float
    p95_baseline_us: float
    p95_canary_us: float
    p95_ratio: float
    details: dict = field(default_factory=dict)


def canary_health(baseline_traces, canary_traces,
                  min_canary: int = MIN_CANARY_TRACES) -> CanaryReport:
    nb, nc = len(baseline_traces), len(canary_traces)
    kb = sum(_trace_errored(t) for t in baseline_traces)
    kc = sum(_trace_errored(t) for t in canary_traces)
    eb = kb / nb if nb else 0.0
    ec = kc / nc if nc else 0.0
    p95b = _p95([_trace_latency(t) for t in baseline_traces])
    p95c = _p95([_trace_latency(t) for t in canary_traces])
    ratio = (p95c / p95b) if p95b else 1.0

    err_component = 1.0 - min(1.0, max(0.0, ec - eb) / 0.10)
    lat_component = 1.0 - min(1.0, max(0.0, ratio - 1.0) / (MAX_P95_RATIO - 1.0))
    score = round(0.6 * err_component + 0.4 * lat_component, 4)

    significantly_worse = (nb > 0 and nc > 0 and
                           wilson_lower_bound(kc, nc) > _wilson_upper(kb, nb))

    if nc < min_canary:
        verdict, reason = "HOLD", (f"insufficient evidence: {nc} canary traces "
                                   f"< {min_canary} required")
    elif significantly_worse:
        verdict, reason = "ROLLBACK", (
            f"canary error rate {ec:.1%} is significantly above baseline "
            f"{eb:.1%} (Wilson intervals do not overlap)")
    elif ratio > MAX_P95_RATIO:
        verdict, reason = "ROLLBACK", (
            f"canary p95 latency is {ratio:.1f}x baseline (limit "
            f"{MAX_P95_RATIO:.1f}x)")
    elif score >= PROMOTE_SCORE:
        verdict, reason = "PROMOTE", "error rate and latency within tolerance"
    else:
        verdict, reason = "HOLD", f"health score {score} below {PROMOTE_SCORE}"

    return CanaryReport(verdict, score, reason, nb, nc, round(eb, 4),
                        round(ec, 4), p95b, p95c, round(ratio, 3),
                        {"err_component": round(err_component, 3),
                         "lat_component": round(lat_component, 3)})
