"""
S5: Prometheus text-format metrics for the triage service.

Exposed at GET /metrics so Prometheus can scrape the agent and Grafana can plot
it (see infra/prometheus.yml and infra/grafana/). Written by hand in the
exposition format rather than pulling in prometheus_client, because the service
only needs a handful of gauges/counters read straight from the decisions table.

  triage_decisions_total{classification,action}   counter
  triage_decisions_autonomous_total               counter
  agent_trust_tier{agent_id}                      gauge (0-3)
  agent_scored_accuracy{agent_id}                 gauge (0-1, scored decisions only)
"""
from __future__ import annotations

from src.policy_engine import database as db


def _esc(v) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"')


def render_metrics() -> str:
    lines = []
    with db._conn() as c:
        rows = c.execute(
            "SELECT classification, action, COUNT(*) AS n FROM decisions "
            "GROUP BY classification, action").fetchall()
        auto = c.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE autonomous=1"
        ).fetchone()["n"]
        agents = [r["agent_id"] for r in c.execute(
            "SELECT agent_id FROM agent_trust").fetchall()]

    lines += ["# HELP triage_decisions_total Decisions taken, by class and action.",
              "# TYPE triage_decisions_total counter"]
    for r in rows:
        lines.append(
            f'triage_decisions_total{{classification="{_esc(r["classification"])}",'
            f'action="{_esc(r["action"])}"}} {r["n"]}')

    lines += ["# HELP triage_decisions_autonomous_total Decisions executed without a human.",
              "# TYPE triage_decisions_autonomous_total counter",
              f"triage_decisions_autonomous_total {auto}"]

    lines += ["# HELP agent_trust_tier Current trust tier (0 observe .. 3 autonomous).",
              "# TYPE agent_trust_tier gauge"]
    acc_lines = []
    for a in agents:
        card = db.agent_scorecard(a)
        lines.append(f'agent_trust_tier{{agent_id="{_esc(a)}"}} {card["tier"]}')
        if card["accuracy"] is not None:
            acc_lines.append(
                f'agent_scored_accuracy{{agent_id="{_esc(a)}"}} {card["accuracy"]}')
    lines += ["# HELP agent_scored_accuracy Accuracy over decisions with known ground truth.",
              "# TYPE agent_scored_accuracy gauge"] + acc_lines
    return "\n".join(lines) + "\n"
