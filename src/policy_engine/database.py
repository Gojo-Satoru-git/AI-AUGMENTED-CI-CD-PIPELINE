"""
Trust state for the policy engine.

Phase 1 kept a single static table of DEVELOPERS with fixed tiers 1-3 that were
never updated - which is not the model the zeroth review promised. The zeroth
review promised a stateful T0-T3 model over the AGENT's own historical
accuracy. `agent_trust`/`decisions`/`tier_history` laid that schema; this
module now implements the S3 promotion/demotion machinery on top of it:

  agent_trust     current tier per agent
  decisions       every decision, with ground truth filled in later
  tier_history    every promotion/demotion with the evidence behind it

`developer_trust` is kept so the Phase 1 simulate_triage.py still runs.

Promotion model
----------------
A raw accuracy ("17/20 correct = 85%") is not a safe promotion signal on
small samples - 17/20 has a 95% confidence interval of roughly [62%, 95%],
so "85%" could really be 65%. We promote on the WILSON SCORE LOWER BOUND of
accuracy instead: the statistically defensible floor of what the agent's true
accuracy is, given the evidence so far. This is the "mathematically sound,
stateful trust mechanism" the project's own problem statement asks for - it
directly punishes small sample sizes, so an agent cannot talk its way into a
higher tier on a lucky streak.

Demotion is asymmetric and immediate: a single REAL_BUG that the agent
classified as something else AND acted on autonomously (a genuine defect
reaching production because the agent waved it through) drops the agent one
tier on the spot, bypassing the promotion statistic entirely. Missing a real
bug is the one error class this project treats as unacceptable at any
tier - see policy.py's own comment on why REAL_BUG gets the hardest weight.
"""
from __future__ import annotations

import math
import os
import sqlite3
import time

# TRUST_DB_PATH lets a container mount the trust state on a persistent volume
DB_PATH = os.environ.get("TRUST_DB_PATH") or os.path.join(
    os.path.dirname(__file__), "trust_tiers.db")

TIERS = {0: "observe-only", 1: "recommend", 2: "act-with-rollback",
         3: "act-autonomously"}


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS developer_trust (
            developer_id TEXT PRIMARY KEY,
            trust_tier   INTEGER NOT NULL);

        CREATE TABLE IF NOT EXISTS agent_trust (
            agent_id     TEXT PRIMARY KEY,
            trust_tier   INTEGER NOT NULL DEFAULT 0,
            updated_at   REAL);

        CREATE TABLE IF NOT EXISTS decisions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id       TEXT NOT NULL,
            test_id        TEXT,
            classification TEXT,
            confidence     REAL,
            action         TEXT,
            rule           TEXT,
            trust_tier     INTEGER,
            autonomous     INTEGER,
            ground_truth   TEXT,          -- filled in when the truth is known
            created_at     REAL);

        CREATE TABLE IF NOT EXISTS tier_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id    TEXT NOT NULL,
            from_tier   INTEGER,
            to_tier     INTEGER,
            evidence    TEXT,
            created_at  REAL);
        """)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(tier_history)")}
        if "after_decision_id" not in cols:
            c.execute("ALTER TABLE tier_history ADD COLUMN after_decision_id INTEGER")
        if not c.execute("SELECT 1 FROM developer_trust LIMIT 1").fetchone():
            c.executemany(
                "INSERT INTO developer_trust VALUES (?,?)",
                [("dev_junior_1", 1), ("dev_mid_2", 2), ("dev_senior_3", 3)])
        if not c.execute("SELECT 1 FROM agent_trust LIMIT 1").fetchone():
            # A new agent starts at T0 - observe only - and earns its way up.
            c.execute("INSERT INTO agent_trust VALUES (?,?,?)",
                      ("triage_agent", 0, time.time()))


def get_developer_tier(developer_id: str) -> int:
    with _conn() as c:
        r = c.execute("SELECT trust_tier FROM developer_trust "
                      "WHERE developer_id=?", (developer_id,)).fetchone()
    return r["trust_tier"] if r else 1


def get_agent_tier(agent_id: str = "triage_agent") -> int:
    with _conn() as c:
        r = c.execute("SELECT trust_tier FROM agent_trust WHERE agent_id=?",
                      (agent_id,)).fetchone()
        if r is None:
            c.execute("INSERT INTO agent_trust VALUES (?,?,?)",
                      (agent_id, 0, time.time()))
            return 0
    return r["trust_tier"]


def set_agent_tier(agent_id: str, tier: int, evidence: str = "") -> None:
    prev = get_agent_tier(agent_id)
    with _conn() as c:
        c.execute("INSERT INTO agent_trust VALUES (?,?,?) "
                  "ON CONFLICT(agent_id) DO UPDATE SET trust_tier=excluded."
                  "trust_tier, updated_at=excluded.updated_at",
                  (agent_id, tier, time.time()))
        boundary = c.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM decisions WHERE agent_id=?",
            (agent_id,)).fetchone()["m"]
        c.execute("INSERT INTO tier_history "
                  "(agent_id, from_tier, to_tier, evidence, created_at, "
                  "after_decision_id) VALUES (?,?,?,?,?,?)",
                  (agent_id, prev, tier, evidence, time.time(), boundary))


# Evidence for promotion and for retention counts only decisions made SINCE THE
# LAST TIER CHANGE. Without this the demotion is cosmetic: the very next correct
# decision re-ran the promotion test over the same last-200 history that had
# earned the tier (e.g. 198/200 correct -> Wilson 0.96) and promoted the agent
# straight back - found when a live run with a defect invisible to the model
# produced 53 tier flips in 50 decisions. Evidence earned at one tier must not be
# recycled to buy the tier back after a failure.
EVIDENCE_SINCE_LAST_CHANGE = True


def _evidence_floor(c, agent_id: str) -> int:
    if not EVIDENCE_SINCE_LAST_CHANGE:
        return 0
    r = c.execute("SELECT after_decision_id FROM tier_history WHERE agent_id=? "
                  "ORDER BY id DESC LIMIT 1", (agent_id,)).fetchone()
    return (r["after_decision_id"] or 0) if r else 0


def log_decision(agent_id, test_id, classification, confidence, action, rule,
                 trust_tier, autonomous) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO decisions (agent_id, test_id, classification, "
            "confidence, action, rule, trust_tier, autonomous, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (agent_id, test_id, classification, confidence, action, rule,
             trust_tier, int(autonomous), time.time()))
        return cur.lastrowid


def record_ground_truth(decision_id: int, ground_truth: str) -> dict:
    """Close the loop on one decision, then run the trust-tier state machine:
    an immediate demotion check, followed by a promotion check. Returns
    whatever tier action (if any) fired, so the caller can surface it."""
    with _conn() as c:
        c.execute("UPDATE decisions SET ground_truth=? WHERE id=?",
                  (ground_truth, decision_id))
        row = c.execute(
            "SELECT agent_id, classification, autonomous, test_id "
            "FROM decisions WHERE id=?", (decision_id,)).fetchone()
    if row is None:
        return {"tier_action": None}

    agent_id = row["agent_id"]
    demoted = maybe_demote_on_missed_bug(
        agent_id, decision_id, row["classification"], ground_truth,
        bool(row["autonomous"]), row["test_id"])
    if demoted:
        return {"tier_action": "demoted", **demoted}

    if RETENTION_ENABLED:
        dropped = evaluate_retention(agent_id)
        if dropped:
            return {"tier_action": "demoted", **dropped}

    promoted = evaluate_promotion(agent_id)
    if promoted and promoted["action"] != "hold":
        return {"tier_action": promoted["action"], **promoted}
    return {"tier_action": None}


# ----------------------------------------------------------------------------
# S3: stateful T0-T3 promotion / demotion
# ----------------------------------------------------------------------------
def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score confidence interval for a proportion
    k/n at confidence level implied by z (1.96 = 95%). Returns 0.0 for n=0.

    This is what "17/20 correct" should actually be judged on: not the point
    estimate 0.85, but the worst accuracy consistent with that evidence at
    95% confidence (~0.64 for 17/20). It naturally demands more evidence
    before trusting a small sample, without a separate min-N rule doing all
    the work.
    """
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - spread) / denom)


# tier -> (min scored decisions required, wilson lower-bound threshold to
# promote INTO the tier above it). Thresholds rise with tier because a wrong
# autonomous action at T3 is more expensive than one at T1.
PROMOTION_THRESHOLDS = {
    0: (15, 0.70),   # T0 -> T1: recommend
    1: (30, 0.85),   # T1 -> T2: act with rollback
    2: (50, 0.93),   # T2 -> T3: act autonomously
}
DEMOTE_ON_MISSED_REAL_BUG = True


def evaluate_promotion(agent_id: str = "triage_agent", last_n: int = 200
                       ) -> dict | None:
    """Check whether `agent_id` has earned the next tier up, on the most
    recent `last_n` ground-truthed decisions. Pure read+maybe-write; safe to
    call after every ground-truth submission."""
    tier = get_agent_tier(agent_id)
    if tier >= 3:
        return {"action": "hold", "reason": "already at T3", "tier": tier}
    min_n, threshold = PROMOTION_THRESHOLDS[tier]

    with _conn() as c:
        floor_id = _evidence_floor(c, agent_id)
        rows = c.execute(
            "SELECT classification, ground_truth FROM decisions "
            "WHERE agent_id=? AND ground_truth IS NOT NULL AND id > ? "
            "ORDER BY id DESC LIMIT ?", (agent_id, floor_id, last_n)).fetchall()
    n = len(rows)
    if n < min_n:
        return {"action": "hold", "reason": f"only {n}/{min_n} scored "
                "decisions at this tier so far", "tier": tier, "n": n}

    k = sum(1 for r in rows if r["classification"] == r["ground_truth"])
    lb = wilson_lower_bound(k, n)
    if lb >= threshold:
        evidence = (f"wilson lower-bound accuracy {lb:.3f} >= {threshold} "
                    f"over last {n} decisions ({k}/{n} correct)")
        set_agent_tier(agent_id, tier + 1, evidence)
        return {"action": "promoted", "from_tier": tier, "to_tier": tier + 1,
                "reason": evidence, "wilson_lower_bound": round(lb, 4),
                "n": n, "correct": k}
    return {"action": "hold", "reason": f"wilson lower-bound {lb:.3f} < "
            f"{threshold} over last {n} decisions ({k}/{n} correct)",
            "tier": tier, "wilson_lower_bound": round(lb, 4), "n": n}


# ----------------------------------------------------------------------------
# Accuracy RETENTION: promotion is evidence that an agent EARNED a tier; this is
# the symmetric check that it still deserves it. Without it, an agent whose
# accuracy collapses (the model drifts, a chaos experiment degrades the system)
# keeps its tier until it happens to miss a REAL_BUG. A tier is kept while the
# Wilson lower bound over the recent window stays above the threshold that
# earned it, minus a hysteresis margin so one bad stretch does not flap tiers.
#
# OFF by default: the replay reported in reports/trust_trajectory.png was
# produced without it, and its result should stay reproducible. Chaos runs and
# deployments opt in (RETENTION_ENABLED = True or TRUST_RETENTION=1).
# ----------------------------------------------------------------------------
RETENTION_ENABLED = os.environ.get("TRUST_RETENTION", "0") == "1"
RETENTION_WINDOW = 50
RETENTION_MIN_N = 30
RETENTION_MARGIN = 0.10


def evaluate_retention(agent_id: str = "triage_agent") -> dict | None:
    tier = get_agent_tier(agent_id)
    if tier <= 0:
        return None
    with _conn() as c:
        floor_id = _evidence_floor(c, agent_id)
        rows = c.execute(
            "SELECT classification, ground_truth FROM decisions "
            "WHERE agent_id=? AND ground_truth IS NOT NULL AND id > ? "
            "ORDER BY id DESC LIMIT ?",
            (agent_id, floor_id, RETENTION_WINDOW)).fetchall()
    n = len(rows)
    if n < RETENTION_MIN_N:
        return None
    k = sum(1 for r in rows if r["classification"] == r["ground_truth"])
    lb = wilson_lower_bound(k, n)
    floor = PROMOTION_THRESHOLDS[tier - 1][1] - RETENTION_MARGIN
    if lb >= floor:
        return None
    evidence = (f"retention check failed: wilson lower-bound {lb:.3f} < "
                f"{floor:.2f} (tier T{tier} floor) over last {n} decisions "
                f"({k}/{n} correct)")
    set_agent_tier(agent_id, tier - 1, evidence)
    return {"from_tier": tier, "to_tier": tier - 1, "reason": evidence,
            "wilson_lower_bound": round(lb, 4), "n": n}


def maybe_demote_on_missed_bug(agent_id: str, decision_id: int,
                               classification: str, ground_truth: str,
                               autonomous: bool, test_id: str | None
                               ) -> dict | None:
    """A REAL_BUG the agent misclassified AND acted on autonomously is a
    defect that reached production on the agent's own authority - the exact
    failure the trust-tier system exists to prevent. Demote one tier
    immediately, independent of the promotion statistic."""
    if not DEMOTE_ON_MISSED_REAL_BUG:
        return None
    if ground_truth != "REAL_BUG" or classification == "REAL_BUG":
        return None
    if not autonomous:
        return None  # a human was in the loop; not the agent's failure alone
    tier = get_agent_tier(agent_id)
    if tier == 0:
        return None            # nothing left to take away; no no-op history rows
    new_tier = max(0, tier - 1)
    evidence = (f"decision #{decision_id} (test {test_id}): agent classified "
               f"a REAL_BUG as {classification!r} and acted on it "
               f"autonomously - immediate demotion")
    set_agent_tier(agent_id, new_tier, evidence)
    return {"from_tier": tier, "to_tier": new_tier, "reason": evidence,
            "decision_id": decision_id}


def agent_scorecard(agent_id: str = "triage_agent", last_n: int = 200) -> dict:
    """Accuracy over decisions whose ground truth is known. S3 turns this into
    the promotion statistic; the endpoint already exposes it."""
    with _conn() as c:
        rows = c.execute(
            "SELECT classification, ground_truth FROM decisions "
            "WHERE agent_id=? AND ground_truth IS NOT NULL "
            "ORDER BY id DESC LIMIT ?", (agent_id, last_n)).fetchall()
    n = len(rows)
    correct = sum(1 for r in rows if r["classification"] == r["ground_truth"])
    return {"agent_id": agent_id, "tier": get_agent_tier(agent_id),
            "tier_name": TIERS.get(get_agent_tier(agent_id)),
            "scored_decisions": n, "correct": correct,
            "accuracy": round(correct / n, 4) if n else None}


if __name__ == "__main__":
    init_db()
    print("initialised", DB_PATH)
