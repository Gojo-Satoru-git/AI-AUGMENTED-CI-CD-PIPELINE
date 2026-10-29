"""
S7: chaos experiments.

Real chaos engineering injects faults into a LIVE system (Chaos Mesh, tc netem,
`docker kill`). That needs a running cluster and is the part only the team can
run (see USER_TODO.md). This module is the offline half, and it is useful on
its own: it injects faults that mirror TraceBench's fault families into
HEALTHY held-out traces, so we can ask the question chaos engineering exists to
answer - *when the system is hurt in a known way, does the agent notice, and
does the trust engine react when it does not?*

Fault catalogue (each mirrors a TraceBench family and carries the class the
triage agent SHOULD output):

    slow_datanode        AN_Net_slowDN / slowHDFS   INFRA_FLAKE
    network_disconnect   AN_Net_disconnectDN         INFRA_FLAKE
    datanode_kill        AN_Proc_killDN              INFRA_FLAKE
    block_corruption     AN_Data_corruptBlk          DATA_FAULT
    readonly_datanode    AN_Sys_readOnlyDN           ENV_FAULT
    silent_regression    COM_Sin_Bug (latent defect) REAL_BUG

`silent_regression` is deliberately the odd one out: a code defect that leaves
NO failure signature in the trace. No trace-based classifier can see it, so the
agent is expected to wave it through - which is exactly the situation the
trust engine's immediate-demotion rule exists for. It tests the safety net, not
the model.

Honest caveat: these are SYNTHETIC injections built from signatures observed in
the real fault traces, not real faulty runs. Detection rates here are a
stress-test of the agent's sensitivity, not a substitute for the real test-set
metrics in reports/evaluation.json.
"""
from __future__ import annotations

import collections
import copy
import random
from dataclasses import dataclass
from typing import Callable

from src.policy_engine import database as db
from src.policy_engine import policy as P

DATANODE_HR = "1"          # host-role code for DataNode spans
DATANODE_AG = "1"          # agent code for Datanode
HUGE_US = 3_005_000_000    # the ~50-minute hang seen in real killDN traces


# ----------------------------------------------------------------------------
# record plumbing
# ----------------------------------------------------------------------------
def _explode(rec):
    return {
        "toks": rec["toks"].split(" "),
        "dur": [int(x) for x in rec["dur"].split(",")],
        "hr": list(rec["hr"]),
        "ag": list(rec["ag"]),
        "dep": [int(x) for x in rec["dep"].split(",")],
    }


def _insert(L, pos, tok, dur, hr=DATANODE_HR, ag=DATANODE_AG, dep=2):
    L["toks"].insert(pos, tok)
    L["dur"].insert(pos, dur)
    L["hr"].insert(pos, hr)
    L["ag"].insert(pos, ag)
    L["dep"].insert(pos, dep)


def _implode(rec, L, extra_lat=0):
    out = copy.deepcopy(rec)
    added = len(L["toks"]) - rec["toks"].count(" ") - 1
    out["toks"] = " ".join(L["toks"])
    out["dur"] = ",".join(map(str, L["dur"]))
    out["hr"] = "".join(L["hr"])
    out["ag"] = "".join(L["ag"])
    out["dep"] = ",".join(map(str, L["dep"]))
    f = out["f"]
    f["n_ev"] = len(L["toks"])
    f["n_tid"] += max(added, 0)
    f["n_edge"] += max(added, 0)
    f["n_rep"] += max(added, 0)
    f["n_err"] = sum(1 for t in L["toks"]
                     if t.rsplit("|", 1)[-1] not in ("OK", "USER"))
    f["n_op"] = len({t.rsplit("|", 1)[0] for t in L["toks"]})
    f["lat"] = f["lat"] + extra_lat
    return out


def _positions(rng, n, k):
    """k insertion points in the back half of the trace (faults surface late)."""
    return sorted(rng.randint(n // 2, max(n // 2, n - 1)) for _ in range(k))


# ----------------------------------------------------------------------------
# injectors: (record, severity, rng) -> record
# ----------------------------------------------------------------------------
def slow_datanode(rec, severity, rng):
    L = _explode(rec)
    factor = 1 + 99 * severity
    grew = 0
    for i, h in enumerate(L["hr"]):
        if h == DATANODE_HR:
            new = int(L["dur"][i] * factor)
            grew = max(grew, new - L["dur"][i])
            L["dur"][i] = new
    return _implode(rec, L, extra_lat=grew)


def network_disconnect(rec, severity, rng):
    L = _explode(rec)
    for i, h in enumerate(L["hr"]):
        if h == DATANODE_HR and rng.random() < 0.5 * severity:
            op = L["toks"][i].rsplit("|", 1)[0]
            L["toks"][i] = f"{op}|IOEXC"
    for pos in reversed(_positions(rng, len(L["toks"]), max(1, round(3 * severity)))):
        _insert(L, pos, "OP:_connect_next_Datanode|IOEXC", 94_067)
    return _implode(rec, L)


def datanode_kill(rec, severity, rng):
    L = _explode(rec)
    k = max(1, round(3 * severity))
    for pos in reversed(_positions(rng, len(L["toks"]), k)):
        _insert(L, pos, "chooseDataNode|OK", HUGE_US)
    for pos in reversed(_positions(rng, len(L["toks"]), 2)):
        _insert(L, pos, "chooseDataNode|IOEXC", 94_067)
    return _implode(rec, L, extra_lat=HUGE_US)


def block_corruption(rec, severity, rng):
    L = _explode(rec)
    for _ in range(max(1, round(4 * severity))):
        pos = rng.randint(len(L["toks"]) // 2, len(L["toks"]))
        for tok in ("reportBadBlocks|OK", "sendBlock|ERR", "OP:_receive_block|ERR"):
            _insert(L, pos, tok, rng.randint(2_000, 40_000))
    return _implode(rec, L)


def readonly_datanode(rec, severity, rng):
    L = _explode(rec)
    for _ in range(max(1, round(3 * severity))):
        pos = rng.randint(len(L["toks"]) // 2, len(L["toks"]))
        for tok in ("OP:_new_BlockReceiver|OTH", "OP:_connect_next_Datanode|OTH",
                    "receiveBlock|OTH"):
            _insert(L, pos, tok, rng.randint(1_000, 20_000))
    return _implode(rec, L)


def silent_regression(rec, severity, rng):
    """A latent defect that changes nothing the trace can see."""
    return copy.deepcopy(rec)


@dataclass(frozen=True)
class Fault:
    name: str
    truth: str
    family: str
    inject: Callable
    detectable_by_trace: bool = True
    needs_datanode_spans: bool = False   # injector only acts on existing DataNode spans


FAULTS = {f.name: f for f in (
    Fault("slow_datanode", "INFRA_FLAKE", "AN_Net_slowDN / slowHDFS", slow_datanode,
          needs_datanode_spans=True),
    Fault("network_disconnect", "INFRA_FLAKE", "AN_Net_disconnectDN", network_disconnect),
    Fault("datanode_kill", "INFRA_FLAKE", "AN_Proc_killDN", datanode_kill),
    Fault("block_corruption", "DATA_FAULT", "AN_Data_corruptBlk", block_corruption),
    Fault("readonly_datanode", "ENV_FAULT", "AN_Sys_readOnlyDN", readonly_datanode),
    Fault("silent_regression", "REAL_BUG", "COM_Sin_Bug (latent defect)",
          silent_regression, detectable_by_trace=False),
)}


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------
def _unsafe(action, autonomous, truth):
    if not autonomous:
        return None
    if action == P.ALLOW and truth != "PASS":
        return "unsafe-merge"
    if action == P.RETRY and truth in P.NEVER_RETRY:
        return "wasted-retry"
    if action == P.BLOCK and truth == "PASS":
        return "false-block"
    if action == P.QUARANTINE and truth != "ENV_FAULT":
        return "wrong-quarantine"
    return None


def run_chaos(predictor, pass_records, faults=None, n_per_fault=30,
              severity=1.0, seed=42, agent_id="chaos_agent", start_tier=3,
              retention=True, log=True):
    """Inject each fault into n_per_fault healthy traces (plus the same number
    of untouched controls), stream everything through triage -> policy -> trust
    engine in shuffled order, and report how the agent coped."""
    rng = random.Random(seed)
    names = list(faults or FAULTS)
    experiments = []
    for name in names:
        fault = FAULTS[name]
        pool = ([r for r in pass_records if DATANODE_HR in r["hr"]]
                if fault.needs_datanode_spans else pass_records)
        for r in rng.sample(pool, min(n_per_fault, len(pool))):
            experiments.append((name, fault.truth, fault.inject(r, severity, rng), r))
    # controls use their own RNG so the sample is identical whatever the
    # injectors consumed - otherwise sweeps compare against a moving baseline
    ctrl_rng = random.Random(seed + 1)
    for r in ctrl_rng.sample(pass_records, min(n_per_fault, len(pass_records))):
        experiments.append(("control", "PASS", r, r))
    rng.shuffle(experiments)

    old_retention = db.RETENTION_ENABLED
    if log:
        db.init_db()
        with db._conn() as c:
            for t in ("decisions", "tier_history", "agent_trust"):
                c.execute(f"DELETE FROM {t} WHERE agent_id=?", (agent_id,))
        db.set_agent_tier(agent_id, start_tier, f"chaos run start at T{start_tier}")
        with db._conn() as c:
            c.execute("DELETE FROM tier_history WHERE agent_id=?", (agent_id,))
        db.RETENTION_ENABLED = retention

    per = {n: collections.defaultdict(int) for n in names + ["control"]}
    preds = {n: collections.Counter() for n in names + ["control"]}
    conf_sum = collections.defaultdict(float)
    unsafe_total = collections.Counter()
    trajectory, demotions = [], []
    tier = start_tier

    try:
        for i, (name, truth, rec, _orig) in enumerate(experiments):
            pred = predictor.predict_record(rec)
            cls, conf = pred["classification"], pred["confidence"]
            if log:
                tier = db.get_agent_tier(agent_id)
            d = P.decide(cls, conf, tier)

            s = per[name]
            s["n"] += 1
            s["exact"] += int(cls == truth)
            s["flagged"] += int(cls != "PASS")
            conf_sum[name] += conf
            preds[name][cls] += 1
            u = _unsafe(d.action, d.autonomous, truth)
            if u:
                s["unsafe"] += 1
                unsafe_total[u] += 1
            if truth == "REAL_BUG" and cls != "REAL_BUG":
                s["missed_real_bug"] += 1

            if log:
                did = db.log_decision(agent_id, f"chaos:{name}:{rec['task']}",
                                      cls, conf, d.action, d.rule, tier,
                                      d.autonomous)
                res = db.record_ground_truth(did, truth)
                new_tier = db.get_agent_tier(agent_id)
                if res.get("tier_action") == "demoted":
                    demotions.append({"experiment": i, "fault": name,
                                      "from": tier, "to": new_tier,
                                      "reason": res.get("reason", "")})
                tier = new_tier
            trajectory.append(tier)
    finally:
        db.RETENTION_ENABLED = old_retention

    by_fault = {}
    for n in names + ["control"]:
        s = per[n]
        cnt = max(s["n"], 1)
        by_fault[n] = {
            "truth": FAULTS[n].truth if n != "control" else "PASS",
            "family": FAULTS[n].family if n != "control" else "healthy trace, untouched",
            "n": s["n"],
            "class_match_rate": round(s["exact"] / cnt, 3),
            "flagged_as_fault_rate": round(s["flagged"] / cnt, 3),
            "mean_confidence": round(conf_sum[n] / cnt, 3),
            "predicted": dict(preds[n]),
            "unsafe_autonomous_actions": s["unsafe"],
            "missed_real_bugs": s["missed_real_bug"],
        }
    return {
        "severity": severity, "n_per_fault": n_per_fault, "seed": seed,
        "start_tier": start_tier, "retention_enabled": retention,
        "final_tier": tier, "tier_trajectory": trajectory,
        "demotions": demotions, "unsafe_actions": dict(unsafe_total),
        "by_fault": by_fault,
    }


def severity_sweep(predictor, pass_records, severities=(0.1, 0.25, 0.5, 1.0),
                   n_per_fault=30, seed=42):
    """Detection (flagged-as-fault) vs fault severity, no DB side effects."""
    out = {}
    for sev in severities:
        rep = run_chaos(predictor, pass_records, n_per_fault=n_per_fault,
                        severity=sev, seed=seed, log=False)
        out[sev] = {n: v["flagged_as_fault_rate"] for n, v in rep["by_fault"].items()}
    return out
