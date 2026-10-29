"""
minishop - the smallest system that is still worth tracing.

    frontend  GET /checkout | GET /browse
      └─ orders     create_order -> validate -> persist
      └─ inventory  reserve / list
           └─ db    select / insert

Real wall-clock sleeps (millisecond scale, `ms` stretches or shrinks them), real
retries with backoff, real error paths - so the traces have the structure of a
production trace: a transient fault shows up as a failed attempt followed by a
successful retry, a dead host as the same failure repeated, a data fault as a
deterministic error that is NOT retried, a code bug as an exception in business
logic.

Faults mirror the project's classes (src/chaos/experiments.py, TraceBench):

    kind            class         what happens
    slow_db         INFRA_FLAKE   db calls take 2-50x longer (mild cases overlap normal jitter)
    flaky_net       INFRA_FLAKE   db.insert times out intermittently; retries mostly recover
    db_corruption   DATA_FAULT    db.insert hits a bad block (checksum mismatch) for a
                                  fixed share of keys - deterministic, never retried
    host_down       ENV_FAULT     the inventory host refuses connections on every attempt
    code_bug        REAL_BUG      orders.validate raises on some requests (a regression)
    novel_bug       REAL_BUG      a regression in an operation NOT present in the training
                                  data (orders.apply_discount) - used to show what the agent
                                  does with a failure it has never seen
    silent_bug      REAL_BUG      a defect that leaves NO trace signature: the request returns
                                  200 with a wrong answer and is only caught later by a test.
                                  Invisible to any trace-based model by construction - it
                                  exists to show the trust engine, not the classifier, at work

The injector knows whether a fault actually bit THIS request (`trace.fault_fired`),
so labels are exact: a request that never touched the faulty path is PASS even
inside a faulty experiment. That is a deliberate difference from TraceBench, which
labels whole scenarios.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass

from src.live.tracer import Trace

FAULT_CLASS = {"none": "PASS", "slow_db": "INFRA_FLAKE", "flaky_net": "INFRA_FLAKE",
               "db_corruption": "DATA_FAULT", "host_down": "ENV_FAULT",
               "code_bug": "REAL_BUG", "novel_bug": "REAL_BUG",
               "silent_bug": "REAL_BUG"}
RETRIES = 3


@dataclass
class Fault:
    kind: str = "none"
    hit_p: float = 1.0          # share of requests the fault can touch
    severity: float = 1.0       # 0..1: how hard it bites (meaning depends on kind)


class Shop:
    def __init__(self, fault: Fault | None = None, seed: int = 0, ms: float = 1.0):
        self.fault = fault or Fault()
        self.rng = random.Random(seed)
        self.ms = ms

    def set_fault(self, fault: Fault):
        self.fault = fault

    # ----------------------------------------------------------------- helpers
    def _sleep(self, base_ms: float):
        time.sleep(base_ms * self.ms / 1000 * self.rng.uniform(0.7, 1.3))

    def _fire(self, tr: Trace, cls: str):
        tr.fault_fired = tr.fault_fired or cls

    # ------------------------------------------------------------ service calls
    def _inventory(self, tr, parent, op, db_op, dbmult):
        """inventory.<op>: connect (can be refused), then a db read."""
        with tr.span("inventory", f"inventory.{op}", parent=parent) as inv:
            for attempt in range(RETRIES):
                if self.fault.kind == "host_down":
                    with tr.span("inventory", "inventory.connect", parent=inv) as c:
                        self._sleep(1)
                        c.fail("connection refused")
                    self._fire(tr, "ENV_FAULT")
                    self._sleep(2 * (attempt + 1))                 # backoff
                    continue
                with tr.span("db", db_op, parent=inv):
                    self._sleep(3 * dbmult)
                if dbmult > 1:
                    self._fire(tr, "INFRA_FLAKE")
                return True
            inv.fail("inventory unavailable after retries")
            return False

    def _persist(self, tr, parent, key, dbmult):
        with tr.span("orders", "orders.persist", parent=parent) as p:
            for attempt in range(RETRIES):
                transient = False
                with tr.span("db", "db.insert order", parent=p) as d:
                    self._sleep(5 * dbmult)
                    if (self.fault.kind == "flaky_net"
                            and self.rng.random() < 0.05 + 0.5 * self.fault.severity):
                        self._sleep(20)                              # waited for a timeout
                        d.fail("read timeout")
                        transient = True
                    elif (self.fault.kind == "db_corruption"
                          and key % 5 < round(1 + 3 * self.fault.severity)):
                        d.fail("checksum mismatch", **{"block": key})
                        self._fire(tr, "DATA_FAULT")
                        p.fail("block corrupt - not retrying")        # deterministic
                        return False
                if transient:
                    self._fire(tr, "INFRA_FLAKE")
                    self._sleep(2 * (attempt + 1))
                    continue
                if dbmult > 1:
                    self._fire(tr, "INFRA_FLAKE")
                return True
            p.fail("db unreachable after retries")
            return False

    # ------------------------------------------------------------------ requests
    def handle(self, workload: str = "checkout") -> Trace:
        tr = Trace(uuid.UUID(int=self.rng.getrandbits(128)).hex, self.rng)
        dbmult = 1.0
        if (self.fault.kind == "slow_db" and self.rng.random() < self.fault.hit_p):
            dbmult = 2 + 48 * self.fault.severity      # 2x (gray) .. 50x
        ok = True
        with tr.span("frontend", f"GET /{workload}") as root:
            self._sleep(1)
            if workload == "browse":
                ok = self._inventory(tr, root, "list", "db.select catalog", dbmult)
            else:
                with tr.span("orders", "orders.create_order", parent=root) as order:
                    with tr.span("orders", "orders.validate", parent=order) as v:
                        self._sleep(2)
                        if (self.fault.kind == "code_bug"
                                and self.rng.random() < self.fault.hit_p):
                            v.fail("ValueError: quantity must be positive",
                                   **{"exception.type": "ValueError"})
                            self._fire(tr, "REAL_BUG")
                            ok = False
                    if ok and self.fault.kind == "novel_bug":
                        with tr.span("orders", "orders.apply_discount",
                                     parent=order) as dsc:
                            self._sleep(1)
                            if self.rng.random() < self.fault.hit_p:
                                dsc.fail("ValueError: discount exceeds price",
                                         **{"exception.type": "ValueError"})
                                self._fire(tr, "REAL_BUG")
                                ok = False
                    if ok:
                        ok = self._inventory(tr, order, "reserve", "db.select stock",
                                             dbmult)
                    if ok:
                        ok = self._persist(tr, order, self.rng.randrange(1000),
                                           dbmult)
                    if not ok:
                        order.fail("order failed")
            if (self.fault.kind == "silent_bug" and ok
                    and self.rng.random() < self.fault.hit_p):
                self._fire(tr, "REAL_BUG")          # wrong answer, healthy-looking trace
            if self.fault.kind == "host_down" and not ok:
                root.tag("http.status_code", 503)
            else:
                root.tag("http.status_code", 200 if ok else 500)
            if not ok:
                root.fail("upstream failure")
            self._sleep(1)
        tr.ok = ok
        return tr

    @staticmethod
    def label(trace: Trace) -> str:
        return trace.fault_fired or "PASS"
