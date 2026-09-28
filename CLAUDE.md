# CLAUDE.md — AI-Augmented CI/CD Pipeline (Team 18)

Final-year project, Dept. of IT, MIT Anna University.
"Autonomous AI-Augmented CI/CD Pipeline with Adaptive Trust-Tiering and Chaos Engineering".
Team: Praveen Kumar C, Jaiprakash S, Dharshini T. Guide: Dr. P. Anandha Kumar.

Roadmap: `PHASE2_PLAN.md` (9 workstreams, S0–S8). Component docs: `README.md`.
Complete: **S1** (data), **S2** (triage model), **S3** (stateful trust
tiers), **S4** (root-cause localization). S5–S8 deliberately scoped down or
open — see PHASE2_PLAN.md's "scope calls" section for what was cut and why.

---

## Commands

```bash
pip install -r requirements.txt

# Build corpus (resumable; shards per scenario folder, then merge)
python src/data/build_sequences.py --tracebench "<...>/HDFS_v3_TraceBench/tracebench" \
    --out-dir data/processed --per-class 20000            # add --budget 140 to time-box
python src/data/build_sequences.py --tracebench "<...>" --out-dir data/processed --merge

python src/models/train.py --epochs 20 --batch-size 256 --max-len 256 --emb-dim 48 --hidden 64
python src/models/train.py --shuffle-tokens                # order ablation
python src/models/evaluate.py                              # metrics + baselines -> reports/
python scripts/run_rca_eval.py                              # S4: builds op_latency_baseline.json,
                                                              # top-k localization vs 2 baselines -> reports/
python scripts/replay_trust_trajectory.py                   # S3: replays the test set through the
                                                              # trust engine, T0->T3, then a demotion demo
python run_api.py                                          # http://localhost:8000/docs
pytest -q                                                  # 65 tests (11 skip without the opa CLI)
```

Dataset lives outside the repo at `D:\PROJECTS\7th SEM\dataset\HDFS_v3_TraceBench\tracebench`
(363 scenario folders). `data/` is gitignored — rebuild rather than commit it.

---

## The Phase 1 defect this codebase exists to correct

`data/raw/failure_trace.csv` rows are `TaskID` + exactly **2155 numbers**, and the CSV header is
identical in order to `eventId.json` — each row is an **event-count vector**, not a sequence.
The old pipeline fed those counts into `nn.Embedding` as token IDs and truncated to the first
100 columns. Its reported **88.18%** accuracy equals a one-feature rule on column 0 (~88.2%),
which is why it was frozen across epochs 3–5.

Do not resurrect `src/models/preprocess_data.py` or `src/features/trace_anomaly_lstm.py`.
They are the old path, kept only for reference. The live path is `src/data/build_sequences.py`
-> `src/data/dataset.py` -> `src/features/triage_lstm.py`.

---

## Design decisions that must not be silently reverted

Each of these is load-bearing, and most are covered by a test in `tests/test_triage.py`.

1. **Clock alignment.** An event's `StartTime` is on its own host's clock. Offsets are recovered
   from `edge.csv` (`FatherStartTime` = the father's clock reading when it spawned the child)
   and propagated down the call tree; ties break by depth so a caller precedes its handler.
   All 81,777 traces align fully. Sorting by raw `StartTime` is wrong and drops the root op.
2. **Grouped split by scenario folder.** Never split at trace level — traces in one scenario
   share a fault, cluster size and workload. `prepare()` asserts no TaskID or scenario appears
   in two splits.
3. **`title` is NOT a model input.** Folder names encode workload (`_w_`, `_rw_`, `_rpc_`)
   alongside fault, so Title leaks scenario identity. It stays in the corpus for analysis only.
4. **OpName is normalised** (`spaces -> _`, `| -> /`). OpNames like `OP: connect next Datanode`
   contain spaces and the token stream is space-joined; without this the per-token feature
   arrays silently desynchronise from the token list. This bug shipped once already.
5. **COMPOUND is held out entirely** as a generalisation probe, never trained on.
6. **Trust tiers are AGENT autonomy levels**, not developer seniority: T0 observe-only,
   T1 recommend, T2 act-with-rollback, T3 act-autonomously. Phase 1 stored developer tiers.
7. **Promotion is gated on the Wilson lower-bound of accuracy, not raw accuracy.** 17/20
   correct is 85% but its 95%-confidence floor is ~0.65 — see
   `src/policy_engine/database.py:wilson_lower_bound`. This is what makes the tier system a
   real statistical claim rather than a vibes-based counter, and it's why a small early
   streak doesn't fast-track a promotion.
8. **Demotion on a missed REAL_BUG is immediate and bypasses the promotion statistic** — one
   autonomous action on a misclassified real defect drops the agent one tier on the spot
   (`maybe_demote_on_missed_bug`). Deliberately asymmetric with promotion, which requires
   sustained evidence.
9. **ENV_FAULT was NOT merged into INFRA_FLAKE**, despite that being flagged as an option at
   one point. They map to different actions (`QUARANTINE_NODE` vs `RETRY`) and merging would
   have deleted exactly the operational distinction the project exists to demonstrate. The
   actual fix was softening the class-weight formula (`--weight-power` in `train.py`) — see
   `reports/` for the before/after comparison.
10. **RCA has no true root-cause label to evaluate against** — TraceBench doesn't mark "the"
    faulty span. `src/rca/localize.py` evaluates against a stated proxy (a span whose own
    status is non-OK/USER), not independently-verified ground truth. Say this plainly in any
    report; don't imply the top-k numbers are validated against a labelled root cause.

---

## Measured results (test set, 10,513 traces / 57 held-out scenarios)

Current default checkpoint (`src/models/saved/triage_lstm.pth`) uses `--weight-power 0.5`
(design decision 9); the `power=1.0` run is kept at `*_baseline_backup.*` in `reports/` and
`src/models/saved/` for the comparison below.

| System | macro-F1 | excl. REAL_BUG | acc |
|---|---|---|---|
| **BiLSTM, weight-power 0.5 (current default)** | **0.7464** | **0.6829** | **0.7690** |
| BiLSTM, weight-power 1.0 (previous default) | 0.7297 | 0.6621 | 0.7314 |
| B5 scalars + bag-of-ops, GBM (order-free) | 0.7037 | 0.6838 | 0.7407 |
| B2 task scalars, GBM | 0.6670 | 0.6588 | 0.7115 |
| B4 bag-of-ops TF-IDF, SVM | 0.5789 | 0.5598 | 0.5936 |
| B3 bag-of-ops counts, LR (= Phase 1's representation) | 0.5423 | 0.5174 | 0.5409 |
| B1 majority | 0.0904 | 0.1129 | 0.2918 |

Per-class F1 (current default): PASS .685 · INFRA_FLAKE .756 · DATA_FAULT .809 · ENV_FAULT
.481 · REAL_BUG 1.000. Operational counts on the full test set: 139/10,513 (1.3%) wrongly
retried, **0 REAL_BUG misses**.

Note the order-free B5 baseline (macro-F1 0.7037) still beats the current BiLSTM on
`excl. REAL_BUG` (0.6838 vs 0.6829) by a hair — the "order doesn't carry much signal" finding
below still holds even after the reweighting.

### Two findings to report, not hide

**REAL_BUG's 1.000 is a workload confound.** The three `COM_Sin_Bug` scenarios run a small
metadata workload — median **7 spans on 2 hosts** — versus 88–134 spans on 9–12 hosts for every
other class. Trace length alone separates them, so this is scenario recognition, not bug
detection, and it inflates macro-F1 by ~7 points. Only 3 folders exist (1 per split), so
resplitting cannot fix it. Quote the **excl. REAL_BUG** column as the headline.

**Sequence order carries no usable signal**, confirmed two ways: the order-free B5 baseline beats
the BiLSTM on the honest four classes, and the token-shuffle ablation reaches val macro-F1
0.7552 at epoch 3 against 0.7429 ordered at epoch 3 / 0.7476 ordered best. Scrambling order does
not hurt. This does **not** make the sequence pipeline waste — S4's root-cause localisation needs
the aligned call DAG regardless — it means the *classifier* does not need order.

### Known weaknesses

- **ENV_FAULT is still the weak class, but the failure mode changed.** Softening the class
  weight (design decision 9) traded precision for recall: 0.276→**0.611** precision,
  0.709→0.397 recall, net F1 0.397→**0.481**. It over-triggered before (612 PASS + 736
  DATA_FAULT + 203 INFRA_FLAKE misrouted into it); now it under-catches (only 178+1+31=210
  false positives, but recall dropped). Whether precision or recall matters more here is a
  real policy question — `QUARANTINE_NODE` false positives waste capacity, false negatives
  leave a bad node in rotation — worth a deliberate call, not left as an accident of the loss
  weighting.
- **COMPOUND probe still mostly fails, modestly better**: 15,563 / 20,010 (77.8%, was 82%) of
  multi-fault traces classified PASS.
- **Attention has collapsed**: weight 1.0 on one token, ~0 elsewhere, so `evidence_spans` are
  not yet informative. Does not block S4 (RCA uses the separate z-score method in
  `src/rca/localize.py`, not attention) but should be fixed before those spans are trusted.

---

## Next actions, in order

Done since the table above was last true: ENV_FAULT weight softened (`--weight-power`, see
design decision 9), S3 (Wilson-bound promotion + immediate demotion, `scripts/replay_trust_trajectory.py`
demonstrates T0→T3 then a demotion) and S4 (`src/rca/localize.py`, `scripts/run_rca_eval.py`
— beats random and slowest-span baselines on top-k localisation, see `reports/rca_evaluation.json`).

Still open, in order:

1. Multi-label output instead of 5-way softmax, to attack the COMPOUND failure (82% of
   multi-fault traces still classified PASS).
2. Fix attention collapse (entropy regularisation or multi-head pooling) — not on S4's
   critical path (RCA uses the z-score method, not attention), but still needed before
   `evidence_spans` in the triage API response are trustworthy.
3. Decide how REAL_BUG is reported: shape-matched control, or excluded from the headline
   (currently: excluded).
4. Scoped-down S5–S8 (Jaeger adapter, pip-audit+reachability, chaos script, GH Actions gate)
   — see PHASE2_PLAN.md's scope-call table for what each was cut down to and why.

---

## Class taxonomy and policy

| Class | TraceBench families | Action |
|---|---|---|
| `PASS` | NM_CL, NM_DN | `ALLOW_MERGE` |
| `INFRA_FLAKE` | AN_Net_*, AN_Proc_* | `AUTO_RETRY_PIPELINE` (budget 2, then escalate) |
| `DATA_FAULT` | AN_Data_* | `ESCALATE_TO_INFRA` — never retry |
| `ENV_FAULT` | AN_Sys_* | `QUARANTINE_NODE` |
| `REAL_BUG` | COM_Sin_Bug (real HDFS defects) | `BLOCK_MERGE_ALERT_DEV` |
| `COMPOUND` | COM_Sin_{Sys,Proc,Net,Data}, COM_Mul_AA | held out |

Phase 1 mapped every non-anomalous trace to `"Code Bug" -> BLOCK_MERGE`, so healthy traces were
merge-blocked. Corrected in `src/policy_engine/policy.py`, which is a pure function (no I/O) so
it stays unit-testable and mirrors `src/policy_engine/rego/triage_policy.rego`
(`tests/test_rego_policy.py` checks the two agree, skipped in this environment — no `opa` CLI
installed; see that test's docstring). Confidence below 0.60 always
routes to a human; an agent below an action's required tier can only recommend it.

---

## Conventions

- Keep `src/policy_engine/policy.py` pure — no I/O, no model calls. The decision engine decides;
  the actuator (S8) executes. That separation is what makes the trust tiers meaningful.
- Training is CPU-slow (~6–7 min/epoch for 125k params on 41k traces, 2 cores). `dataset.py`
  has `BucketBatchSampler` — keep length-bucketed batching, random batching pads to the longest
  member and wastes most of the compute.
- Report per-class metrics, never a single accuracy number. That is how Phase 1's 88.18% hid a
  one-feature rule for a whole semester.
- Any new claim in a review deck needs a baseline it beats. `src/models/evaluate.py` holds them.
